"""Locate, capture, and validate the model's actual Unembedding input."""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as functional
from torch import Tensor, nn
from torch.autograd.function import once_differentiable
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper
from torch.distributed.fsdp import FullyShardedDataParallel


class _LinearWithRowGradients(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, bias, logits):
        ctx.save_for_backward(hidden, weight)
        ctx.logits_dtype = logits.dtype
        ctx.bias_dtype = None if bias is None else bias.dtype
        ctx.set_materialize_grads(False)
        return logits, weight, bias

    @staticmethod
    @once_differentiable
    def backward(ctx, logits_gradient, row_gradient, bias_row_gradient):
        if logits_gradient is None:
            raise RuntimeError("geometry must retain its graph-connected logits path")
        hidden, weight = ctx.saved_tensors
        hidden_gradient = weight_gradient = bias_gradient = None
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            flat_gradient = logits_gradient.reshape(-1, weight.shape[0])
            if ctx.needs_input_grad[0]:
                hidden_gradient = (logits_gradient @ weight.to(ctx.logits_dtype)).to(hidden.dtype)
            if ctx.needs_input_grad[1]:
                weight_gradient = (
                    flat_gradient.t() @ hidden.reshape(-1, hidden.shape[-1]).to(ctx.logits_dtype)
                ).to(weight.dtype)
                if row_gradient is not None:
                    if row_gradient.is_sparse:
                        weight_gradient.index_add_(
                            0, row_gradient._indices()[0], row_gradient._values()
                        )
                    else:
                        weight_gradient.add_(row_gradient)
            if ctx.needs_input_grad[2]:
                bias_gradient = flat_gradient.sum(dim=0).to(ctx.bias_dtype)
                if bias_row_gradient is not None:
                    bias_gradient.add_(bias_row_gradient)
        return hidden_gradient, weight_gradient, bias_gradient, None


def get_unembedding(model: nn.Module) -> nn.Module:
    candidate = model
    visited: set[int] = set()
    getter = getattr(candidate, "get_output_embeddings", None)
    while getter is None and hasattr(candidate, "module") and id(candidate) not in visited:
        visited.add(id(candidate))
        candidate = candidate.module
        getter = getattr(candidate, "get_output_embeddings", None)
    if getter is None:
        raise TypeError(f"{type(model).__name__} does not expose get_output_embeddings()")
    output_embeddings = getter()
    if output_embeddings is None or not hasattr(output_embeddings, "weight"):
        raise TypeError("the model output embedding is missing or has no weight parameter")
    return output_embeddings


def has_tied_input_output_embeddings(model: nn.Module) -> bool:
    input_embeddings = model.get_input_embeddings()
    output_embeddings = get_unembedding(model)
    return input_embeddings.weight.data_ptr() == output_embeddings.weight.data_ptr()


def untie_output_embeddings(model: nn.Module) -> nn.Module:
    output_embeddings = get_unembedding(model)
    if not has_tied_input_output_embeddings(model):
        return output_embeddings
    setter = getattr(model, "set_output_embeddings", None)
    if setter is None:
        raise TypeError(f"{type(model).__name__} cannot replace its tied output embedding")

    detached_output = copy.deepcopy(output_embeddings)
    setter(detached_output)
    if hasattr(model, "config"):
        model.config.tie_word_embeddings = False
    if has_tied_input_output_embeddings(model):
        raise RuntimeError("failed to detach the output embedding from the input embedding")
    return get_unembedding(model)


def _first_tensor(value: Any) -> Tensor | None:
    if isinstance(value, Tensor):
        return value
    if isinstance(value, tuple | list):
        for item in value:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, dict):
        for item in value.values():
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    to_tuple = getattr(value, "to_tuple", None)
    if callable(to_tuple):
        return _first_tensor(to_tuple())
    return None


class UnembeddingInputCapture:
    """Context-managed pre-hook that preserves gradients to selected hidden states."""

    class ForwardStopped(RuntimeError):
        """Internal signal used after the real Unembedding input is captured."""

    def __init__(
        self,
        model: nn.Module,
        positions: Tensor | None = None,
        *,
        stop_forward: bool = False,
        observer: Callable[[Tensor, Tensor | None, Tensor, Tensor], None] | None = None,
    ) -> None:
        self.module = get_unembedding(model)
        while isinstance(self.module, FullyShardedDataParallel):
            self.module = self.module._fsdp_wrapped_module
        self.positions = positions
        self.stop_forward = stop_forward
        self.observer = observer
        self.hidden_states: Tensor | None = None
        self._handle: torch.utils.hooks.RemovableHandle | None = None
        self._observer_handle: torch.utils.hooks.RemovableHandle | None = None

    def _capture(self, _module: nn.Module, args: tuple[Any, ...]) -> None:
        hidden_states = _first_tensor(args)
        if hidden_states is None:
            raise TypeError("the Unembedding pre-hook did not receive a tensor input")
        if self.positions is not None:
            if hidden_states.ndim != 3 or self.positions.ndim != 2:
                raise ValueError(
                    "selected positions require [batch, sequence, hidden] and [batch, k]"
                )
            positions = self.positions.to(hidden_states.device)
            gather_index = positions.unsqueeze(-1).expand(-1, -1, hidden_states.shape[-1])
            hidden_states = hidden_states.gather(dim=1, index=gather_index)
        self.hidden_states = hidden_states
        if self.stop_forward:
            raise self.ForwardStopped()

    def __enter__(self) -> UnembeddingInputCapture:
        if self._handle is not None:
            raise RuntimeError("UnembeddingInputCapture cannot be entered twice")
        self._handle = self.module.register_forward_pre_hook(self._capture)
        if self.observer is not None:

            def observe(module, args, output):
                logits = _first_tensor(output)
                if logits is None or self.hidden_states is None:
                    raise RuntimeError(
                        "dynamic geometry observer requires hidden states and logits"
                    )
                weight, bias = module.weight, getattr(module, "bias", None)
                if torch.is_grad_enabled():
                    linear = module
                    while isinstance(linear, CheckpointWrapper):
                        linear = linear._checkpoint_wrapped_module
                    hidden = _first_tensor(args)
                    if not isinstance(linear, nn.Linear) or not isinstance(output, Tensor):
                        raise TypeError("dynamic geometry requires a linear Tensor-output head")
                    if hidden is None:
                        raise RuntimeError("dynamic geometry requires the full head input")
                    # Both branches wait for the logits FSDP hook, then share one dense
                    # head gradient instead of an out-of-place sparse/dense addition.
                    logits, weight, bias = _LinearWithRowGradients.apply(
                        hidden, weight, bias, logits.detach()
                    )
                self.observer(weight, bias, self.hidden_states, logits)
                return logits

            self._observer_handle = self.module.register_forward_hook(observe)
        return self

    def close(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        if self._observer_handle is not None:
            self._observer_handle.remove()
            self._observer_handle = None

    def __exit__(self, _exc_type: Any, _exc_value: Any, _traceback: Any) -> None:
        self.close()


@dataclass(frozen=True)
class UnembeddingForwardAudit:
    max_absolute_error: float
    argmax_matches: bool
    transform: str
    hidden_shape: tuple[int, ...]


def _known_monotonic_softcap(model: nn.Module, logits: Tensor) -> tuple[Tensor, str] | None:
    config = getattr(model, "config", None)
    for attribute in ("final_logit_softcapping", "logit_softcapping"):
        cap = getattr(config, attribute, None)
        if cap is not None and float(cap) > 0:
            cap_value = float(cap)
            return cap_value * torch.tanh(logits / cap_value), f"{attribute}:tanh"
    return None


@torch.no_grad()
def audit_unembedding_forward(
    model: nn.Module,
    model_inputs: dict[str, Tensor],
    *,
    absolute_tolerance: float = 1e-5,
    relative_tolerance: float = 1e-4,
) -> UnembeddingForwardAudit:
    was_training = model.training
    model.eval()
    try:
        with UnembeddingInputCapture(model) as capture:
            outputs = model(**model_inputs)
        if capture.hidden_states is None:
            raise RuntimeError("the model forward did not invoke its registered Unembedding")
        output_embeddings = get_unembedding(model)
        bias = getattr(output_embeddings, "bias", None)
        linear_logits = functional.linear(
            capture.hidden_states,
            output_embeddings.weight,
            bias,
        )
        model_logits = outputs.logits
        comparable_logits = linear_logits.to(model_logits.dtype)
        transform = "identity"
        if not torch.allclose(
            comparable_logits,
            model_logits,
            atol=absolute_tolerance,
            rtol=relative_tolerance,
        ):
            softcap = _known_monotonic_softcap(model, linear_logits)
            if softcap is None:
                max_error = (comparable_logits.float() - model_logits.float()).abs().max().item()
                raise RuntimeError(
                    "F.linear(captured_hidden, W, b) does not reproduce model logits "
                    f"(max absolute error={max_error:.6g})"
                )
            comparable_logits, transform = softcap
            comparable_logits = comparable_logits.to(model_logits.dtype)
            if not torch.allclose(
                comparable_logits,
                model_logits,
                atol=absolute_tolerance,
                rtol=relative_tolerance,
            ):
                raise RuntimeError(
                    "the declared monotonic logit soft-cap does not reproduce model logits"
                )

        argmax_matches = bool(
            torch.equal(linear_logits.argmax(dim=-1), model_logits.argmax(dim=-1))
        )
        if not argmax_matches:
            raise RuntimeError(
                "linear Unembedding logits and model logits have different argmax tokens"
            )
        max_absolute_error = (comparable_logits.float() - model_logits.float()).abs().max().item()
        return UnembeddingForwardAudit(
            max_absolute_error=max_absolute_error,
            argmax_matches=argmax_matches,
            transform=transform,
            hidden_shape=tuple(capture.hidden_states.shape),
        )
    finally:
        model.train(was_training)
