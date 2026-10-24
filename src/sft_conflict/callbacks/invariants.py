"""Fail-fast checks for frozen/trainable parameter drift."""

from __future__ import annotations

from collections.abc import Mapping

from torch import nn
from torch.optim import Optimizer
from transformers import TrainerCallback


def canonical_parameter_name(name: str) -> str:
    wrappers = ("_fsdp_wrapped_module", "_checkpoint_wrapped_module")
    for wrapper in wrappers:
        name = name.replace(f".{wrapper}", "").removeprefix(f"{wrapper}.")
    return name


def parameter_requires_grad_state(model: nn.Module) -> dict[str, bool]:
    state: dict[str, bool] = {}
    for name, parameter in model.named_parameters():
        canonical_name = canonical_parameter_name(name)
        if canonical_name in state:
            raise RuntimeError(f"duplicate canonical parameter name: {canonical_name}")
        state[canonical_name] = parameter.requires_grad
    return state


def assert_optimizer_matches_trainable_parameters(model: nn.Module, optimizer: Optimizer) -> None:
    trainable_ids = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    optimizer_ids = {
        id(parameter) for group in optimizer.param_groups for parameter in group["params"]
    }
    missing = trainable_ids - optimizer_ids
    unexpected = optimizer_ids - trainable_ids
    if missing or unexpected:
        raise RuntimeError(
            "optimizer parameter groups do not match requires_grad state "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )


class ParameterInvariantCallback(TrainerCallback):
    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.expected_requires_grad: Mapping[str, bool] = parameter_requires_grad_state(model)
        self.expected_trainable_numel = sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        )

    def _is_valid_flattened_full_ft(self) -> bool:
        if not self.expected_requires_grad or not all(self.expected_requires_grad.values()):
            return False
        flat_parameters = [
            parameter
            for name, parameter in self.model.named_parameters()
            if canonical_parameter_name(name).endswith("_flat_param")
        ]
        if not flat_parameters or any(not parameter.requires_grad for parameter in flat_parameters):
            return False
        flattened_numel = sum(
            getattr(parameter, "_unpadded_unsharded_size", parameter.shape).numel()
            for parameter in flat_parameters
        )
        if flattened_numel != self.expected_trainable_numel:
            raise RuntimeError(
                "FSDP flattened parameter count changed "
                f"(expected={self.expected_trainable_numel}, current={flattened_numel})"
            )
        return True

    def validate(self, optimizer: Optimizer | None) -> None:
        current = parameter_requires_grad_state(self.model)
        if current != self.expected_requires_grad and not self._is_valid_flattened_full_ft():
            changed = sorted(
                name
                for name in set(current) | set(self.expected_requires_grad)
                if current.get(name) != self.expected_requires_grad.get(name)
            )
            raise RuntimeError(f"requires_grad state changed during training: {changed[:10]}")
        if optimizer is not None:
            assert_optimizer_matches_trainable_parameters(self.model, optimizer)

    def on_step_end(self, args, state, control, optimizer=None, **kwargs):
        self.validate(optimizer)
        return control
