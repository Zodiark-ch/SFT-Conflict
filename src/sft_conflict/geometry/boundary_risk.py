"""Selected non-gold winner risk and an exact global position mean.

No full-vocabulary differentiable logits are constructed here. The caller
selects winners from detached current logits; only indexed frozen head rows
move from the source (normally CPU) to the current hidden-state device.
"""

from __future__ import annotations

import math

import torch
import torch.distributed as distributed
import torch.nn.functional as functional
from torch import Tensor
from torch.autograd import Function
from torch.utils.checkpoint import checkpoint

__all__ = ["distributed_global_mean", "winner_risk_rows"]


def _temperature(value: float) -> float:
    try:
        valid = (
            isinstance(value, int | float)
            and not isinstance(value, bool)
            and math.isfinite(value)
            and value > 0
        )
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError("temperature must be a positive finite number")
    return float(value)


def winner_risk_rows(
    current_hidden: Tensor,
    unembedding_weight: Tensor,
    label_token_ids: Tensor,
    winner_token_ids: Tensor,
    unembedding_bias: Tensor | None = None,
    temperature: float = 1.0,
    position_chunk_size: int = 4,
) -> Tensor:
    """Return FP32 ``T * softplus((z_winner - z_gold) / T)`` for every row.

    Only ``current_hidden`` is differentiable, even if the supplied head has
    ``requires_grad=True``. Winners must exclude gold; choosing the true
    non-gold maximum is the caller's responsibility. Equal head rows remain
    legitimate constant risks, including their bias difference. No center,
    S0, crossing, correctness, or EOS gate applies.

    Checkpointing bounds saved selected-head intermediates to one position
    chunk during backward. Empty inputs return a connected empty vector.
    Temperature and actual arithmetic must be finite and representable in
    FP32; invalid/overflowing computations fail rather than returning NaNs.
    """
    temperature = _temperature(temperature)
    if type(position_chunk_size) is not int or position_chunk_size <= 0:
        raise ValueError("position_chunk_size must be a positive integer")
    if (
        not isinstance(current_hidden, Tensor)
        or current_hidden.layout != torch.strided
        or current_hidden.device.type == "meta"
        or current_hidden.ndim != 2
        or not current_hidden.is_floating_point()
        or current_hidden.shape[1] == 0
    ):
        raise ValueError("current_hidden must have floating-point shape [N,H] on a real device")
    count, hidden_size = current_hidden.shape
    if (
        not isinstance(unembedding_weight, Tensor)
        or unembedding_weight.layout != torch.strided
        or unembedding_weight.device.type == "meta"
        or unembedding_weight.ndim != 2
        or unembedding_weight.shape[1] != hidden_size
        or unembedding_weight.shape[0] == 0
        or not unembedding_weight.is_floating_point()
    ):
        raise ValueError("unembedding_weight must have floating-point shape [V,H] matching hidden")
    vocabulary_size = unembedding_weight.shape[0]
    if unembedding_bias is not None and (
        not isinstance(unembedding_bias, Tensor)
        or unembedding_bias.layout != torch.strided
        or unembedding_bias.device.type == "meta"
        or unembedding_bias.shape != (vocabulary_size,)
        or not unembedding_bias.is_floating_point()
    ):
        raise ValueError("unembedding_bias must have floating-point shape [V] on a real device")
    for name, ids in (("label_token_ids", label_token_ids), ("winner_token_ids", winner_token_ids)):
        if (
            not isinstance(ids, Tensor)
            or ids.layout != torch.strided
            or ids.device.type == "meta"
            or ids.shape != (count,)
            or ids.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError(f"{name} must have integer shape [N] on a real device")
        if bool(((ids < 0) | (ids >= vocabulary_size)).any()):
            raise ValueError(f"{name} contains an out-of-vocabulary token")
    labels = label_token_ids.to(device=current_hidden.device, dtype=torch.long)
    winners = winner_token_ids.to(device=current_hidden.device, dtype=torch.long)
    if bool(winners.eq(labels).any()):
        raise ValueError("winner_token_ids must exclude label_token_ids")
    scalar = torch.tensor(temperature, device=current_hidden.device, dtype=torch.float32)
    if not bool(torch.isfinite(scalar) & scalar.gt(0)):
        raise ValueError("temperature must be positive and finite in FP32")

    source_weight = unembedding_weight.detach()
    source_bias = unembedding_bias.detach() if unembedding_bias is not None else None

    def compute_chunk(current: Tensor, gold: Tensor, winner: Tensor) -> Tensor:
        with torch.autocast(device_type=current.device.type, enabled=False):
            hidden = current.float()
            gold_rows = source_weight[gold.to(source_weight.device)].to(
                device=current.device, dtype=torch.float32
            )
            winner_rows = source_weight[winner.to(source_weight.device)].to(
                device=current.device, dtype=torch.float32
            )
            if not bool(
                torch.isfinite(hidden).all()
                & torch.isfinite(gold_rows).all()
                & torch.isfinite(winner_rows).all()
            ):
                raise ValueError("selected hidden and Unembedding rows must be finite in FP32")
            difference = winner_rows - gold_rows
            gap = (hidden * difference).sum(dim=-1)
            if source_bias is not None:
                gold_bias = source_bias[gold.to(source_bias.device)].to(
                    device=current.device, dtype=torch.float32
                )
                winner_bias = source_bias[winner.to(source_bias.device)].to(
                    device=current.device, dtype=torch.float32
                )
                if not bool(torch.isfinite(gold_bias).all() & torch.isfinite(winner_bias).all()):
                    raise ValueError("selected Unembedding biases must be finite in FP32")
                gap = gap + (winner_bias - gold_bias)
            scaled_gap = gap / temperature
            if not bool(torch.isfinite(difference).all() & torch.isfinite(scaled_gap).all()):
                raise ValueError("winner logit gap / temperature must be finite in FP32")
            values = functional.softplus(scaled_gap) * temperature
            if not bool(torch.isfinite(values).all()):
                raise ValueError("winner risk must be finite in FP32")
            return values

    if count == 0:
        return current_hidden.float().sum(dim=-1) * 0.0
    chunks = []
    for start in range(0, count, position_chunk_size):
        selection = slice(start, start + position_chunk_size)
        inputs = current_hidden[selection], labels[selection], winners[selection]
        if torch.is_grad_enabled() and current_hidden.requires_grad:
            chunks.append(checkpoint(compute_chunk, *inputs, use_reentrant=False))
        else:
            chunks.append(compute_chunk(*inputs))
    return torch.cat(chunks, dim=0)


class _DistributedGlobalMean(Function):
    @staticmethod
    def forward(ctx, local_values, compensate_distributed_gradient_average, process_group):
        active = distributed.is_available() and distributed.is_initialized()
        world_size = distributed.get_world_size(process_group) if active else 1
        count = torch.tensor(local_values.numel(), dtype=torch.long, device=local_values.device)
        # Accumulate in FP64 so a finite global mean need not have an FP32-finite
        # unnormalized sum. Return the same FP32 scalar on every rank.
        total = local_values.detach().double().sum()
        if world_size > 1:
            distributed.all_reduce(count, op=distributed.ReduceOp.SUM, group=process_group)
            distributed.all_reduce(total, op=distributed.ReduceOp.SUM, group=process_group)
        global_count = int(count.item())
        ctx.local_count = local_values.numel()
        ctx.gradient_scale = (
            (world_size if compensate_distributed_gradient_average else 1) / global_count
            if global_count
            else 0.0
        )
        return (total / global_count).float() if global_count else local_values.new_zeros(())

    @staticmethod
    def backward(ctx, output_gradient):
        return output_gradient.expand(ctx.local_count) * ctx.gradient_scale, None, None


def distributed_global_mean(
    local_values: Tensor,
    *,
    compensate_distributed_gradient_average: bool = False,
    process_group: distributed.ProcessGroup | None = None,
) -> Tensor:
    """Global sum / global N, with local derivative 1/N (or world_size/N).

    Every rank participates, including zero-row ranks and an all-empty group.
    Backward has no collective: the optional factor compensates the caller's
    subsequent DDP/FSDP gradient average, not the reported global loss. There
    is no extra position mean to apply at the caller.

    For real tensors on process-group-compatible devices, a collective
    preflight propagates a rank-local shape/dtype/finiteness error before any
    rank enters the count/sum reductions. Non-tensors and meta devices cannot
    supply a collective device and are rejected immediately.
    """
    if not isinstance(local_values, Tensor) or local_values.device.type == "meta":
        raise ValueError("local_values must be a real-device tensor with floating-point shape [N]")
    error = None
    if (
        local_values.layout != torch.strided
        or local_values.ndim != 1
        or not local_values.is_floating_point()
    ):
        error = "local_values must have floating-point shape [N] and strided layout"
    elif not bool(torch.isfinite(local_values.detach().float()).all()):
        error = "local_values must be finite in FP32"
    if type(compensate_distributed_gradient_average) is not bool:
        error = "compensate_distributed_gradient_average must be boolean"
    active = distributed.is_available() and distributed.is_initialized()
    world_size = distributed.get_world_size(process_group) if active else 1
    if world_size > 1:
        failed = torch.tensor(int(error is not None), dtype=torch.long, device=local_values.device)
        distributed.all_reduce(failed, op=distributed.ReduceOp.MAX, group=process_group)
        if bool(failed):
            raise ValueError(error or "invalid local_values or mean options on another rank")
    elif error is not None:
        raise ValueError(error)
    with torch.autocast(device_type=local_values.device.type, enabled=False):
        return _DistributedGlobalMean.apply(
            local_values.float(), compensate_distributed_gradient_average, process_group
        )
