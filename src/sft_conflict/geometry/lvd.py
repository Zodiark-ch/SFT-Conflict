"""Local Unembedding LVD margins and weighted-pressure second-moment loss."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.distributed as distributed
import torch.nn.functional as functional
from torch import Tensor
from torch.autograd import Function
from torch.utils.checkpoint import checkpoint

from sft_conflict.config import (
    LVDConfig,
    LVDMomentMode,
    LVDNeighborStrategy,
    LVDPressureMode,
    LVDResponseMode,
)


@dataclass(frozen=True)
class PressureResponse:
    values: Tensor
    regions: Tensor
    exponent_clamp_count: Tensor
    maximum_unclamped_exponent: Tensor


@dataclass(frozen=True)
class LVDPressureResult:
    loss: Tensor
    pressure_vectors: Tensor
    baseline_margins_raw: Tensor
    baseline_margins_safe: Tensor
    current_margins: Tensor
    erosion_ratios: Tensor
    capacity_weights: Tensor
    response: PressureResponse
    safety_core_violation_rate: Tensor
    crossing_rate: Tensor
    tiny_baseline_margin_rate: Tensor
    raw_pressure_vectors: Tensor | None = None


@dataclass(frozen=True)
class PressureSpectrumStatistics:
    sample_count: int
    trace: float
    frobenius_squared: float
    maximum_eigenvalue: float
    effective_rank: float
    normalized_frobenius: float
    maximum_eigenvalue_share: float


def _gather_detached_rows(
    local_rows: Tensor,
    process_group: distributed.ProcessGroup | None = None,
) -> Tensor:
    if not distributed.is_available() or not distributed.is_initialized():
        return local_rows.detach().float()
    world_size = distributed.get_world_size(process_group)
    local_count = torch.tensor([local_rows.shape[0]], dtype=torch.long, device=local_rows.device)
    counts_tensors = [torch.zeros_like(local_count) for _ in range(world_size)]
    distributed.all_gather(counts_tensors, local_count, group=process_group)
    counts = [int(value.item()) for value in counts_tensors]
    maximum = max(counts)
    if maximum == 0:
        return local_rows.detach().new_empty((0, local_rows.shape[1]), dtype=torch.float32)
    padded = local_rows.detach().new_zeros((maximum, local_rows.shape[1]))
    if local_rows.shape[0]:
        padded[: local_rows.shape[0]] = local_rows.detach()
    gathered = [torch.zeros_like(padded) for _ in range(world_size)]
    distributed.all_gather(gathered, padded, group=process_group)
    return torch.cat(
        [rank_rows[:count] for rank_rows, count in zip(gathered, counts, strict=True)],
        dim=0,
    ).float()


@torch.no_grad()
def pressure_spectrum_statistics(
    local_pressure: Tensor,
    *,
    mode: LVDMomentMode,
    process_group: distributed.ProcessGroup | None = None,
    rows_are_global: bool = False,
) -> PressureSpectrumStatistics:
    values = (
        local_pressure.detach().float()
        if rows_are_global
        else _gather_detached_rows(local_pressure, process_group)
    )
    sample_count = values.shape[0]
    if sample_count == 0:
        return PressureSpectrumStatistics(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    if LVDMomentMode(mode) is LVDMomentMode.CENTERED_COVARIANCE:
        values = values - values.mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(values.double().cpu())
    eigenvalues = singular_values.square() / sample_count
    trace = eigenvalues.sum()
    frobenius_squared = eigenvalues.square().sum()
    if float(trace.item()) <= 0:
        effective_rank = torch.zeros((), device=values.device)
        normalized_frobenius = torch.zeros((), device=values.device)
        maximum_share = torch.zeros((), device=values.device)
    else:
        probabilities = eigenvalues / trace
        nonzero = probabilities > 0
        entropy = -(probabilities[nonzero] * probabilities[nonzero].log()).sum()
        effective_rank = entropy.exp()
        normalized_frobenius = frobenius_squared / trace.square()
        maximum_share = eigenvalues.max() / trace
    return PressureSpectrumStatistics(
        sample_count=sample_count,
        trace=float(trace.item()),
        frobenius_squared=float(frobenius_squared.item()),
        maximum_eigenvalue=float(eigenvalues.max().item()),
        effective_rank=float(effective_rank.item()),
        normalized_frobenius=float(normalized_frobenius.item()),
        maximum_eigenvalue_share=float(maximum_share.item()),
    )


def select_lvd_competitors(
    *,
    current_logits: Tensor,
    reference_token_ids: Tensor,
    top_k: int,
    strategy: LVDNeighborStrategy = LVDNeighborStrategy.CURRENT_TOPK,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    if current_logits.ndim != 2:
        raise ValueError("current_logits must have shape [N, vocabulary]")
    sample_count, vocabulary_size = current_logits.shape
    if reference_token_ids.shape != (sample_count,):
        raise ValueError("reference_token_ids must have shape [N]")
    if vocabulary_size <= 1:
        raise ValueError("LVD competitor selection requires at least two vocabulary tokens")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if bool(((reference_token_ids < 0) | (reference_token_ids >= vocabulary_size)).any()):
        raise ValueError("reference_token_ids contains an out-of-vocabulary token")
    competitor_count = min(top_k, vocabulary_size - 1)
    if LVDNeighborStrategy(strategy) is LVDNeighborStrategy.CURRENT_TOPK:
        ranked_logits = current_logits.detach().clone()
        if not bool(torch.isfinite(ranked_logits).all()):
            raise ValueError("current_topk requires finite current logits")
        ranked_logits.scatter_(1, reference_token_ids.unsqueeze(1), float("-inf"))
        competitor_ids = ranked_logits.topk(competitor_count, dim=-1).indices
        return competitor_ids, torch.ones_like(competitor_ids, dtype=torch.bool)
    compact_ids = (
        torch.stack(
            [
                torch.randperm(
                    vocabulary_size - 1,
                    device=current_logits.device,
                    generator=generator,
                )[:competitor_count]
                for _ in range(sample_count)
            ],
            dim=0,
        )
        if sample_count
        else torch.empty((0, competitor_count), dtype=torch.long, device=current_logits.device)
    )
    competitor_ids = compact_ids + compact_ids.ge(reference_token_ids.unsqueeze(1))
    competitor_mask = torch.ones_like(competitor_ids, dtype=torch.bool)
    return competitor_ids, competitor_mask


class _DistributedSecondMomentLoss(Function):
    @staticmethod
    def forward(
        ctx,
        local_pressure: Tensor,
        centered: bool,
        compensate_distributed_gradient_average: bool,
        process_group: distributed.ProcessGroup | None,
    ) -> tuple[Tensor, Tensor]:
        world_size = distributed.get_world_size(process_group)
        rank = distributed.get_rank(process_group)
        local_count = torch.tensor(
            [local_pressure.shape[0]], dtype=torch.long, device=local_pressure.device
        )
        count_rows = [torch.zeros_like(local_count) for _ in range(world_size)]
        distributed.all_gather(count_rows, local_count, group=process_group)
        counts = [int(value.item()) for value in count_rows]
        maximum = max(counts)
        if maximum == 0:
            global_pressure = local_pressure.detach().new_empty(
                (0, local_pressure.shape[1]), dtype=torch.float32
            )
            ctx.rank = rank
            ctx.counts = counts
            ctx.world_size = world_size
            ctx.centered = bool(centered)
            ctx.gradient_scale = world_size if compensate_distributed_gradient_average else 1
            ctx.save_for_backward(global_pressure)
            ctx.mark_non_differentiable(global_pressure)
            return local_pressure.new_zeros(()), global_pressure
        padded = local_pressure.detach().new_zeros((maximum, local_pressure.shape[1]))
        if local_pressure.shape[0]:
            padded[: local_pressure.shape[0]] = local_pressure.detach()
        gathered = [torch.zeros_like(padded) for _ in range(world_size)]
        distributed.all_gather(gathered, padded, group=process_group)
        global_pressure = torch.cat(
            [rank_rows[:count] for rank_rows, count in zip(gathered, counts, strict=True)],
            dim=0,
        ).float()
        ctx.rank = rank
        ctx.counts = counts
        ctx.world_size = world_size
        ctx.centered = bool(centered)
        ctx.gradient_scale = world_size if compensate_distributed_gradient_average else 1
        if global_pressure.shape[0] == 0:
            ctx.save_for_backward(global_pressure)
            ctx.mark_non_differentiable(global_pressure)
            return local_pressure.new_zeros(()), global_pressure
        if centered:
            global_pressure = global_pressure - global_pressure.mean(dim=0, keepdim=True)
        ctx.save_for_backward(global_pressure)
        sample_count = global_pressure.shape[0]
        gram = global_pressure @ global_pressure.transpose(0, 1)
        loss = gram.square().sum() / sample_count**2
        ctx.mark_non_differentiable(global_pressure)
        return loss, global_pressure

    @staticmethod
    def backward(
        ctx,
        output_gradient: Tensor,
        _global_pressure_gradient: Tensor,
    ) -> tuple[Tensor, None, None, None]:
        (global_pressure,) = ctx.saved_tensors
        count = ctx.counts[ctx.rank]
        if global_pressure.shape[0] == 0:
            return global_pressure.new_empty((0, global_pressure.shape[1])), None, None, None
        start = sum(ctx.counts[: ctx.rank])
        local_pressure = global_pressure[start : start + count]
        sample_count = global_pressure.shape[0]
        local_gradient = (
            (4.0 / sample_count**2)
            * (local_pressure @ global_pressure.transpose(0, 1))
            @ global_pressure
        )
        local_gradient = local_gradient * ctx.gradient_scale * output_gradient
        return local_gradient, None, None, None


def distributed_second_moment_frobenius_loss(
    local_pressure: Tensor,
    *,
    mode: LVDMomentMode,
    compensate_distributed_gradient_average: bool = True,
    process_group: distributed.ProcessGroup | None = None,
) -> Tensor:
    loss, _ = distributed_second_moment_frobenius_loss_and_global_pressure(
        local_pressure,
        mode=mode,
        compensate_distributed_gradient_average=compensate_distributed_gradient_average,
        process_group=process_group,
    )
    return loss


def distributed_second_moment_frobenius_loss_and_global_pressure(
    local_pressure: Tensor,
    *,
    mode: LVDMomentMode,
    compensate_distributed_gradient_average: bool = True,
    process_group: distributed.ProcessGroup | None = None,
) -> tuple[Tensor, Tensor]:
    if local_pressure.ndim != 2:
        raise ValueError("local pressure must have shape [local_N, hidden]")
    if not distributed.is_available() or not distributed.is_initialized():
        if local_pressure.shape[0] == 0:
            return local_pressure.sum() * 0.0, local_pressure.detach().float()
        return (
            second_moment_frobenius_loss(local_pressure, mode=mode),
            local_pressure.detach().float(),
        )
    centered = LVDMomentMode(mode) is LVDMomentMode.CENTERED_COVARIANCE
    return _DistributedSecondMomentLoss.apply(
        local_pressure,
        centered,
        compensate_distributed_gradient_average,
        process_group,
    )


def _expm1_linear_tail(exponent: Tensor, exponent_clamp: float) -> Tensor:
    """Continue expm1 tangentially without evaluating an uncapped exponential."""
    cap = exponent.new_tensor(exponent_clamp)
    exponential = torch.expm1(exponent.clamp(max=exponent_clamp))
    excess = (exponent - exponent_clamp).clamp_min(0)
    tail = torch.expm1(cap) + torch.exp(cap) * excess
    # Select branches: adding the clamped exponential and a ReLU tail would
    # double the derivative at equality, where both clamps differentiate to 1.
    return torch.where(exponent > exponent_clamp, tail, exponential)


def four_segment_pressure_response(
    erosion_ratios: Tensor,
    *,
    gamma0: float,
    mu3: float,
    mu4: float,
    exponent_clamp: float,
    valid_mask: Tensor | None = None,
    linear_tail: bool = False,
) -> PressureResponse:
    """Finite four-region penalty; opt in to a C1, bounded-slope linear tail.

    The default retains the legacy hard cap and its exact FP32 arithmetic.
    Neither variant is an infinite barrier at the cell boundary.
    """
    if not 0.0 < gamma0 < 1.0:
        raise ValueError("gamma0 must be in (0, 1)")
    if mu3 <= 0 or mu4 <= mu3:
        raise ValueError("four-segment response requires mu4 > mu3 > 0")
    if exponent_clamp <= 0:
        raise ValueError("exponent_clamp must be positive")
    values = erosion_ratios.float()
    if valid_mask is None:
        valid_mask = torch.ones_like(values, dtype=torch.bool)
    elif valid_mask.shape != values.shape:
        raise ValueError("pressure response valid_mask must match erosion ratios")
    if not bool(valid_mask.any()):
        raise ValueError("pressure response requires at least one valid value")
    boundary = 1.0 / gamma0
    exponent3 = mu3 * (values - 1.0)
    exponent4 = mu4 * (values - boundary)
    maximum = torch.stack(
        (
            exponent3.masked_select(valid_mask).max(),
            exponent4.masked_select(valid_mask).max(),
        )
    ).max()
    clamped3 = exponent3.clamp(max=exponent_clamp)
    clamped4 = exponent4.clamp(max=exponent_clamp)
    response3 = (
        _expm1_linear_tail(exponent3, exponent_clamp) if linear_tail else torch.expm1(clamped3)
    )
    segment3 = 1.0 + response3 / mu3
    boundary_exponent = min(mu3 * (boundary - 1.0), exponent_clamp)
    boundary_slope = torch.exp(
        torch.tensor(boundary_exponent, dtype=torch.float32, device=values.device)
    )
    boundary_value = (
        1.0
        + torch.expm1(torch.tensor(boundary_exponent, dtype=torch.float32, device=values.device))
        / mu3
    )
    response4 = (
        _expm1_linear_tail(exponent4, exponent_clamp) if linear_tail else torch.expm1(clamped4)
    )
    segment4 = boundary_value + boundary_slope * response4 / mu4
    result = torch.where(
        values <= 0,
        torch.zeros_like(values),
        torch.where(values < 1, values, torch.where(values < boundary, segment3, segment4)),
    )
    regions = torch.where(
        values <= 0,
        torch.zeros_like(values, dtype=torch.long),
        torch.where(
            values < 1,
            torch.ones_like(values, dtype=torch.long),
            torch.where(
                values < boundary,
                torch.full_like(values, 2, dtype=torch.long),
                torch.full_like(values, 3, dtype=torch.long),
            ),
        ),
    )
    clamp_count = (
        exponent3.gt(exponent_clamp).logical_and(valid_mask).sum()
        + exponent4.gt(exponent_clamp).logical_and(valid_mask).sum()
    )
    return PressureResponse(result, regions, clamp_count, maximum)


def three_segment_pressure_response(
    erosion_ratios: Tensor,
    *,
    gamma0: float,
    mu3: float,
    exponent_clamp: float,
    core_transition_fraction: float = 0.1,
    valid_mask: Tensor | None = None,
) -> PressureResponse:
    """Zero core, smoothly started linear buffer, exponential/linear crossing.

    With xb=1/gamma0, D=xb-1 and delta=core_transition_fraction*D, the
    buffer is H_delta(rho-1): zero for t<=0, t^2/(2*delta) for 0<t<delta,
    and t-delta/2 thereafter. Crossing adds E_C(mu3*(rho-xb))/mu3 to
    H_delta(D), where E_C is expm1 continued by its tangent beyond C.
    All four joins are C1; the deep-tail derivative is exp(C), never zero.

    Region codes are 0 (rho<=1), 1 (1<rho<xb), and 2 (rho>=xb).
    Diagnostics count only valid crossing exponents. Empty/all-masked inputs
    return zeros; masked nonfinite ratios are removed before arithmetic and
    valid ratios must be finite in FP32. This finite penalty is not a barrier,
    nor does it guarantee a representable fourth-order loss at arbitrary rho.
    """
    for name, value in (
        ("gamma0", gamma0),
        ("mu3", mu3),
        ("exponent_clamp", exponent_clamp),
        ("core_transition_fraction", core_transition_fraction),
    ):
        if (
            not isinstance(value, int | float)
            or isinstance(value, bool)
            or not math.isfinite(value)
        ):
            raise ValueError(f"{name} must be a finite number")
    if not 0.0 < gamma0 < 1.0:
        raise ValueError("gamma0 must be in (0, 1)")
    if mu3 <= 0:
        raise ValueError("mu3 must be positive")
    if not 0.0 < core_transition_fraction <= 1.0:
        raise ValueError("core_transition_fraction must be in (0, 1]")
    if not 0.0 < exponent_clamp <= 60.0:
        raise ValueError("exponent_clamp must be in (0, 60] for three_segment")

    values = erosion_ratios.float()
    if valid_mask is not None:
        if valid_mask.shape != values.shape:
            raise ValueError("pressure response valid_mask must match erosion ratios")
        if valid_mask.dtype != torch.bool:
            raise ValueError("pressure response valid_mask must be boolean")
        values = values.masked_fill(~valid_mask, 0.0)
    if not bool(torch.isfinite(values).all()):
        raise ValueError("valid erosion ratios must be finite in FP32")

    boundary = 1.0 / gamma0
    width = boundary - 1.0
    delta = core_transition_fraction * width
    t = values - 1.0
    in_transition = (t > 0.0) & (t < delta)
    # Bound the unused quadratic even for huge rho; normalize before multiplying
    # instead of squaring. Inactive tiny deltas must not cause 0/0 in backward.
    quadratic_t = torch.where(in_transition, t.clamp(min=0.0, max=delta), 0.0)
    quadratic_delta = torch.where(in_transition, values.new_tensor(delta), 1.0)
    quadratic = (quadratic_t / quadratic_delta) * (0.5 * quadratic_t)
    buffer = torch.where(t < delta, quadratic, t - delta / 2.0)

    crossed = values >= boundary
    distance = torch.where(crossed, values - boundary, 0.0)
    # Split extreme scalar multipliers: converting mu3 itself to FP32 must not
    # turn a zero distance into 0*inf or discard a representable tiny exponent.
    exponent = distance
    scale = mu3
    while scale > 1e30:
        exponent = exponent * 1e30
        scale /= 1e30
    while scale < 1e-30:
        exponent = exponent * 1e-30
        scale /= 1e-30
    exponent = exponent * scale

    # Evaluate expm1(z)/mu3 as distance*exprel(z). A direct division has
    # overflowing backward intermediates for tiny mu3 even when g and g' fit.
    # The order-10 series for z<1 is accurate to FP32 rounding and avoids the
    # removable singularity at zero in BOTH forward and backward.
    capped_exponent = exponent.clamp(max=exponent_clamp)
    small = capped_exponent < 1.0
    small_exponent = torch.where(small, capped_exponent, 0.0)
    relative_exponential = torch.full_like(values, 1.0 / math.factorial(11))
    for order in range(9, -1, -1):
        relative_exponential = relative_exponential * small_exponent + 1.0 / math.factorial(
            order + 1
        )
    safe_exponent = torch.where(small, 1.0, capped_exponent)
    relative_exponential = torch.where(
        small, relative_exponential, torch.expm1(safe_exponent) / safe_exponent
    )
    tailed = exponent > exponent_clamp
    cap_distance = values.new_tensor(exponent_clamp / mu3)
    capped_distance = torch.where(tailed, cap_distance, distance)
    exponential_growth = capped_distance * relative_exponential
    tail_excess = torch.where(tailed, distance - cap_distance, 0.0)
    tail_growth = (
        values.new_tensor(math.expm1(exponent_clamp) / mu3)
        + torch.exp(values.new_tensor(exponent_clamp)) * tail_excess
    )
    # Select, rather than add a zero-valued tail: its large, cancelling
    # derivatives would otherwise erase the unit slope at the cell boundary.
    growth = torch.where(tailed, tail_growth, exponential_growth)
    crossing = width - delta / 2.0 + growth
    result = torch.where(values <= 1.0, 0.0, torch.where(crossed, crossing, buffer))
    regions = torch.where(values <= 1.0, 0, torch.where(crossed, 2, 1))
    clamp_count = exponent.gt(exponent_clamp).sum()
    maximum = exponent.max() if exponent.numel() else values.new_zeros(())
    return PressureResponse(result, regions, clamp_count, maximum)


def pressure_response(
    erosion_ratios: Tensor,
    *,
    mode: LVDResponseMode,
    gamma0: float,
    mu3: float,
    mu4: float,
    exponent_clamp: float,
    valid_mask: Tensor | None = None,
    core_transition_fraction: float = 0.1,
) -> PressureResponse:
    mode = LVDResponseMode(mode)
    if mode is LVDResponseMode.THREE_SEGMENT:
        return three_segment_pressure_response(
            erosion_ratios,
            gamma0=gamma0,
            mu3=mu3,
            exponent_clamp=exponent_clamp,
            core_transition_fraction=core_transition_fraction,
            valid_mask=valid_mask,
        )
    if mode in {LVDResponseMode.FOUR_SEGMENT, LVDResponseMode.FOUR_SEGMENT_LINEAR_TAIL}:
        return four_segment_pressure_response(
            erosion_ratios,
            gamma0=gamma0,
            mu3=mu3,
            mu4=mu4,
            exponent_clamp=exponent_clamp,
            valid_mask=valid_mask,
            linear_tail=mode is LVDResponseMode.FOUR_SEGMENT_LINEAR_TAIL,
        )
    values = erosion_ratios.float()
    if mode is LVDResponseMode.LINEAR:
        response = values.clamp_min(0)
    elif mode is LVDResponseMode.SOFTPLUS:
        response = functional.softplus(values) - functional.softplus(torch.zeros_like(values))
        response = response.clamp_min(0)
    elif mode is LVDResponseMode.HINGE:
        response = (values - 1.0).clamp_min(0)
    else:
        raise ValueError(f"unsupported LVD response mode: {mode}")
    regions = values.gt(0).long()
    zero = torch.zeros((), dtype=torch.long, device=values.device)
    maximum = torch.zeros((), dtype=torch.float32, device=values.device)
    return PressureResponse(response, regions, zero, maximum)


def capacity_weights(
    baseline_margins: Tensor,
    competitor_mask: Tensor,
    *,
    exponent: float,
    epsilon_d: float,
    use_capacity_weights: bool,
) -> Tensor:
    if exponent < 1.0 or epsilon_d <= 0:
        raise ValueError("capacity exponent/epsilon must satisfy p >= 1 and epsilon > 0")
    if baseline_margins.shape != competitor_mask.shape:
        raise ValueError("baseline margins and competitor mask must have equal shapes")
    if bool((competitor_mask.sum(dim=-1) == 0).any()):
        raise ValueError("every pressure row requires at least one valid competitor")
    if use_capacity_weights:
        scores = (baseline_margins.float() + epsilon_d).pow(-exponent)
    else:
        scores = torch.ones_like(baseline_margins, dtype=torch.float32)
    scores = scores.masked_fill(~competitor_mask, 0.0)
    return scores / scores.sum(dim=-1, keepdim=True)


def second_moment_frobenius_loss(
    pressure_vectors: Tensor,
    *,
    mode: LVDMomentMode = LVDMomentMode.UNCENTERED_SECOND_MOMENT,
    implementation: str = "auto",
) -> Tensor:
    if pressure_vectors.ndim != 2 or pressure_vectors.shape[0] == 0:
        raise ValueError("pressure_vectors must have shape [N, hidden] with N > 0")
    values = pressure_vectors.float()
    if LVDMomentMode(mode) is LVDMomentMode.CENTERED_COVARIANCE:
        values = values - values.mean(dim=0, keepdim=True)
    elif LVDMomentMode(mode) is not LVDMomentMode.UNCENTERED_SECOND_MOMENT:
        raise ValueError(f"unsupported LVD moment mode: {mode}")
    if implementation not in {"auto", "explicit", "gram"}:
        raise ValueError("second-moment implementation must be auto, explicit, or gram")
    if implementation == "auto":
        implementation = "gram" if values.shape[0] <= values.shape[1] else "explicit"
    sample_count = values.shape[0]
    if implementation == "gram":
        gram = values @ values.transpose(0, 1)
        return gram.square().sum() / sample_count**2
    moment = values.transpose(0, 1) @ values / sample_count
    return moment.square().sum()


def asinh_pressure_transform(pressure_vectors: Tensor) -> Tensor:
    """Radially compress P to asinh(||P||) with the continuous Jacobian I at zero.

    Preserve dtype for numerical oracles; production pressure is already FP32.
    Both branches must be safe under autograd: avoid sqrt(0)/division by zero
    in the unselected branch, and large powers in the unused Taylor branch.
    No detach/straight-through gradient is used in the objective path.
    """
    squared_norm = pressure_vectors.square().sum(dim=-1, keepdim=True)
    small = squared_norm < 1e-6
    taylor_squared_norm = torch.where(small, squared_norm, 0.0)
    taylor_scale = 1.0 - taylor_squared_norm / 6.0 + 3.0 * taylor_squared_norm.square() / 40.0
    safe_norm = torch.where(small, 1.0, squared_norm).sqrt()
    scale = torch.where(small, taylor_scale, safe_norm.asinh() / safe_norm)
    return pressure_vectors * scale


def weighted_lvd_pressure_loss(
    *,
    current_hidden: Tensor,
    reference_hidden: Tensor,
    reference_weight: Tensor,
    competitor_weight: Tensor,
    baseline_margins: Tensor,
    competitor_mask: Tensor,
    gamma0: float,
    mu3: float,
    mu4: float,
    capacity_exponent: float,
    epsilon_d: float,
    normal_epsilon: float,
    baseline_margin_floor: float,
    baseline_negative_tolerance: float,
    exponent_clamp: float,
    response_mode: LVDResponseMode = LVDResponseMode.FOUR_SEGMENT,
    use_capacity_weights: bool = True,
    pressure_mode: LVDPressureMode = LVDPressureMode.WEIGHTED_SUM,
    moment_mode: LVDMomentMode = LVDMomentMode.UNCENTERED_SECOND_MOMENT,
    core_transition_fraction: float = 0.1,
) -> LVDPressureResult:
    """Form weighted pressure and its moment without altering raw crossing geometry.

    Only the erosion-ratio denominator uses floored d0. The crossing rate still
    tests raw d0-erosion<=0, so tiny/floored pairs can cross before rho reaches
    1/gamma0, even inside the three-segment zero-pressure core (rho<=1).
    Ratio regions and this physical crossing diagnostic need not coincide.
    """
    if current_hidden.shape != reference_hidden.shape or current_hidden.ndim != 2:
        raise ValueError("current/reference hidden states must have equal [N, hidden] shapes")
    expected = (*current_hidden.shape[:1], baseline_margins.shape[1], current_hidden.shape[-1])
    if competitor_weight.shape != expected or reference_weight.shape != current_hidden.shape:
        raise ValueError("Unembedding row tensors do not match hidden/margin dimensions")
    if baseline_margins.shape != competitor_mask.shape:
        raise ValueError("baseline margins and competitor mask must have equal shapes")
    raw_d0 = baseline_margins.float()
    valid_d0 = raw_d0.masked_select(competitor_mask)
    if bool((valid_d0 < -baseline_negative_tolerance).any()):
        minimum = float(valid_d0.min().item())
        raise ValueError(f"baseline LVD margin is materially negative: {minimum}")
    safe_d0 = raw_d0.clamp_min(baseline_margin_floor)
    normals = competitor_weight.float() - reference_weight.float().unsqueeze(1)
    normal_norms = torch.linalg.vector_norm(normals, dim=-1)
    if normal_epsilon <= 0:
        raise ValueError("normal_epsilon must be positive")
    if bool((normal_norms.masked_select(competitor_mask) <= normal_epsilon).any()):
        raise ValueError("reference/competitor Unembedding rows have a degenerate normal")
    unit_normals = normals / normal_norms.clamp_min(normal_epsilon).unsqueeze(-1)
    displacement = current_hidden.float() - reference_hidden.float()
    erosion = (unit_normals * displacement.unsqueeze(1)).sum(dim=-1)
    current_margins = raw_d0 - erosion
    erosion_ratios = erosion / (gamma0 * safe_d0)
    response = pressure_response(
        erosion_ratios,
        mode=response_mode,
        gamma0=gamma0,
        mu3=mu3,
        mu4=mu4,
        exponent_clamp=exponent_clamp,
        valid_mask=competitor_mask,
        core_transition_fraction=core_transition_fraction,
    )
    weights = capacity_weights(
        safe_d0,
        competitor_mask,
        exponent=capacity_exponent,
        epsilon_d=epsilon_d,
        use_capacity_weights=use_capacity_weights,
    )
    coefficients = response.values.masked_fill(~competitor_mask, 0.0) * weights
    pressure_mode = LVDPressureMode(pressure_mode)
    raw_pressure_vectors = None
    if pressure_mode is LVDPressureMode.WEIGHTED_SUM:
        pressure_vectors = (coefficients.unsqueeze(-1) * unit_normals).sum(dim=1)
    elif pressure_mode is LVDPressureMode.ASINH_WEIGHTED_SUM:
        raw_pressure = (coefficients.unsqueeze(-1) * unit_normals).sum(dim=1)
        pressure_vectors = asinh_pressure_transform(raw_pressure)
        raw_pressure_vectors = raw_pressure.detach()  # Reporting only, not the objective input.
    elif pressure_mode is LVDPressureMode.MAX_BOUNDARY:
        maximum = coefficients.argmax(dim=-1)
        rows = torch.arange(len(maximum), device=maximum.device)
        pressure_vectors = coefficients[rows, maximum].unsqueeze(-1) * unit_normals[rows, maximum]
    elif pressure_mode is LVDPressureMode.RAW_DISPLACEMENT:
        pressure_vectors = displacement
    else:
        raise ValueError(f"unsupported LVD pressure mode: {pressure_mode}")
    loss = second_moment_frobenius_loss(pressure_vectors, mode=moment_mode)
    core_violations = (
        erosion_ratios.gt(1)
        if response_mode == LVDResponseMode.THREE_SEGMENT
        else erosion_ratios.ge(1)
    )
    valid_count = competitor_mask.sum().clamp_min(1)
    return LVDPressureResult(
        loss=loss,
        pressure_vectors=pressure_vectors,
        baseline_margins_raw=raw_d0,
        baseline_margins_safe=safe_d0,
        current_margins=current_margins,
        erosion_ratios=erosion_ratios,
        capacity_weights=weights,
        response=response,
        safety_core_violation_rate=(
            core_violations.logical_and(competitor_mask).sum() / valid_count
        ),
        crossing_rate=current_margins.le(0).logical_and(competitor_mask).sum() / valid_count,
        tiny_baseline_margin_rate=(
            raw_d0.le(baseline_margin_floor).logical_and(competitor_mask).sum() / valid_count
        ),
        raw_pressure_vectors=raw_pressure_vectors,
    )


@torch.no_grad()
def validate_cached_lvd_margins(
    *,
    reference_hidden: Tensor,
    reference_weight: Tensor,
    competitor_weight: Tensor,
    baseline_margins: Tensor,
    competitor_mask: Tensor,
    normal_epsilon: float,
    margin_tolerance: float,
    reference_bias: Tensor | None = None,
    competitor_bias: Tensor | None = None,
) -> None:
    """Check d0 without requiring bit-identical FP32 GEMM/reduction algorithms.

    Recording subtracts two FP32 GEMM logits; training uses (Wr-Wk).h.
    Cancellation can exceed a fixed margin tolerance even with identical W/h.
    Retain the strict fast path; recheck only failing pairs in FP64 against the
    standard FP32 forward-error bound gamma_n = n*u/(1-n*u), u=eps/2.
    The bound covers both dot products, subtraction/division and normal norm
    rounding; TF32/quantized arithmetic is not covered. This tolerance-based
    check cannot prove head identity. Cache hashes/model bindings remain
    mandatory at the caller, and the loss still rejects degenerate normals.
    No d0 is changed.
    """
    differences = reference_weight.unsqueeze(1) - competitor_weight
    norms = differences.norm(dim=-1).clamp_min(normal_epsilon)
    numerators = (differences * reference_hidden.unsqueeze(1)).sum(dim=-1)
    if reference_bias is not None and competitor_bias is not None:
        numerators = numerators + reference_bias.unsqueeze(1) - competitor_bias
    recomputed = numerators / norms
    failed = competitor_mask & ~torch.isclose(
        recomputed, baseline_margins, atol=margin_tolerance, rtol=1e-4
    )
    if not bool(failed.any()):
        return

    rows, neighbors = failed.nonzero(as_tuple=True)
    hidden = reference_hidden[rows].double()
    reference = reference_weight[rows].double()
    competitor = competitor_weight[rows, neighbors].double()
    exact_norms = (reference - competitor).norm(dim=-1).clamp_min(normal_epsilon)
    exact_gap = ((reference - competitor) * hidden).sum(dim=-1)
    product_scale = ((reference.abs() + competitor.abs()) * hidden.abs()).sum(dim=-1)
    if reference_bias is not None and competitor_bias is not None:
        ref_bias = reference_bias[rows].double()
        comp_bias = competitor_bias[rows, neighbors].double()
        exact_gap = exact_gap + ref_bias - comp_bias
        product_scale = product_scale + ref_bias.abs() + comp_bias.abs()
    cached = baseline_margins[rows, neighbors].double()
    cached_gap = cached * exact_norms
    unit_roundoff = torch.finfo(torch.float32).eps / 2
    operations = hidden.shape[-1] + 4
    gamma = operations * unit_roundoff / (1 - operations * unit_roundoff)
    roundoff_gap = gamma * (product_scale + cached_gap.abs())
    original_gap_tolerance = (margin_tolerance + 1e-4 * cached.abs()) * exact_norms
    residual = (exact_gap - cached_gap).abs()
    valid = (
        torch.isfinite(residual)
        & torch.isfinite(roundoff_gap)
        & (residual <= original_gap_tolerance + roundoff_gap)
    )
    if not bool(valid.all()):
        error = float((recomputed - baseline_margins).masked_select(failed).abs().max())
        excess = float((residual - original_gap_tolerance - roundoff_gap).max())
        raise ValueError(
            "cached d0 margins do not match the current frozen Unembedding "
            f"(max error={error:.6g}, FP32 roundoff gap bound exceeded by {excess:.6g})"
        )


def chunked_lvd_pressure_loss(
    *,
    current_hidden: Tensor,
    reference_hidden: Tensor,
    unembedding_weight: Tensor,
    reference_token_ids: Tensor,
    competitor_ids: Tensor,
    baseline_margins: Tensor,
    competitor_mask: Tensor,
    config: LVDConfig,
    unembedding_bias: Tensor | None = None,
    margin_tolerance: float | None = None,
) -> LVDPressureResult:
    """Bound [N,K,H] memory in BOTH passes without changing the global moment.

    Gather frozen head rows inside each checkpoint (the source may live on CPU).
    Only hidden states, IDs, [N,K] diagnostics and [N,H] pressure survive forward;
    backward recomputes one chunk's directions. Never average per-chunk losses.
    The robust mode also retains detached [N,H] raw pressure for reporting.
    As in the dense loss, crossing uses raw d0, not the floored ratio regions.
    """
    config.validate()
    if current_hidden.ndim != 2 or len(current_hidden) == 0:
        raise ValueError("chunked LVD requires non-empty [N, hidden] current states")
    source_weight = unembedding_weight.detach()
    source_bias = unembedding_bias.detach() if unembedding_bias is not None else None

    def compute_chunk(current, base, reference_ids, neighbors, margins, mask):
        device = current.device
        source_references = reference_ids.to(source_weight.device)
        source_neighbors = neighbors.to(source_weight.device)
        reference_weight = source_weight[source_references].to(device=device, dtype=torch.float32)
        competitor_weight = source_weight[source_neighbors].to(device=device, dtype=torch.float32)
        if margin_tolerance is not None:
            validate_cached_lvd_margins(
                reference_hidden=base,
                reference_weight=reference_weight,
                competitor_weight=competitor_weight,
                baseline_margins=margins,
                competitor_mask=mask,
                normal_epsilon=config.normal_epsilon,
                margin_tolerance=margin_tolerance,
                reference_bias=(
                    source_bias[source_references].to(device=device, dtype=torch.float32)
                    if source_bias is not None
                    else None
                ),
                competitor_bias=(
                    source_bias[source_neighbors].to(device=device, dtype=torch.float32)
                    if source_bias is not None
                    else None
                ),
            )
        result = weighted_lvd_pressure_loss(
            current_hidden=current,
            reference_hidden=base,
            reference_weight=reference_weight,
            competitor_weight=competitor_weight,
            baseline_margins=margins,
            competitor_mask=mask,
            gamma0=config.gamma0,
            mu3=config.mu3,
            mu4=config.mu4,
            capacity_exponent=config.capacity_exponent,
            epsilon_d=config.epsilon_d,
            normal_epsilon=config.normal_epsilon,
            baseline_margin_floor=config.baseline_margin_floor,
            baseline_negative_tolerance=config.baseline_negative_tolerance,
            exponent_clamp=config.exponent_clamp,
            response_mode=config.response_mode,
            use_capacity_weights=config.use_capacity_weights,
            pressure_mode=config.pressure_mode,
            moment_mode=config.moment_mode,
            core_transition_fraction=config.core_transition_fraction,
        )
        output = (
            result.pressure_vectors,
            result.current_margins.detach(),
            result.erosion_ratios.detach(),
            result.capacity_weights.detach(),
            result.response.values.detach(),
            result.response.regions.detach(),
            result.response.exponent_clamp_count.detach(),
            result.response.maximum_unclamped_exponent.detach(),
        )
        if result.raw_pressure_vectors is not None:
            return (*output, result.raw_pressure_vectors.detach())
        return output

    chunks = []
    for start in range(0, len(current_hidden), config.position_chunk_size):
        selection = slice(start, start + config.position_chunk_size)
        inputs = (
            current_hidden[selection],
            reference_hidden[selection],
            reference_token_ids[selection],
            competitor_ids[selection],
            baseline_margins[selection],
            competitor_mask[selection],
        )
        if torch.is_grad_enabled() and current_hidden.requires_grad:
            chunks.append(checkpoint(compute_chunk, *inputs, use_reentrant=False))
        else:
            chunks.append(compute_chunk(*inputs))
    columns = list(zip(*chunks, strict=True))
    pressure, dnew, ratios, weights, responses, regions = (
        torch.cat(column, dim=0) for column in columns[:6]
    )
    core_violations = (
        ratios.gt(1) if config.response_mode == LVDResponseMode.THREE_SEGMENT else ratios.ge(1)
    )
    valid_count = competitor_mask.sum().clamp_min(1)
    raw = baseline_margins.detach().float()
    return LVDPressureResult(
        loss=second_moment_frobenius_loss(pressure, mode=config.moment_mode),
        pressure_vectors=pressure,
        baseline_margins_raw=raw,
        baseline_margins_safe=raw.clamp_min(config.baseline_margin_floor),
        current_margins=dnew,
        erosion_ratios=ratios,
        capacity_weights=weights,
        response=PressureResponse(
            responses, regions, torch.stack(columns[6]).sum(), torch.stack(columns[7]).max()
        ),
        safety_core_violation_rate=core_violations.logical_and(competitor_mask).sum() / valid_count,
        crossing_rate=dnew.le(0).logical_and(competitor_mask).sum() / valid_count,
        tiny_baseline_margin_rate=(
            raw.le(config.baseline_margin_floor).logical_and(competitor_mask).sum() / valid_count
        ),
        raw_pressure_vectors=torch.cat(columns[8], dim=0) if len(columns) > 8 else None,
    )
