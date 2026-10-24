"""Label-site LVD+ pressure, independent of reference states and d0 caches.

Pressure is an input to the existing OUTER second-moment loss, not a prescribed
gradient of that loss. Only the current hidden states are differentiable; the
anchor center (legacy W_j/2 or recorded S0_j) and the head geometry are frozen.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from sft_conflict.config import LVDPlusAnchorMode

if TYPE_CHECKING:
    from sft_conflict.config import LVDPlusConfig


@dataclass(frozen=True)
class LVDPlusResponse:
    values: Tensor
    regions: Tensor
    exponent_clamp_count: Tensor
    maximum_unclamped_exponent: Tensor


@dataclass(frozen=True)
class LVDPlusPressureResult:
    """Differentiable [N,H] pressure and detached geometry diagnostics.

    Distances, margins, response values and regions have shape [N,K]. The full
    L2 distance ||h-c_j|| has shape [N], not a directional/normalized distance.
    Masked pairs (and rows without competitors) have zero diagnostics.
    In recorded-S0 mode, exact duplicate head rows also have zero pair
    diagnostics: no h-dependent boundary exists, even with unequal biases.
    The detached [N,K] degenerate_pair_mask identifies these selected pairs;
    None preserves the legacy half-Unembedding result contract.
    """

    pressure_vectors: Tensor
    signed_distances: Tensor
    center_margins: Tensor
    current_margins: Tensor
    response: LVDPlusResponse
    center_distance_norms: Tensor
    degenerate_pair_mask: Tensor | None = None


def _positive_finite(name: str, value: float) -> None:
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
        raise ValueError(f"{name} must be a positive finite number")


def _validate_response_parameters(rate: float, width: float, cap: float) -> None:
    _positive_finite("exponential_rate", rate)
    _positive_finite("transition_width", width)
    _positive_finite("exponent_clamp", cap)
    if cap > 20.0:
        raise ValueError("exponent_clamp must be in (0, 20] for LVD+")


def _validate_post_crossing_slope_gain(value: float) -> None:
    _positive_finite("post_crossing_slope_gain", value)
    if not 1.0 <= value <= 1000.0:
        raise ValueError("post_crossing_slope_gain must be in [1, 1000]")


def _scale(value: Tensor, scalar: float, *, divide: bool = False) -> Tensor:
    """Apply a finite positive scalar without first rounding it to FP32 inf/0."""
    while scalar > 1e30:
        value = value / 1e30 if divide else value * 1e30
        scalar /= 1e30
    while scalar < 1e-30:
        value = value / 1e-30 if divide else value * 1e-30
        scalar /= 1e-30
    return value / scalar if divide else value * scalar


def _exponential_excess(distance: Tensor, exponent: Tensor, rate: float) -> Tensor:
    """(expm1(mu*u)-mu*u)/mu without cancellation or singular small-z backward."""
    small = exponent < 1.0
    z = torch.where(small, exponent, 0.0)
    series = torch.full_like(z, 1.0 / math.factorial(11))
    for order in range(9, 0, -1):
        series = series * z + 1.0 / math.factorial(order + 1)
    # g=mu*u^2*(1/2+z/6+...). Balance the scale before multiplying: u*z
    # would lose a representable g when z itself underflows in FP32.
    scaled_u = _scale(torch.where(small, distance, 0.0), math.sqrt(rate))
    small_growth = (scaled_u * series) * scaled_u
    safe_z = torch.where(small, 1.0, exponent)
    large_u = torch.where(small, 0.0, distance)
    large_growth = large_u * ((torch.expm1(safe_z) - safe_z) / safe_z)
    return torch.where(small, small_growth, large_growth)


def lvd_plus_pressure_response(
    signed_distances: Tensor,
    center_margins: Tensor,
    *,
    exponential_rate: float,
    transition_width: float,
    exponent_clamp: float,
    valid_mask: Tensor | None = None,
    post_crossing_slope_gain: float = 1.0,
) -> LVDPlusResponse:
    """C1 scalar directional response in FP32, including nonpositive margins.

    For s>0, q=H_tau(s)+(E_C(mu*u)-mu*u)/mu+(c-1)*H_tau(u), where
    u=max(s-max(b,0),0) and c=post_crossing_slope_gain. H_tau is zero at zero,
    quadratic on (0,tau), then linear; E_C continues expm1 by its tangent
    after C. For s<=0, q and its derivative are exactly zero, even if b-s<=0.
    Region precedence is 0: s<=0; 1: s>0,b-s>0; 2: s>0,b-s<=0, for every c.
    Once s>=tau and u>=tau, q'=exp(min(mu*u,C))+c-1. Gain 1 retains the exact
    legacy tensor graph; larger gains leave region 1 values/slopes unchanged.

    b is fixed geometry, never floored or used as a distance denominator.
    Masked nonfinite inputs are ignored; valid inputs must be finite in FP32.
    Counts/maxima are detached and concern only active excess u. An unbounded
    raw exponent may overflow the FP32 diagnostic while its tangent response
    remains finite. This is not an infinite barrier or an overflow guarantee
    for the caller's fourth-order pressure-moment loss. Increasing pairwise q
    does not guarantee a larger outer loss or corrected classifications.
    """
    _validate_response_parameters(exponential_rate, transition_width, exponent_clamp)
    _validate_post_crossing_slope_gain(post_crossing_slope_gain)
    if signed_distances.shape != center_margins.shape:
        raise ValueError("signed_distances and center_margins must have matching shapes")
    if signed_distances.device != center_margins.device:
        raise ValueError("signed_distances and center_margins must be on the same device")
    if signed_distances.is_complex() or center_margins.is_complex():
        raise ValueError("signed_distances and center_margins must be real")
    if valid_mask is not None:
        if valid_mask.shape != signed_distances.shape:
            raise ValueError("valid_mask must match signed_distances")
        if valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must be boolean")
        if valid_mask.device != signed_distances.device:
            raise ValueError("valid_mask must be on the signed_distances device")

    with torch.autocast(device_type=signed_distances.device.type, enabled=False):
        s = signed_distances.float()
        b = center_margins.detach().float()
        if valid_mask is not None:
            s = s.masked_fill(~valid_mask, 0.0)
            b = b.masked_fill(~valid_mask, 0.0)
        if not bool(torch.isfinite(s).all() & torch.isfinite(b).all()):
            raise ValueError("valid signed_distances and center_margins must be finite in FP32")

        positive_s = s.clamp_min(0.0)
        in_transition = (s > 0.0) & (s < transition_width)
        transition_s = torch.where(in_transition, s, 0.0)
        # Balance the quadratic to preserve representable values even if s/tau
        # underflows. No unused huge square or FP32-rounded zero denominator.
        scaled_s = _scale(transition_s, math.sqrt(transition_width), divide=True)
        quadratic = (0.5 * scaled_s) * scaled_s
        linear_offset = torch.where(
            s >= transition_width, s.new_tensor(transition_width / 2.0), 0.0
        )
        smooth_linear = torch.where(in_transition, quadratic, positive_s - linear_offset)

        u = (positive_s - b.clamp_min(0.0)).clamp_min(0.0)
        exponent = _scale(u, exponential_rate)
        cap_distance = s.new_tensor(exponent_clamp / exponential_rate)
        # Compare in physical units as well: mu*u and C can both round to zero
        # although u>C/mu and the tail contribution is representable.
        tailed = u > cap_distance
        capped_exponent = torch.where(tailed, s.new_tensor(exponent_clamp), exponent)
        capped_u = torch.where(tailed, cap_distance, u)
        exponential_growth = _exponential_excess(capped_u, capped_exponent, exponential_rate)

        # Work in physical u: computing (mu*u-C)/mu can be inf/inf. Evaluate
        # the cap constant in scalar precision before its final FP32 cast; a
        # premature C^2 or FP32 C can otherwise erase a representable constant.
        cap_relative = (
            sum(exponent_clamp**order / math.factorial(order + 1) for order in range(1, 11))
            if exponent_clamp < 1.0
            else (math.expm1(exponent_clamp) - exponent_clamp) / exponent_clamp
        )
        cap_growth = s.new_tensor((exponent_clamp / exponential_rate) * cap_relative)
        tail_excess = torch.where(tailed, u - cap_distance, 0.0)
        tail_growth = cap_growth + _scale(tail_excess, math.expm1(exponent_clamp))
        # Select rather than add cancelling capped/tail derivatives, which
        # would lose the small correct slope when exp(C) is large.
        growth = torch.where(tailed, tail_growth, exponential_growth)
        values = torch.where(s > 0.0, smooth_linear + growth, 0.0)
        if post_crossing_slope_gain != 1.0:
            # Gate before scaling: inactive pairs must have zero derivative,
            # including u=0 at the crossing and masked/nonpositive s.
            active_boost = (s > 0.0) & (u > 0.0)
            boost_u = torch.where(active_boost, u, 0.0)
            boost_transition = active_boost & (boost_u < transition_width)
            scaled_boost = _scale(
                torch.where(boost_transition, boost_u, 0.0),
                math.sqrt(transition_width),
                divide=True,
            )
            boost_quadratic = (0.5 * scaled_boost) * scaled_boost
            # Form tau/2 in Python, just as in the legacy H_tau(s) path.
            boost_offset = torch.where(
                boost_u >= transition_width, s.new_tensor(transition_width / 2.0), 0.0
            )
            boost = torch.where(boost_transition, boost_quadratic, boost_u - boost_offset)
            values = values + (post_crossing_slope_gain - 1.0) * boost
        # Compare directly to b to avoid overflowing the subtraction b-s just
        # to determine its sign. The returned geometry retains the raw b-s.
        regions = torch.where(s <= 0.0, 0, torch.where(s < b, 1, 2))
        raw_exponent = exponent.detach()
        maximum = raw_exponent.max() if raw_exponent.numel() else s.new_zeros(())
        return LVDPlusResponse(values, regions, tailed.sum(), maximum)


@torch.no_grad()
def _label_site_geometry(
    current: Tensor,
    source_weight: Tensor,
    source_bias: Tensor | None,
    labels: Tensor,
    neighbors: Tensor,
    mask: Tensor,
    normal_epsilon: float,
    *,
    anchor_centers: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
    """Index only one chunk of the frozen head and form unfloored unit geometry."""
    active_rows = mask.any(dim=-1)
    safe_labels = labels.masked_fill(~active_rows, 0)
    safe_neighbors = neighbors.masked_fill(~mask, 0)
    label_weight = source_weight[safe_labels.to(source_weight.device)].to(
        device=current.device, dtype=torch.float32
    )
    competitor_weight = source_weight[safe_neighbors.to(source_weight.device)].to(
        device=current.device, dtype=torch.float32
    )
    label_weight = label_weight.masked_fill(~active_rows.unsqueeze(-1), 0.0)
    competitor_weight = competitor_weight.masked_fill(~mask.unsqueeze(-1), 0.0)
    if not bool(torch.isfinite(label_weight).all() & torch.isfinite(competitor_weight).all()):
        raise ValueError("valid Unembedding rows must be finite in FP32")
    if anchor_centers is None:
        centers = label_weight * 0.5
    else:
        centers = anchor_centers[safe_labels.to(anchor_centers.device)].to(current.device)
        centers = centers.masked_fill(~active_rows.unsqueeze(-1), 0.0)
        if not bool(torch.isfinite(centers).all()):
            raise ValueError("valid anchor_centers rows must be finite in FP32")
    normals = (competitor_weight - label_weight.unsqueeze(1)).masked_fill(~mask.unsqueeze(-1), 0.0)
    normal_norms = torch.linalg.vector_norm(normals, dim=-1)
    if not bool(torch.isfinite(normal_norms).all()):
        raise ValueError("valid Unembedding normal norms must be finite in FP32")
    boundary_mask = mask
    degenerate_pair_mask = None
    if anchor_centers is not None:
        # A constant logit ratio has no hidden-state direction or plane. Test
        # EVERY coordinate, not the norm: a nonzero normal whose squared norm
        # underflows must still fail the original normal_epsilon guard.
        degenerate_pair_mask = normals.eq(0).all(dim=-1) & mask
        if source_weight.dtype == torch.float64:
            # FP64 rows can become equal only after FP32 rounding. Those are
            # not exact duplicates either; retain the fail-closed norm guard.
            source_labels = source_weight[safe_labels.to(source_weight.device)]
            source_neighbors = source_weight[safe_neighbors.to(source_weight.device)]
            exact_duplicates = source_neighbors.eq(source_labels.unsqueeze(1)).all(dim=-1)
            degenerate_pair_mask &= exact_duplicates.to(current.device)
        boundary_mask = mask & ~degenerate_pair_mask
    if bool(((normal_norms <= normal_epsilon) & boundary_mask).any()):
        raise ValueError("label/competitor Unembedding rows have a degenerate normal")
    denominators = torch.where(boundary_mask, normal_norms, 1.0)
    unit_normals = normals / denominators.unsqueeze(-1)
    center_margins = -(unit_normals * centers.unsqueeze(1)).sum(dim=-1)
    if source_bias is not None:
        label_bias = source_bias[safe_labels.to(source_bias.device)].to(
            device=current.device, dtype=torch.float32
        )
        competitor_bias = source_bias[safe_neighbors.to(source_bias.device)].to(
            device=current.device, dtype=torch.float32
        )
        label_bias = label_bias.masked_fill(~active_rows, 0.0)
        competitor_bias = competitor_bias.masked_fill(~mask, 0.0)
        if not bool(torch.isfinite(label_bias).all() & torch.isfinite(competitor_bias).all()):
            raise ValueError("valid Unembedding biases must be finite in FP32")
        center_margins = center_margins + (label_bias.unsqueeze(1) - competitor_bias) / denominators
    return (
        centers,
        unit_normals,
        center_margins.masked_fill(~boundary_mask, 0.0),
        degenerate_pair_mask,
    )


def chunked_lvd_plus_pressure(
    *,
    current_hidden: Tensor,
    unembedding_weight: Tensor,
    label_token_ids: Tensor,
    competitor_ids: Tensor,
    competitor_mask: Tensor,
    config: LVDPlusConfig,
    unembedding_bias: Tensor | None = None,
    anchor_centers: Tensor | None = None,
) -> LVDPlusPressureResult:
    """Form label-site pressure with bounded [chunk,K,H] memory in both passes.

    The source head is detached without copying the full table (CPU supported);
    only indexed rows move to the hidden-state device. Competitors are supplied
    by the caller, normally current Top-512 excluding the GOLD label j. No
    reference hidden state, base-correct filter or d0 participates here.
    Every selected pair uses alpha=1/K_selected, including already-wrong labels
    and, in recorded-S0 mode only, exact duplicates with zero direction/response.

    HALF_UNEMBEDDING forbids a supplied center table. RECORDED_S0 requires a
    frozen FP32 [V,H] table on CPU or the hidden-state device. Only selected
    center rows are transferred and checked for finiteness here; the training
    caller must validate the full S0 cache and its head binding once upstream.
    center_* diagnostics refer to the selected anchor, not necessarily W_j/2.

    Checkpoints retain [N,H] pressure, IDs/masks and detached diagnostics, not
    all [N,K,H] directions. The caller applies ONE existing global outer moment
    to pressure, never an average of per-chunk losses. Empty/all-masked inputs
    return differentiable zeros. All geometry runs outside CPU/CUDA autocast.
    """
    _validate_response_parameters(
        config.exponential_rate, config.transition_width, config.exponent_clamp
    )
    post_crossing_slope_gain = config.post_crossing_slope_gain
    _validate_post_crossing_slope_gain(post_crossing_slope_gain)
    _positive_finite("normal_epsilon", config.normal_epsilon)
    if type(config.position_chunk_size) is not int or config.position_chunk_size <= 0:
        raise ValueError("position_chunk_size must be a positive integer")
    if current_hidden.ndim != 2 or not current_hidden.is_floating_point():
        raise ValueError("current_hidden must have floating-point shape [N,H]")
    n, hidden_size = current_hidden.shape
    if (
        unembedding_weight.ndim != 2
        or unembedding_weight.shape[1] != hidden_size
        or unembedding_weight.shape[0] == 0
        or hidden_size == 0
        or not unembedding_weight.is_floating_point()
    ):
        raise ValueError("unembedding_weight must have floating-point shape [V,H] matching hidden")
    vocabulary_size = unembedding_weight.shape[0]
    if unembedding_bias is not None and (
        unembedding_bias.shape != (vocabulary_size,) or not unembedding_bias.is_floating_point()
    ):
        raise ValueError("unembedding_bias must have floating-point shape [V]")
    if config.anchor_mode is LVDPlusAnchorMode.HALF_UNEMBEDDING:
        if anchor_centers is not None:
            raise ValueError("half_unembedding forbids anchor_centers")
    elif config.anchor_mode is LVDPlusAnchorMode.RECORDED_S0:
        if (
            not isinstance(anchor_centers, Tensor)
            or anchor_centers.layout != torch.strided
            or anchor_centers.shape != unembedding_weight.shape
            or anchor_centers.dtype != torch.float32
            or anchor_centers.requires_grad
        ):
            raise ValueError("recorded_s0 requires frozen FP32 anchor_centers with shape [V,H]")
        if anchor_centers.device.type == "meta" or (
            anchor_centers.device.type != "cpu" and anchor_centers.device != current_hidden.device
        ):
            raise ValueError("anchor_centers must be on CPU or the current_hidden device")
    else:
        raise ValueError("LVD+ anchor_mode must be half_unembedding or recorded_s0")
    if label_token_ids.shape != (n,) or label_token_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("label_token_ids must have integer shape [N]")
    if (
        competitor_ids.ndim != 2
        or competitor_ids.shape[0] != n
        or competitor_ids.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("competitor_ids must have integer shape [N,K]")
    if competitor_mask.shape != competitor_ids.shape or competitor_mask.dtype != torch.bool:
        raise ValueError("competitor_mask must be boolean and match competitor_ids [N,K]")
    labels = label_token_ids.to(device=current_hidden.device, dtype=torch.long)
    neighbors = competitor_ids.to(device=current_hidden.device, dtype=torch.long)
    mask = competitor_mask.to(device=current_hidden.device)
    active_rows = mask.any(dim=-1)
    if bool((((labels < 0) | (labels >= vocabulary_size)) & active_rows).any()):
        raise ValueError("valid label_token_ids contains an out-of-vocabulary token")
    if bool((((neighbors < 0) | (neighbors >= vocabulary_size)) & mask).any()):
        raise ValueError("valid competitor_ids contains an out-of-vocabulary token")
    if bool((neighbors.eq(labels.unsqueeze(1)) & mask).any()):
        raise ValueError("valid competitor_ids must exclude the label_token_ids")
    source_weight = unembedding_weight.detach()
    source_bias = unembedding_bias.detach() if unembedding_bias is not None else None
    center_kwargs = {"anchor_centers": anchor_centers} if anchor_centers is not None else {}
    # Snapshot scalar response settings for checkpoint reevaluation. Omit the
    # new keyword at gain 1 so legacy response adapters retain their signature.
    response_kwargs = {
        "exponential_rate": config.exponential_rate,
        "transition_width": config.transition_width,
        "exponent_clamp": config.exponent_clamp,
    }
    if post_crossing_slope_gain != 1.0:
        response_kwargs["post_crossing_slope_gain"] = post_crossing_slope_gain

    def compute_chunk(current, chunk_labels, chunk_neighbors, chunk_mask):
        with torch.autocast(device_type=current.device.type, enabled=False):
            centers, unit_normals, b, degenerate_pairs = _label_site_geometry(
                current,
                source_weight,
                source_bias,
                chunk_labels,
                chunk_neighbors,
                chunk_mask,
                config.normal_epsilon,
                **center_kwargs,
            )
            boundary_mask = (
                chunk_mask if degenerate_pairs is None else chunk_mask & ~degenerate_pairs
            )
            hidden = current.float().masked_fill(~chunk_mask.any(dim=-1, keepdim=True), 0.0)
            displacement = hidden - centers
            if not bool(torch.isfinite(displacement).all()):
                raise ValueError("valid current_hidden displacements must be finite in FP32")
            signed = (unit_normals * displacement.unsqueeze(1)).sum(dim=-1)
            response = lvd_plus_pressure_response(
                signed,
                b,
                **response_kwargs,
                valid_mask=boundary_mask,
            )
            # Constant pairs stay selected: never renormalize over boundaries.
            alpha = chunk_mask.float() / chunk_mask.sum(dim=-1, keepdim=True).clamp_min(1)
            pressure = ((alpha * response.values).unsqueeze(-1) * unit_normals).sum(dim=1)
            with torch.no_grad():
                residual = b - signed
                center_norms = torch.linalg.vector_norm(displacement, dim=-1)
            return (
                pressure,
                signed.detach(),
                b,
                residual,
                response.values.detach(),
                response.regions,
                response.exponent_clamp_count,
                response.maximum_unclamped_exponent,
                center_norms,
                degenerate_pairs,
            )

    if n == 0:
        pairs = current_hidden.new_zeros(competitor_ids.shape, dtype=torch.float32)
        return LVDPlusPressureResult(
            pressure_vectors=current_hidden.float() * 0.0,
            signed_distances=pairs,
            center_margins=pairs.clone(),
            current_margins=pairs.clone(),
            response=LVDPlusResponse(
                pairs.clone(),
                pairs.long(),
                pairs.new_zeros((), dtype=torch.long),
                pairs.new_zeros(()),
            ),
            center_distance_norms=pairs.new_zeros((0,)),
            degenerate_pair_mask=torch.zeros_like(mask) if anchor_centers is not None else None,
        )

    chunks = []
    for start in range(0, n, config.position_chunk_size):
        selection = slice(start, start + config.position_chunk_size)
        inputs = (
            current_hidden[selection],
            labels[selection],
            neighbors[selection],
            mask[selection],
        )
        if torch.is_grad_enabled() and current_hidden.requires_grad:
            chunks.append(checkpoint(compute_chunk, *inputs, use_reentrant=False))
        else:
            chunks.append(compute_chunk(*inputs))
    columns = list(zip(*chunks, strict=True))
    pressure, signed, margins, residual, responses, regions = (
        torch.cat(column, dim=0) for column in columns[:6]
    )
    return LVDPlusPressureResult(
        pressure_vectors=pressure,
        signed_distances=signed,
        center_margins=margins,
        current_margins=residual,
        response=LVDPlusResponse(
            responses, regions, torch.stack(columns[6]).sum(), torch.stack(columns[7]).max()
        ),
        center_distance_norms=torch.cat(columns[8], dim=0),
        degenerate_pair_mask=torch.cat(columns[9], dim=0) if anchor_centers is not None else None,
    )
