"""Differentiable current-head geometry, independent of S0 and legacy pressure."""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.autograd.function import once_differentiable


def normalized_margins(
    hidden: Tensor,
    weight: Tensor,
    labels: Tensor,
    competitors: Tensor,
    *,
    bias: Tensor | None = None,
    normal_epsilon: float = 1e-6,
    hidden_epsilon: float = 1e-6,
) -> Tensor:
    """Current signed distances; gradients include both head rows and both norms."""
    for name, value in (("normal_epsilon", normal_epsilon), ("hidden_epsilon", hidden_epsilon)):
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if hidden.ndim != 2 or weight.ndim != 2 or hidden.shape[1] != weight.shape[1]:
        raise ValueError("hidden and weight must have matching [N,H] / [V,H] shapes")
    if labels.shape != (len(hidden),) or competitors.ndim != 2:
        raise ValueError("labels and competitors must have [N] / [N,K] shapes")
    if competitors.shape[0] != len(hidden) or competitors.shape[1] == 0:
        raise ValueError("competitors must have at least one direction per position")
    if labels.dtype != torch.long or competitors.dtype != torch.long:
        raise ValueError("labels and competitors must be int64")
    if bias is not None and bias.shape != (len(weight),):
        raise ValueError("bias must have shape [V]")
    if labels.numel() and (
        labels.min() < 0
        or labels.max() >= len(weight)
        or competitors.min() < 0
        or competitors.max() >= len(weight)
        or torch.any(competitors == labels[:, None])
    ):
        raise ValueError("competitors must be valid non-gold vocabulary IDs")
    compute_dtype = (
        torch.float64
        if hidden.dtype == torch.float64 or weight.dtype == torch.float64
        else torch.float32
    )
    with torch.autocast(device_type=hidden.device.type, enabled=False):
        current_hidden = hidden.to(dtype=compute_dtype)
        gold_rows = weight[labels].to(dtype=compute_dtype)
        candidate_rows = weight[competitors].to(dtype=compute_dtype)
        normals = gold_rows[:, None, :] - candidate_rows
        gaps = (normals * current_hidden[:, None, :]).sum(dim=-1)
        if bias is not None:
            gaps = gaps + bias[labels, None].to(compute_dtype) - bias[competitors].to(compute_dtype)
        normal_lengths = (normals.square().sum(dim=-1) + normal_epsilon**2).sqrt()
        hidden_lengths = (current_hidden.square().sum(dim=-1) + hidden_epsilon**2).sqrt()
        return gaps / (normal_lengths * hidden_lengths[:, None])


def centered_huber_rows(
    current: Tensor,
    reference: Tensor,
    weights: Tensor,
    *,
    delta: float = 0.01,
) -> tuple[Tensor, Tensor]:
    """Per-position weighted Huber and common drift; references/weights are immutable."""
    if isinstance(delta, bool) or not math.isfinite(delta) or delta <= 0:
        raise ValueError("delta must be finite and positive")
    if current.ndim != 2 or reference.shape != current.shape or weights.shape != current.shape:
        raise ValueError("current, reference and weights must have matching [N,K] shapes")
    fixed_weights = weights.detach().to(device=current.device, dtype=current.dtype)
    if not torch.isfinite(fixed_weights).all() or torch.any(fixed_weights < 0):
        raise ValueError("weights must be finite and non-negative")
    if not torch.allclose(
        fixed_weights.sum(dim=-1),
        torch.ones(len(current), dtype=current.dtype, device=current.device),
        atol=1e-6,
        rtol=1e-6,
    ):
        raise ValueError("weights must sum to one per position")
    drift = current - reference.detach().to(device=current.device, dtype=current.dtype)
    common = (fixed_weights * drift).sum(dim=-1)
    residual = (drift - common[:, None]) / delta
    magnitude = residual.abs()
    huber = torch.where(magnitude <= 1, residual.square() / 2, magnitude - 0.5)
    return (fixed_weights * huber).sum(dim=-1), common


@torch.no_grad()
def base_conditioned_band_widths(
    base_logits: Tensor,
    peak_logits: Tensor,
    scale_margins: Tensor,
    frequencies: Tensor,
    *,
    max_frequency: int,
    scale_ratio: float = 0.1,
    temperature: float = 2.0,
    frequency_strength: float = 0.1,
    epsilon: float = 1e-6,
) -> Tensor:
    """Base-only geometric scale times a tempered probability ratio to the non-gold peak."""
    if base_logits.ndim != 2 or base_logits.shape[1] == 0:
        raise ValueError("Base logits must have nonempty [N,K] shape")
    if peak_logits.shape != (len(base_logits),) or frequencies.shape != base_logits.shape:
        raise ValueError("Base peaks/frequencies must match candidate logits")
    if (
        scale_margins.ndim != 2
        or len(scale_margins) != len(base_logits)
        or scale_margins.shape[1] == 0
    ):
        raise ValueError("Base scale margins must have nonempty [N,S] shape")
    for name, value in (
        ("scale_ratio", scale_ratio),
        ("temperature", temperature),
        ("epsilon", epsilon),
    ):
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if (
        isinstance(frequency_strength, bool)
        or not math.isfinite(frequency_strength)
        or not 0 <= frequency_strength < 1
    ):
        raise ValueError("frequency_strength must be in [0,1)")
    if type(max_frequency) is not int or max_frequency < 0:
        raise ValueError("max_frequency must be a non-negative integer")
    if not all(
        torch.isfinite(value).all()
        for value in (base_logits, peak_logits, scale_margins, frequencies)
    ):
        raise ValueError("Base band inputs must be finite")
    if torch.any(frequencies < 0) or torch.any(frequencies > max_frequency):
        raise ValueError("candidate frequencies must be between zero and max_frequency")
    dtype = torch.float64 if base_logits.dtype == torch.float64 else torch.float32
    device = base_logits.device
    logits = base_logits.to(dtype=dtype)
    peaks = peak_logits.to(device=device, dtype=dtype)
    scale = scale_margins.to(device=device, dtype=dtype).clamp_min(0).median(dim=-1).values
    amplitude = scale.clamp_min(epsilon) * scale_ratio
    frequency_score = frequencies.to(device=device, dtype=dtype).log1p() / math.log1p(
        max(max_frequency, 1)
    )
    frequency_factor = 1 - frequency_strength + frequency_strength * frequency_score
    log_width = (
        amplitude.log()[:, None]
        + (logits - peaks[:, None]).clamp_max(0) / temperature
        + frequency_factor.log()
    )
    return log_width.clamp_min(math.log(torch.finfo(dtype).tiny)).exp()


def relative_band_rows(
    current: Tensor,
    reference: Tensor,
    weights: Tensor,
    *,
    tolerance: float = 0.5,
    delta: float = 0.25,
    floor: float = 1e-3,
    surrogate: str | None = None,
) -> tuple[Tensor, Tensor]:
    """Per-position weighted Huber of relative margin erosion beyond a free tolerance.

    Each direction keeps its own Base scale: with ``s = max(reference, floor)`` the
    relative erosion is ``r = (reference - current) / s`` (0 at Base, 1 on the
    boundary, above 1 crossed).  Erosion up to ``tolerance`` is free; beyond it
    ``rows = sum_k w_k * huber_1((r - tolerance)_+ / delta)``.  Margin increases are
    never penalised.  The second output is the weighted mean erosion per position
    (diagnostic, detached).

    ``surrogate`` replaces the objective by a linear calibration surrogate whose
    gradient is the maximal Huber slope: ``"all"`` saturates every direction,
    ``"nearest"`` only the direction with the smallest Base scale per position
    (the competitor closest to its boundary).  Surrogates are used only to
    calibrate the coefficient, never for training.
    """
    for name, value in (("delta", delta), ("floor", floor)):
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if surrogate not in (None, "all", "nearest"):
        raise ValueError("surrogate must be None, 'all' or 'nearest'")
    if isinstance(tolerance, bool) or not math.isfinite(tolerance) or not 0 <= tolerance < 1:
        raise ValueError("tolerance must be in [0,1)")
    if current.ndim != 2 or reference.shape != current.shape or weights.shape != current.shape:
        raise ValueError("current, reference and weights must have matching [N,K] shapes")
    fixed_weights = weights.detach().to(device=current.device, dtype=current.dtype)
    fixed_reference = reference.detach().to(device=current.device, dtype=current.dtype)
    if not torch.isfinite(fixed_weights).all() or torch.any(fixed_weights < 0):
        raise ValueError("weights must be finite and non-negative")
    if not torch.isfinite(fixed_reference).all():
        raise ValueError("reference margins must be finite")
    if not torch.allclose(
        fixed_weights.sum(dim=-1),
        torch.ones(len(current), dtype=current.dtype, device=current.device),
        atol=1e-6,
        rtol=1e-6,
    ):
        raise ValueError("weights must sum to one per position")
    scale = fixed_reference.clamp_min(floor)
    erosion = (fixed_reference - current) / scale
    mean_erosion = (fixed_weights * erosion.detach()).sum(dim=-1)
    if surrogate == "all":
        return (fixed_weights * (-current / scale) / delta).sum(dim=-1), mean_erosion
    if surrogate == "nearest":
        nearest = torch.where(fixed_weights > 0, scale, torch.inf).argmin(dim=-1, keepdim=True)
        picked = fixed_weights.gather(1, nearest) * (-current.gather(1, nearest))
        return (picked / scale.gather(1, nearest) / delta).sum(dim=-1), mean_erosion
    scaled = (erosion - tolerance).clamp_min(0) / delta
    huber = torch.where(scaled <= 1, scaled.square() / 2, scaled - 0.5)
    return (fixed_weights * huber).sum(dim=-1), mean_erosion


@torch.no_grad()
def relative_margin_gradients(
    current: Tensor,
    reference: Tensor,
    weights: Tensor,
    *,
    tolerance: float = 0.5,
    delta: float = 0.25,
    floor: float = 1e-3,
) -> Tensor:
    """Closed-form d(rows)/d(current) of :func:`relative_band_rows` (no surrogate)."""
    fixed_weights = weights.detach().to(device=current.device, dtype=current.dtype)
    fixed_reference = reference.detach().to(device=current.device, dtype=current.dtype)
    scale = fixed_reference.clamp_min(floor)
    erosion = (fixed_reference - current.detach()) / scale
    slope = ((erosion - tolerance).clamp_min(0) / delta).clamp_max(1)
    return torch.where(fixed_weights > 0, -fixed_weights * slope / (delta * scale), 0.0)


@torch.no_grad()
def immediate_hidden_gradients(
    hidden: Tensor,
    weight: Tensor,
    labels: Tensor,
    competitors: Tensor,
    margin_gradient: Tensor,
    *,
    normal_epsilon: float = 1e-6,
    hidden_epsilon: float = 1e-6,
    position_chunk_size: int = 4,
) -> Tensor:
    """Per-position gradient w.r.t. the hidden state of ``sum_k g_k * a_k`` (cosine margins).

    ``d a_k / d h = n_k / (|n_k| |h|) - a_k h / |h|^2`` with ``n_k = w_gold - w_k``; the
    result is the diagnostic "immediate" hidden gradient (before the model Jacobian).
    """
    if hidden.ndim != 2 or margin_gradient.shape != competitors.shape:
        raise ValueError("hidden must be [N,H] and margin_gradient must match competitors [N,K]")
    output = torch.zeros_like(hidden)
    for start in range(0, len(hidden), position_chunk_size):
        stop = start + position_chunk_size
        local_hidden = hidden[start:stop].float()
        gold = weight[labels[start:stop]].float()
        rows = weight[competitors[start:stop]].float()
        normals = gold[:, None, :] - rows
        normal_lengths = (normals.square().sum(dim=-1) + normal_epsilon**2).sqrt()
        hidden_lengths = (local_hidden.square().sum(dim=-1) + hidden_epsilon**2).sqrt()
        margins = (normals * local_hidden[:, None, :]).sum(dim=-1) / (
            normal_lengths * hidden_lengths[:, None]
        )
        gradient = margin_gradient[start:stop].float()
        coefficient = gradient / (normal_lengths * hidden_lengths[:, None])
        radial = (gradient * margins).sum(dim=-1) / hidden_lengths.square()
        output[start:stop] = (
            torch.einsum("nk,nkd->nd", coefficient, normals) - radial[:, None] * local_hidden
        ).to(output.dtype)
    return output


class _BandHuberRows(torch.autograd.Function):
    @staticmethod
    def solve(current, reference, weights, widths, delta, solver_iterations):
        drift = current.double() - reference.double()
        fixed_weights = weights.double()
        band = widths.double()
        boundary = (-current.double()).clamp_min(0)

        def derivative(center):
            offset = center[:, None] - drift
            excess = offset.abs() - band
            distance = torch.maximum(excess, boundary)
            slope = fixed_weights * (distance / delta).clamp(0, 1) / delta
            left = (offset >= 0) & (excess >= boundary)
            right = (offset < 0) & (excess > boundary)
            return (slope * (left.double() - right.double())).sum(dim=-1)

        lower = torch.zeros(len(current), device=current.device, dtype=torch.float64)
        upper = torch.where(fixed_weights > 0, drift + band + boundary, 0).amax(dim=-1).clamp_min(0)
        bracket = upper.clone()
        at_zero = derivative(lower) >= 0
        for _ in range(solver_iterations):
            midpoint = (lower + upper) / 2
            increasing = derivative(midpoint) >= 0
            upper = torch.where(increasing, midpoint, upper)
            lower = torch.where(increasing, lower, midpoint)
        center = torch.where(at_zero, 0, upper)
        offset = center[:, None] - drift
        excess = offset.abs() - band
        distance = torch.maximum(excess, boundary)
        scaled = distance / delta
        huber = torch.where(scaled <= 1, scaled.square() / 2, scaled - 0.5)
        rows = (fixed_weights * huber).sum(dim=-1)
        tolerance = bracket / 2**solver_iterations + 32 * torch.finfo(torch.float64).eps * (
            1 + drift.abs().amax(dim=-1) + band.amax(dim=-1) + boundary.amax(dim=-1)
        )
        tied = (excess - boundary).abs().le(tolerance[:, None]) & boundary.gt(0)
        strict = excess > boundary + tolerance[:, None]
        left = (offset >= 0) & (strict | tied)
        right = (offset < 0) & strict
        slope = fixed_weights * scaled.clamp(0, 1) / delta
        center_gradient = (slope * (left.double() - right.double())).sum(dim=-1)
        tied_slope = (slope * tied).sum(dim=-1)
        fraction = torch.where(
            (center > 0) & (tied_slope > 0),
            (center_gradient / tied_slope.clamp_min(torch.finfo(torch.float64).tiny)).clamp(0, 1),
            0,
        )
        gradient = slope * (2 * right.double() - 1)
        gradient += 2 * fraction[:, None] * slope * (tied & (offset < 0))
        return tuple(value.to(dtype=current.dtype) for value in (rows, center, gradient))

    @staticmethod
    def forward(ctx, current, reference, weights, widths, delta, solver_iterations):
        rows, centers, gradient = _BandHuberRows.solve(
            current, reference, weights, widths, delta, solver_iterations
        )
        ctx.save_for_backward(gradient)
        ctx.mark_non_differentiable(centers)
        return rows, centers

    @staticmethod
    @once_differentiable
    def backward(ctx, output_gradient, _center_gradient):
        (gradient,) = ctx.saved_tensors
        return output_gradient[:, None] * gradient, None, None, None, None, None


def _band_inputs(
    current: Tensor,
    reference: Tensor,
    weights: Tensor,
    widths: Tensor,
    *,
    delta: float = 0.01,
    solver_iterations: int = 48,
) -> list[Tensor]:
    if isinstance(delta, bool) or not math.isfinite(delta) or delta <= 0:
        raise ValueError("delta must be finite and positive")
    if type(solver_iterations) is not int or not 16 <= solver_iterations <= 96:
        raise ValueError("solver_iterations must be an integer in [16,96]")
    if current.ndim != 2 or current.shape[1] == 0:
        raise ValueError("band margins must have nonempty [N,K] shape")
    if any(value.shape != current.shape for value in (reference, weights, widths)):
        raise ValueError("reference, weights and widths must match current margins")
    fixed = [
        value.detach().to(device=current.device, dtype=current.dtype)
        for value in (reference, weights, widths)
    ]
    if not all(torch.isfinite(value).all() for value in (current, *fixed)):
        raise ValueError("band inputs must be finite")
    if torch.any(fixed[1] < 0) or torch.any(fixed[2] < 0):
        raise ValueError("band weights and widths must be non-negative")
    if not torch.allclose(fixed[1].sum(-1), torch.ones_like(current[:, 0]), atol=1e-6, rtol=1e-6):
        raise ValueError("band weights must sum to one per position")
    return fixed


def band_huber_rows(
    current: Tensor,
    reference: Tensor,
    weights: Tensor,
    widths: Tensor,
    *,
    delta: float = 0.01,
    solver_iterations: int = 48,
) -> tuple[Tensor, Tensor]:
    """Convex nonnegative-center band projection with KKT-balanced first-order gradients."""
    fixed = _band_inputs(
        current, reference, weights, widths, delta=delta, solver_iterations=solver_iterations
    )
    return _BandHuberRows.apply(current, *fixed, delta, solver_iterations)


class _ChunkedCenteredHuber(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden,
        weight,
        labels,
        competitors,
        reference,
        weights,
        bias,
        delta,
        normal_epsilon,
        hidden_epsilon,
        position_chunk_size,
        widths,
        solver_iterations,
        relative,
        head_gradient_scale,
        hidden_gradient_scale,
    ):
        ctx.options = (
            delta,
            normal_epsilon,
            hidden_epsilon,
            position_chunk_size,
            solver_iterations,
        )
        ctx.relative = relative
        ctx.head_gradient_scale = head_gradient_scale
        ctx.hidden_gradient_scale = hidden_gradient_scale
        margins = []
        for start in range(0, len(hidden), position_chunk_size):
            stop = start + position_chunk_size
            current = normalized_margins(
                hidden[start:stop],
                weight,
                labels[start:stop],
                competitors[start:stop],
                bias=bias,
                normal_epsilon=normal_epsilon,
                hidden_epsilon=hidden_epsilon,
            )
            margins.append(current)
        current = torch.cat(margins)
        margin_gradient = None
        if relative is not None:
            tolerance, relative_delta, floor, surrogate = relative
            output, common = relative_band_rows(
                current,
                reference,
                weights,
                tolerance=tolerance,
                delta=relative_delta,
                floor=floor,
                surrogate=surrogate,
            )
        elif widths is None:
            output, common = centered_huber_rows(current, reference, weights, delta=delta)
        else:
            fixed = _band_inputs(
                current,
                reference,
                weights,
                widths,
                delta=delta,
                solver_iterations=solver_iterations,
            )
            output, common, margin_gradient = _BandHuberRows.solve(
                current, *fixed, delta, solver_iterations
            )
        ctx.save_for_backward(
            hidden, weight, labels, competitors, reference, weights, bias, margin_gradient
        )
        ctx.mark_non_differentiable(common, current)
        return output, common, current

    @staticmethod
    @once_differentiable
    def backward(ctx, output_gradient, _common_gradient, _margin_gradient):
        hidden, weight, labels, competitors, reference, weights, bias, margin_gradient = (
            ctx.saved_tensors
        )
        delta, normal_epsilon, hidden_epsilon, chunk_size, solver_iterations = ctx.options
        hidden_gradient = torch.zeros_like(hidden) if ctx.needs_input_grad[0] else None
        weight_gradient = torch.zeros_like(weight) if ctx.needs_input_grad[1] else None
        bias_gradient = (
            torch.zeros_like(bias) if bias is not None and ctx.needs_input_grad[6] else None
        )
        width = competitors.shape[1] + 1
        for start in range(0, len(hidden), chunk_size):
            stop = min(start + chunk_size, len(hidden))
            ids = torch.cat((labels[start:stop, None], competitors[start:stop]), dim=1).flatten()
            with torch.enable_grad():
                local_hidden = hidden[start:stop].detach().requires_grad_(True)
                local_weight = weight[ids].detach().requires_grad_(True)
                local_bias = None if bias is None else bias[ids].detach().requires_grad_(True)
                local_gold = torch.arange(stop - start, device=ids.device) * width
                local_competitors = local_gold[:, None] + torch.arange(1, width, device=ids.device)
                current = normalized_margins(
                    local_hidden,
                    local_weight,
                    local_gold,
                    local_competitors,
                    bias=local_bias,
                    normal_epsilon=normal_epsilon,
                    hidden_epsilon=hidden_epsilon,
                )
                if margin_gradient is None:
                    if ctx.relative is not None:
                        tolerance, relative_delta, floor, surrogate = ctx.relative
                        rows, _common = relative_band_rows(
                            current,
                            reference[start:stop],
                            weights[start:stop],
                            tolerance=tolerance,
                            delta=relative_delta,
                            floor=floor,
                            surrogate=surrogate,
                        )
                    else:
                        rows, _common = centered_huber_rows(
                            current, reference[start:stop], weights[start:stop], delta=delta
                        )
                    upstream = output_gradient[start:stop]
                else:
                    rows = current
                    upstream = margin_gradient[start:stop] * output_gradient[start:stop, None]
                inputs = (
                    (local_hidden, local_weight)
                    if local_bias is None
                    else (
                        local_hidden,
                        local_weight,
                        local_bias,
                    )
                )
                gradients = torch.autograd.grad(rows, inputs, upstream)
            if hidden_gradient is not None:
                hidden_scale = ctx.hidden_gradient_scale
                hidden_gradient[start:stop].copy_(
                    gradients[0] * hidden_scale if hidden_scale != 1 else gradients[0]
                )
            # Head rows are shared by every position and token; a scale below one
            # keeps most of the geometry correction on the hidden representation.
            scale = ctx.head_gradient_scale
            if weight_gradient is not None:
                weight_gradient.index_add_(
                    0, ids, gradients[1] * scale if scale != 1 else gradients[1]
                )
            if bias_gradient is not None:
                bias_gradient.index_add_(
                    0, ids, gradients[2] * scale if scale != 1 else gradients[2]
                )
        return (
            hidden_gradient,
            weight_gradient,
            None,
            None,
            None,
            None,
            bias_gradient,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def chunked_centered_huber_rows(
    hidden: Tensor,
    weight: Tensor,
    labels: Tensor,
    competitors: Tensor,
    reference: Tensor,
    weights: Tensor,
    *,
    bias: Tensor | None = None,
    delta: float = 0.01,
    normal_epsilon: float = 1e-6,
    hidden_epsilon: float = 1e-6,
    position_chunk_size: int = 4,
    widths: Tensor | None = None,
    solver_iterations: int = 48,
    relative: dict | None = None,
    head_gradient_scale: float = 1.0,
    hidden_gradient_scale: float = 1.0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Exact first-order local autograd, accumulating into one vocabulary-sized gradient.

    ``head_gradient_scale`` in [0,1] multiplies only the head-row (and bias) part
    of the geometry gradient and ``hidden_gradient_scale`` in [0,1] only the
    hidden-state part; at most one of them may be zero.

    ``relative`` selects the relative-erosion objective (keys ``tolerance``,
    ``delta``, ``floor`` and optional ``surrogate``); it cannot be combined with
    band ``widths``.  Without it, ``widths`` selects the Base-conditioned band and
    ``None`` the centered balance objective.
    """
    if type(position_chunk_size) is not int or position_chunk_size <= 0:
        raise ValueError("position_chunk_size must be a positive integer")
    if len(hidden) == 0:
        raise ValueError("chunked geometry requires at least one eligible position")
    if (
        isinstance(head_gradient_scale, bool)
        or not math.isfinite(head_gradient_scale)
        or not 0 <= head_gradient_scale <= 1
    ):
        raise ValueError("head_gradient_scale must be in [0,1]")
    if (
        isinstance(hidden_gradient_scale, bool)
        or not math.isfinite(hidden_gradient_scale)
        or not 0 <= hidden_gradient_scale <= 1
    ):
        raise ValueError("hidden_gradient_scale must be in [0,1]")
    if head_gradient_scale == 0 and hidden_gradient_scale == 0:
        raise ValueError("head_gradient_scale and hidden_gradient_scale cannot both be zero")
    relative_options = None
    if relative is not None:
        if widths is not None:
            raise ValueError("relative erosion and band widths are mutually exclusive")
        relative_options = (
            float(relative["tolerance"]),
            float(relative["delta"]),
            float(relative["floor"]),
            relative.get("surrogate"),
        )
    return _ChunkedCenteredHuber.apply(
        hidden,
        weight,
        labels,
        competitors,
        reference.detach(),
        weights.detach(),
        bias,
        delta,
        normal_epsilon,
        hidden_epsilon,
        position_chunk_size,
        None if widths is None else widths.detach(),
        solver_iterations,
        relative_options,
        float(head_gradient_scale),
        float(hidden_gradient_scale),
    )


@torch.no_grad()
def coverage_probabilities(frequencies: Tensor, *, cap_multiplier: float = 8.0) -> Tensor:
    """Capped inverse-sqrt probabilities over the supplied remaining vocabulary."""
    if frequencies.ndim != 1 or frequencies.numel() == 0:
        raise ValueError("frequencies must be a nonempty vector")
    if not torch.isfinite(frequencies).all() or torch.any(frequencies < 0):
        raise ValueError("frequencies must be finite and non-negative")
    if isinstance(cap_multiplier, bool) or not math.isfinite(cap_multiplier) or cap_multiplier < 1:
        raise ValueError("cap_multiplier must be finite and at least one")
    raw = (frequencies.detach().double() + 1).rsqrt()
    cap = min(1.0, cap_multiplier / len(raw))
    normalized = raw / raw.sum()
    if float(normalized.max()) <= cap:
        return normalized
    lower = 0.0
    upper = 1.0 / float(raw.min())
    for _ in range(64):
        midpoint = (lower + upper) / 2
        if float((raw * midpoint).clamp_max(cap).sum()) < 1:
            lower = midpoint
        else:
            upper = midpoint
    probabilities = (raw * upper).clamp_max(cap)
    return probabilities / probabilities.sum()


@torch.no_grad()
def dynamic_competitors(
    logits: Tensor,
    base_top_ids: Tensor,
    reference_weight: Tensor,
    label: int,
    frequencies: Tensor,
    config,
    *,
    seed: int,
) -> tuple[Tensor, Tensor, tuple[int, int, int]]:
    """Disjoint current/Base/coverage groups; sampling priorities are not loss weights."""
    vocabulary = len(reference_weight)
    if logits.shape != (vocabulary,) or frequencies.shape != (vocabulary,):
        raise ValueError("candidate logits/frequencies must match the reference vocabulary")
    if not 0 <= label < vocabulary:
        raise ValueError("candidate label is outside the reference vocabulary")
    ranking = logits.detach().float().cpu().clone()
    ranking[label] = -torch.inf
    current = ranking.topk(min(config.current_directions, vocabulary - 1)).indices
    used = torch.zeros(vocabulary, dtype=torch.bool)
    used[label] = True
    used[current] = True
    base = base_top_ids.detach().cpu().long()
    if base.numel() and (base.min() < 0 or base.max() >= vocabulary):
        raise ValueError("invalid Base candidate vocabulary IDs")
    base = base[~used[base]][: config.base_directions]
    if base.numel() != base.unique().numel():
        raise ValueError("duplicate Base candidate IDs")
    used[base] = True
    remaining = (~used).nonzero().flatten()
    if len(remaining):
        probabilities = coverage_probabilities(
            frequencies[remaining],
            cap_multiplier=config.coverage_cap_multiplier,
        )
        generator = torch.Generator(device="cpu").manual_seed(seed)
        coverage = remaining[
            torch.multinomial(
                probabilities,
                min(config.coverage_directions, len(remaining)),
                replacement=False,
                generator=generator,
            )
        ]
    else:
        coverage = torch.empty(0, dtype=torch.long)
    groups = []
    for ids in (current, base, coverage):
        lengths = (reference_weight[ids] - reference_weight[label]).norm(dim=-1)
        groups.append(ids[lengths > config.normal_epsilon])
    masses = (config.current_mass, config.base_mass, config.coverage_mass)
    total_mass = sum(mass for ids, mass in zip(groups, masses, strict=True) if len(ids))
    if not total_mass:
        return torch.empty(0, dtype=torch.long), torch.empty(0), (0, 0, 0)
    weights = torch.cat(
        [
            torch.full((len(ids),), mass / total_mass / len(ids), dtype=torch.float32)
            for ids, mass in zip(groups, masses, strict=True)
            if len(ids)
        ]
    )
    return torch.cat(groups), weights, tuple(len(ids) for ids in groups)
