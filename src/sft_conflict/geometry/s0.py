"""Full-vocabulary, fixed-head probability maximizers on an explicit L2 ball.

For each label j this minimizes the convex function
F_j(h) = log(1 + sum_{i != j} exp((W_i-W_j) h + b_i-b_j)), ||h|| <= R.
There is deliberately NO label-cell constraint. A maximizer can be outside its
label's cell, and a small numerical gradient is not an exact finite-maximizer
certificate. The Frank--Wolfe gap bounds suboptimality in F, not distance to a
unique maximizer. Duplicate rows and nonunique optima are allowed.

Import this module directly; it has no configuration, model, or cache dependency.
The configuration owner supplies radius (normally DEFAULT_RADIUS = 512). Cache
owners must bind the original head/bias AND summary['solver_config'], including
solver version 1. This module neither publishes nor rejects an incomplete cache.

Memory is O(VH + B(C+H)), never V-by-V: resident centered FP64/FP32 heads, a CPU
FP32 output, and token/vocabulary tiles. The FP64 head is prepared once, not
copied from the host on every audit. With C >= V each evaluation uses one logits
GEMM and, when needed, one expected-weight GEMM. Already-audited converged rows
are frozen and removed from optimization batches. No autograd graph is built.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum

import torch
from torch import Tensor

DEFAULT_RADIUS = 512.0
S0_SOLVER_VERSION = 1
_MAX_BACKTRACKS = 64
_ARMIJO = 1e-4


class S0Status(IntEnum):
    """Disjoint interpretation labels; convergence is a SEPARATE boolean."""

    UNKNOWN_UNRESOLVED = 0
    P1_ASYMPTOTIC_CERTIFIED = 1
    FINITE_CONSTANT_SOFTMAX = 2
    FINITE_INTERIOR_NUMERICAL = 3


S0_STATUS_NAMES = {
    int(S0Status.UNKNOWN_UNRESOLVED): "unknown_unresolved",
    int(S0Status.P1_ASYMPTOTIC_CERTIFIED): "p1_asymptotic_certified",
    int(S0Status.FINITE_CONSTANT_SOFTMAX): "finite_constant_softmax",
    int(S0Status.FINITE_INTERIOR_NUMERICAL): "finite_interior_numerical",
}


@dataclass(frozen=True)
class S0OptimizationResult:
    """All V labels, including unresolved ones, in original vocabulary order.

    sites: detached CPU float32 [V,H], with ||sites[j]|| <= radius.
    per_token: detached CPU [V] tensors, real diagnostics in float64, flags in
        bool, and IDs/status/iteration counts in int64. `normal_norm` denotes
        ||W_j-mean(W)||, the initialization normal, NOT a floored pair normal.
        `projected_gradient_norm` uses the unit-step mapping h-P_ball(h-grad F).
        `on_boundary` uses 1-||h||/R <= audit_tolerance (dimensionless).
        `inside_cell`, `cell_tie`, `outside_cell` use the FP64 minimum *logit*
        margin with strict >tol, absolute <=tol, and strict <-tol respectively.
        `converged` requires the FP64 gap of the ACTUAL returned FP32 point.
        `status` decodes through S0_STATUS_NAMES. Only the identical-weight case
        sets `finite_optimum_certified`. `p1_asymptotic_certified` instead proves
        an unattained supremum of one along a ray, independent of ball convergence.
        Underflow flags distinguish unrepresentable tiny values from exact zero.
    summary: JSON-compatible metadata, exact solver settings and independent
        status/convergence/cell/boundary counts. Publication is the caller's job.
    """

    sites: Tensor
    per_token: dict[str, Tensor]
    summary: dict[str, object]

    @property
    def summary_json(self) -> str:
        return json.dumps(self.summary, sort_keys=True, allow_nan=False)


@dataclass(frozen=True)
class _Head:
    weight64: Tensor
    bias64: Tensor
    weight32: Tensor | None
    bias32: Tensor | None
    column_abs_max: Tensor
    constant: bool


@dataclass(frozen=True)
class _Evaluation:
    loss: Tensor
    gradient: Tensor | None
    conditional_gradient: Tensor | None
    log_other_mass: Tensor
    log_odds: Tensor
    cell_margin: Tensor
    linear_margin: Tensor
    difference_gradient: Tensor

    def select(self, indices: Tensor) -> _Evaluation:
        return _Evaluation(
            self.loss[indices],
            None if self.gradient is None else self.gradient[indices],
            None if self.conditional_gradient is None else self.conditional_gradient[indices],
            self.log_other_mass[indices],
            self.log_odds[indices],
            self.cell_margin[indices],
            self.linear_margin[indices],
            self.difference_gradient[indices],
        )


def _positive_finite(name: str, value: float) -> float:
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
    return float(value)


def _positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _norm(value: Tensor) -> Tensor:
    """Scaled FP64 norm: no square overflow/underflow and no epsilon floor."""
    value = value.double()
    scale = value.abs().amax(dim=-1)
    divisor = torch.where(scale == 0, 1.0, scale)
    return scale * (value / divisor.unsqueeze(-1)).square().sum(dim=-1).sqrt()


def _project_ball(value: Tensor, radius: float, dtype: torch.dtype) -> Tensor:
    value = value.double()
    norm = _norm(value)
    denominator = torch.where(norm == 0, 1.0, norm)
    scale = torch.where(norm > radius, radius / denominator, 1.0)
    projected = (value * scale.unsqueeze(-1)).to(dtype)
    # Casting to FP32 can round a boundary point OUT of the ball. Move only
    # offending rows inward by representable ULPs, then audit that stored point.
    for _ in range(4):
        outside = _norm(projected) > radius
        if not bool(outside.any()):
            break
        projected[outside] = torch.nextafter(
            projected[outside], torch.zeros_like(projected[outside])
        )
    if bool((_norm(projected) > radius).any()):
        raise ArithmeticError("cannot represent the projected site inside the requested ball")
    return projected


def _prepare_head(weight: Tensor, bias: Tensor | None, device: torch.device) -> _Head:
    if (
        not isinstance(weight, Tensor)
        or weight.layout != torch.strided
        or weight.ndim != 2
        or weight.shape[0] < 2
        or weight.shape[1] < 1
        or not weight.is_floating_point()
        or weight.device.type == "meta"
    ):
        raise ValueError("weight must be a real floating tensor [V,H], V >= 2 and H >= 1")
    if bias is not None and (
        not isinstance(bias, Tensor)
        or bias.layout != torch.strided
        or bias.shape != (weight.shape[0],)
        or not bias.is_floating_point()
        or bias.device.type == "meta"
    ):
        raise ValueError("bias must be a real floating tensor [V] or None")
    w = weight.detach().to(device=device, dtype=torch.float64, copy=True)
    b = (
        torch.zeros(len(w), device=device, dtype=torch.float64)
        if bias is None
        else bias.detach().to(device=device, dtype=torch.float64, copy=True)
    )
    if not bool(torch.isfinite(w).all()):
        raise ValueError("weight must contain only finite values")
    if not bool(torch.isfinite(b).all()):
        raise ValueError("bias must contain only finite values")
    # Remove a common row BEFORE averaging to preserve small differences under
    # a large affine gauge shift. These operations touch only owned copies.
    w.sub_(w[0].clone())
    constant = int(torch.count_nonzero(w).item()) == 0
    w.sub_(w.mean(dim=0))
    b.sub_(b.max())
    if not bool(torch.isfinite(w).all() & torch.isfinite(b).all()):
        raise ValueError("centered weight/bias differences must be representable in float64")
    w32, b32 = w.float(), b.float()
    if not bool(torch.isfinite(w32).all() & torch.isfinite(b32).all()):
        w32, b32 = None, None
    column_abs_max = torch.maximum(w.amax(dim=0), -w.amin(dim=0))
    return _Head(w, b, w32, b32, column_abs_max, constant)


def _scale_gradient(conditional: Tensor, log_mass: Tensor) -> Tensor:
    gradient = conditional * log_mass.exp().unsqueeze(-1)
    subnormal_mass = log_mass < math.log(torch.finfo(conditional.dtype).tiny)
    if bool(subnormal_mass.any()):
        values = conditional[subnormal_mass]
        gradient[subnormal_mass] = (
            values.sign() * (values.abs().log() + log_mass[subnormal_mass].unsqueeze(-1)).exp()
        )
    return gradient


def _difference_gradient(
    head: _Head,
    sites: Tensor,
    labels: Tensor,
    log_odds: Tensor,
    vocabulary_chunk_size: int,
) -> Tensor:
    """Rare FP64 cancellation fallback, one [C,H] difference tile at a time.

    A dominant duplicate has an EXACT zero difference. Its weight cannot mask
    a tiny but consequential contribution from another competitor. Subnormal
    probabilities are multiplied in log space before summing the differences.
    This is not an epsilon mask or an approximate/sampled competitor set.
    """
    result = torch.zeros_like(sites)
    for row, label in enumerate(labels.tolist()):
        for start in range(0, len(head.weight64), vocabulary_chunk_size):
            stop = min(start + vocabulary_chunk_size, len(head.weight64))
            difference = head.weight64[start:stop] - head.weight64[label]
            relative = torch.mv(difference, sites[row])
            relative.add_(head.bias64[start:stop] - head.bias64[label])
            if start <= label < stop:
                relative[label - start] = -torch.inf
            result[row] += _scale_gradient(difference, relative - log_odds[row]).sum(dim=0)
    return result


def _evaluate(
    head: _Head,
    sites: Tensor,
    labels: Tensor,
    vocabulary_chunk_size: int,
    *,
    gradient: bool = True,
) -> _Evaluation:
    """Online full-vocabulary LSE and conditional expected OTHER weight.

    grad F = other_mass * (E[W_i | i != j] - W_j). Neither log probability nor
    the gradient subtracts a rounded target probability from one. Normalizing
    each running weighted mean also avoids a V-fold overflow of its numerator.
    The target logit is gathered from the SAME GEMM as competitor logits.
    """
    if sites.dtype == torch.float64:
        weight, bias = head.weight64, head.bias64
    else:
        assert head.weight32 is not None and head.bias32 is not None
        weight, bias = head.weight32, head.bias32
    count, hidden = sites.shape
    maximum = sites.new_full((count,), -torch.inf)
    maximum_linear = maximum.clone()
    total = sites.new_zeros(count)
    target_linear = sites.new_zeros(count)
    expected = sites.new_zeros((count, hidden)) if gradient else None
    for start in range(0, len(weight), vocabulary_chunk_size):
        stop = min(start + vocabulary_chunk_size, len(weight))
        chunk = weight[start:stop]
        logits = torch.mm(sites, chunk.T)
        rows = torch.nonzero((labels >= start) & (labels < stop)).flatten()
        columns = labels[rows] - start
        target_linear[rows] = logits[rows, columns]
        logits[rows, columns] = -torch.inf
        maximum_linear = torch.maximum(maximum_linear, logits.amax(dim=1))
        logits.add_(bias[start:stop])
        new_maximum = torch.maximum(maximum, logits.amax(dim=1))
        # A size-one tile can contain only the excluded label for a row.
        safe_maximum = torch.where(new_maximum == -torch.inf, 0.0, new_maximum)
        old_mass = total * (maximum - safe_maximum).exp()
        logits.sub_(safe_maximum.unsqueeze(-1)).exp_()
        total = old_mass + logits.sum(dim=1)
        divisor = torch.where(total == 0, 1.0, total)
        if expected is not None:
            expected.mul_((old_mass / divisor).unsqueeze(-1))
            logits.div_(divisor.unsqueeze(-1))
            expected.addmm_(logits, chunk)
        maximum = new_maximum
    target = target_linear + bias[labels]
    # This ordering retains log(total) even when maximum and target are huge.
    log_odds = (maximum - target) + total.log()
    loss = torch.logaddexp(torch.zeros_like(log_odds), log_odds)
    log_other_mass = log_odds - loss
    conditional = None if expected is None else expected - weight[labels]
    difference_gradient = torch.zeros(count, dtype=torch.bool, device=sites.device)
    if conditional is not None and sites.dtype == torch.float64 and not head.constant:
        # The fast other-mass formula avoids p_j=1 cancellation, but a dominant
        # duplicate can still cancel E[W_other]-W_j itself. Recompute small
        # residuals relative to the head's scale using explicit differences.
        gamma = (max(weight.shape) + 4) * torch.finfo(torch.float64).eps
        cancellation_scale = 8.0 * gamma * _norm(head.column_abs_max)
        difference_gradient = _norm(conditional) <= cancellation_scale
        if bool(difference_gradient.any()):
            conditional[difference_gradient] = _difference_gradient(
                head,
                sites[difference_gradient],
                labels[difference_gradient],
                log_odds[difference_gradient],
                vocabulary_chunk_size,
            )
    derivative = None if conditional is None else _scale_gradient(conditional, log_other_mass)
    return _Evaluation(
        loss,
        derivative,
        conditional,
        log_other_mass,
        log_odds,
        target - maximum,
        target_linear - maximum_linear,
        difference_gradient,
    )


def _gap_and_gradient_norm(
    sites: Tensor, evaluation: _Evaluation, radius: float
) -> tuple[Tensor, Tensor]:
    assert evaluation.conditional_gradient is not None
    conditional = evaluation.conditional_gradient.double()
    normal = _norm(conditional)
    site_norm = _norm(sites)
    unit_h = sites.double() / torch.where(site_norm == 0, 1.0, site_norm).unsqueeze(-1)
    unit_g = conditional / torch.where(normal == 0, 1.0, normal).unsqueeze(-1)
    # Algebraically g.h + R||g||, but nonnegative without cancellation of two
    # large antiparallel dot products. No epsilon zeroes a small normal/gap.
    distance = (radius - site_norm).clamp_min(0.0) + (
        0.5 * site_norm * (unit_h + unit_g).square().sum(dim=-1)
    )
    log_norm = evaluation.log_other_mass.double() + normal.log()
    return (log_norm + distance.log()).exp(), log_norm.exp()


def _audit_sites(
    head: _Head,
    sites: Tensor,
    labels: Tensor,
    radius: float,
    vocabulary_chunk_size: int,
    gap_tolerance: float,
    audit_tolerance: float,
) -> dict[str, Tensor]:
    """All decisions concern the returned FP32 point, recomputed in FP64."""
    assert sites.dtype == torch.float32
    h = sites.double()
    evaluation = _evaluate(head, h, labels, vocabulary_chunk_size)
    assert evaluation.gradient is not None and evaluation.conditional_gradient is not None
    gap, gradient_norm = _gap_and_gradient_norm(h, evaluation, radius)
    site_norm = _norm(h)
    normal_norm = _norm(head.weight64[labels])
    direction = h / torch.where(site_norm == 0, 1.0, site_norm).unsqueeze(-1)
    directional_margin = evaluation.linear_margin / torch.where(site_norm == 0, 1.0, site_norm)
    fallback = (site_norm == 0) & (normal_norm > 0)
    if bool(fallback.any()):
        direction[fallback] = head.weight64[labels[fallback]] / normal_norm[fallback, None]
        directional_margin[fallback] = _evaluate(
            head, direction[fallback], labels[fallback], vocabulary_chunk_size, gradient=False
        ).linear_margin
    direction_norm = _norm(direction)
    # Conservative dot-product roundoff allowance prevents a duplicate weight
    # from becoming a positive exposure certificate through GEMM roundoff.
    gamma = (h.shape[1] + 4) * torch.finfo(torch.float64).eps
    gamma /= 1.0 - gamma
    roundoff = 2.0 * gamma * (direction.abs() * head.column_abs_max).sum(dim=-1)
    lower_margin = directional_margin - roundoff
    p1 = (direction_norm > 0) & (lower_margin > audit_tolerance)
    feasible = site_norm <= radius
    converged = feasible & (gap <= gap_tolerance)
    norm_over_radius = site_norm / radius
    boundary = (1.0 - norm_over_radius) <= audit_tolerance
    constant = torch.full_like(converged, head.constant)
    numerical_interior = converged & ~boundary & (gradient_norm <= audit_tolerance)
    status = torch.full_like(labels, int(S0Status.UNKNOWN_UNRESOLVED))
    status[numerical_interior] = int(S0Status.FINITE_INTERIOR_NUMERICAL)
    status[p1] = int(S0Status.P1_ASYMPTOTIC_CERTIFIED)
    status[constant] = int(S0Status.FINITE_CONSTANT_SOFTMAX)
    # Identical weights have whole-space cells only for truly maximal biases;
    # their other cells are empty. Do not label those outside cells unbounded.
    cell_unbounded = p1 | (constant & (evaluation.cell_margin >= 0))
    probability = (-evaluation.loss).exp()
    projected = _project_ball(h - evaluation.gradient, radius, torch.float64)
    metrics = {
        "token_id": labels.clone(),
        "log_probability": -evaluation.loss,
        "probability": probability,
        "objective": evaluation.loss,
        "log_other_probability_mass": evaluation.log_other_mass,
        "other_probability_mass": evaluation.log_other_mass.exp(),
        "site_norm": site_norm,
        "norm_over_radius": norm_over_radius,
        "normal_norm": normal_norm,
        "gradient_norm": gradient_norm,
        "projected_gradient_norm": _norm(h - projected),
        "frank_wolfe_gap": gap,
        "cell_margin": evaluation.cell_margin,
        "p1_directional_margin": directional_margin,
        "p1_directional_margin_lower_bound": lower_margin,
        "p1_direction_norm": direction_norm,
        "converged": converged,
        "feasible": feasible,
        "on_boundary": boundary,
        "inside_cell": evaluation.cell_margin > audit_tolerance,
        "cell_tie": evaluation.cell_margin.abs() <= audit_tolerance,
        "outside_cell": evaluation.cell_margin < -audit_tolerance,
        "p1_asymptotic_certified": p1,
        "finite_optimum_certified": constant,
        "cell_unbounded_certified": cell_unbounded,
        "cell_boundedness_unknown": ~cell_unbounded,
        "loss_underflow": evaluation.loss == 0,
        "probability_underflow": probability == 0,
        "gradient_underflow": (gradient_norm == 0) & (_norm(evaluation.conditional_gradient) > 0),
        "gradient_difference_fallback": evaluation.difference_gradient,
        "status": status,
    }
    if any(
        value.is_floating_point() and not bool(torch.isfinite(value).all())
        for value in metrics.values()
    ):
        raise FloatingPointError(
            "full-vocabulary FP64 audit is nonfinite; input scale is too large"
        )
    return metrics


def _backtracking_step(
    head: _Head,
    sites: Tensor,
    labels: Tensor,
    evaluation: _Evaluation,
    steps: Tensor,
    radius: float,
    vocabulary_chunk_size: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    assert evaluation.gradient is not None
    result = sites.clone()
    accepted = torch.zeros(len(sites), dtype=torch.bool, device=sites.device)
    backtracks = torch.zeros_like(labels)
    steps = steps.clone()
    pending = torch.arange(len(sites), device=sites.device)
    for _ in range(_MAX_BACKTRACKS):
        if not len(pending):
            break
        x = sites[pending].double()
        g = evaluation.gradient[pending].double()
        candidate = _project_ball(x - steps[pending, None] * g, radius, sites.dtype)
        delta = candidate.double() - x
        slope = (g * delta).sum(dim=-1)
        movable = (
            torch.isfinite(candidate).all(dim=-1)
            & torch.isfinite(slope)
            & (slope < 0)
            & (candidate != sites[pending]).any(dim=-1)
        )
        trial_loss = x.new_full((len(pending),), torch.inf)
        if bool(movable.any()):
            trial_loss[movable] = _evaluate(
                head,
                candidate[movable],
                labels[pending[movable]],
                vocabulary_chunk_size,
                gradient=False,
            ).loss.double()
        base_loss = evaluation.loss[pending].double()
        good = movable & (trial_loss <= base_loss) & (trial_loss <= base_loss + _ARMIJO * slope)
        chosen = pending[good]
        result[chosen] = candidate[good]
        accepted[chosen] = True
        rejected = pending[~good]
        backtracks[rejected] += 1
        steps[rejected] *= 0.5
        # An unchanged finite projection cannot start moving by halving the
        # step. FP32 failures are promoted; FP64 failures remain unresolved.
        retry = ~good & ((candidate != sites[pending]).any(dim=-1))
        pending = pending[retry]
    return result, accepted, steps, backtracks


def _optimize_batch(
    head: _Head,
    labels: Tensor,
    *,
    radius: float,
    vocabulary_chunk_size: int,
    max_iterations: int,
    initial_step_size: float,
    gap_tolerance: float,
    audit_tolerance: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    normals = head.weight64[labels]
    norm = _norm(normals)
    initial = radius * (normals / torch.where(norm == 0, 1.0, norm).unsqueeze(-1))
    sites = _project_ball(initial, radius, torch.float32)

    def audit(points: Tensor, ids: Tensor) -> dict[str, Tensor]:
        return _audit_sites(
            head, points, ids, radius, vocabulary_chunk_size, gap_tolerance, audit_tolerance
        )

    metrics = audit(sites, labels)
    converged = metrics["converged"].clone()
    current = sites.double()
    steps = torch.full_like(norm, initial_step_size)
    iterations = torch.zeros_like(labels)
    fp32_iterations = torch.zeros_like(labels)
    fp64_iterations = torch.zeros_like(labels)
    backtracks = torch.zeros_like(labels)
    audits = torch.ones_like(labels)
    failed = torch.zeros_like(converged)

    def record(rows: Tensor, points: Tensor) -> None:
        observed = audit(points, labels[rows])
        sites[rows] = points
        for name, values in observed.items():
            metrics[name][rows] = values
        converged[rows] = observed["converged"]
        audits[rows] += 1

    # Reserve at least a quarter of the per-row update budget for FP64. A
    # misleading FP32 stop or a stalled FP32 line search promotes earlier.
    for dtype, limit in (
        (torch.float32, (3 * max_iterations) // 4),
        (torch.float64, max_iterations),
    ):
        if dtype == torch.float32 and head.weight32 is None:
            continue
        active = torch.nonzero(~converged & (iterations < limit)).flatten()
        while len(active):
            x = current[active].to(dtype)
            evaluation = _evaluate(head, x, labels[active], vocabulary_chunk_size)
            assert evaluation.gradient is not None and evaluation.conditional_gradient is not None
            finite = (
                torch.isfinite(evaluation.loss)
                & torch.isfinite(evaluation.gradient).all(dim=-1)
                & torch.isfinite(evaluation.conditional_gradient).all(dim=-1)
            )
            if dtype == torch.float64 and not bool(finite.all()):
                failed[active[~finite]] = True
            active, x, evaluation = active[finite], x[finite], evaluation.select(finite)
            if not len(active):
                break
            approximate_gap, _ = _gap_and_gradient_norm(x, evaluation, radius)
            possible_stop = approximate_gap <= gap_tolerance
            if bool(possible_stop.any()):
                rows = active[possible_stop]
                record(rows, _project_ball(x[possible_stop], radius, torch.float32))
                # NEVER freeze on the FP32 estimate. A failed FP64 audit
                # continues in double precision, within the same total budget.
                keep = ~possible_stop if dtype == torch.float32 else ~converged[active]
                active, x, evaluation = active[keep], x[keep], evaluation.select(keep)
            if not len(active):
                break
            candidate, accepted, used_steps, rejected = _backtracking_step(
                head, x, labels[active], evaluation, steps[active], radius, vocabulary_chunk_size
            )
            iterations[active] += 1
            phase_iterations = fp32_iterations if dtype == torch.float32 else fp64_iterations
            phase_iterations[active] += 1
            backtracks[active] += rejected
            current[active[accepted]] = candidate[accepted].double()
            steps[active[accepted]] = (
                used_steps[accepted].clamp_max(torch.finfo(torch.float64).max / 2.0) * 2.0
            )
            steps[active[~accepted]] = initial_step_size
            if dtype == torch.float64:
                failed[active[~accepted]] = True
            active = active[accepted & (iterations[active] < limit)]
    remaining = torch.nonzero(~converged).flatten()
    if len(remaining):
        record(remaining, _project_ball(current[remaining], radius, torch.float32))
    metrics.update(
        iterations=iterations,
        fp32_iterations=fp32_iterations,
        fp64_iterations=fp64_iterations,
        fp64_refined=fp64_iterations > 0,
        backtracking_steps=backtracks,
        audit_count=audits,
        line_search_failed=failed,
    )
    return sites, metrics


@torch.no_grad()
def optimize_s0(
    weight: Tensor,
    bias: Tensor | None,
    *,
    radius: float,
    token_batch_size: int = 128,
    vocabulary_chunk_size: int = 32768,
    max_iterations: int = 500,
    initial_step_size: float = 16.0,
    gap_tolerance: float = 1e-4,
    audit_tolerance: float = 1e-5,
    device: str = "cpu",
    progress_callback: Callable[[dict[str, object]], None] | None = None,
) -> S0OptimizationResult:
    """Deterministic projected GD/Armijo, FP32 search and authoritative FP64 audit.

    max_iterations bounds attempted projected updates PER ROW across both
    precisions, not the number of backtracking objective evaluations. All
    positive tolerances are absolute except the dimensionless boundary flag.
    A callback receives one JSON-compatible `batch_complete` dictionary per
    completed token batch. Callback errors propagate. This function does not
    change RNG, global matmul precision, the head, or any model gradients.

    The fixed backend/precision settings determine numerical reproducibility;
    bitwise equivalence across devices or different tile sizes is not promised.
    Unrepresentable FP64 diagnostics raise rather than being silently clipped.
    """
    radius = _positive_finite("radius", radius)
    if radius > torch.finfo(torch.float32).max:
        raise ValueError("radius must be representable in the float32 site format")
    initial_step_size = _positive_finite("initial_step_size", initial_step_size)
    gap_tolerance = _positive_finite("gap_tolerance", gap_tolerance)
    audit_tolerance = _positive_finite("audit_tolerance", audit_tolerance)
    for name, value in (
        ("token_batch_size", token_batch_size),
        ("vocabulary_chunk_size", vocabulary_chunk_size),
        ("max_iterations", max_iterations),
    ):
        _positive_integer(name, value)
    if progress_callback is not None and not callable(progress_callback):
        raise ValueError("progress_callback must be callable or None")
    if not isinstance(device, str):
        raise ValueError("device must name a CPU or CUDA device")
    compute_device = torch.device(device)
    if compute_device.type not in {"cpu", "cuda"}:
        raise ValueError("device must support float64 audits (cpu or cuda)")
    solver_config = {
        "version": S0_SOLVER_VERSION,
        "radius": radius,
        "token_batch_size": token_batch_size,
        "vocabulary_chunk_size": vocabulary_chunk_size,
        "max_iterations": max_iterations,
        "initial_step_size": initial_step_size,
        "gap_tolerance": gap_tolerance,
        "audit_tolerance": audit_tolerance,
        "device": device,
        "algorithm": "projected_gradient_armijo",
        "initialization": "radius_normalized_centered_head_row_or_zero",
        "fp32_budget_fraction": 0.75,
        "max_backtracking_steps": _MAX_BACKTRACKS,
        "armijo_coefficient": _ARMIJO,
        "fp64_cancellation_fallback": "explicit_weight_differences",
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }
    started = time.perf_counter()
    with torch.autocast(device_type=compute_device.type, enabled=False):
        head = _prepare_head(weight, bias, compute_device)
        vocabulary, hidden = head.weight64.shape
        sites = torch.empty((vocabulary, hidden), dtype=torch.float32, device="cpu")
        per_token: dict[str, Tensor] = {}
        completed_converged = 0
        for start in range(0, vocabulary, token_batch_size):
            stop = min(start + token_batch_size, vocabulary)
            labels = torch.arange(start, stop, device=compute_device)
            batch_sites, batch_metrics = _optimize_batch(
                head,
                labels,
                radius=radius,
                vocabulary_chunk_size=vocabulary_chunk_size,
                max_iterations=max_iterations,
                initial_step_size=initial_step_size,
                gap_tolerance=gap_tolerance,
                audit_tolerance=audit_tolerance,
            )
            sites[start:stop] = batch_sites.cpu()
            for name, values in batch_metrics.items():
                if name not in per_token:
                    per_token[name] = torch.empty(vocabulary, dtype=values.dtype, device="cpu")
                per_token[name][start:stop] = values.cpu()
            completed_converged += int(batch_metrics["converged"].sum().item())
            if progress_callback is not None:
                progress_callback(
                    {
                        "event": "batch_complete",
                        "completed_tokens": stop,
                        "vocabulary_size": vocabulary,
                        "converged_tokens": completed_converged,
                        "unresolved_tokens": stop - completed_converged,
                        "batch_max_frank_wolfe_gap": float(
                            batch_metrics["frank_wolfe_gap"].max().item()
                        ),
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                )

    def count(name: str) -> int:
        return int(per_token[name].sum().item())

    converged_count = count("converged")
    summary: dict[str, object] = {
        "solver_version": S0_SOLVER_VERSION,
        "solver_config": solver_config,
        "vocabulary_size": vocabulary,
        "hidden_size": hidden,
        "audit_dtype": "float64",
        "site_dtype": "float32",
        "bias_supplied": bias is not None,
        "bias_verified": True,
        "fp32_optimizer_available": head.weight32 is not None,
        "all_weights_identical": head.constant,
        "all_converged": converged_count == vocabulary,
        "all_sites_feasible": bool(per_token["feasible"].all()),
        "converged_count": converged_count,
        "unresolved_count": vocabulary - converged_count,
        "status_counts": {
            name: int((per_token["status"] == code).sum()) for code, name in S0_STATUS_NAMES.items()
        },
        "status_codes": {name: code for code, name in S0_STATUS_NAMES.items()},
        "cell_counts": {
            "inside": count("inside_cell"),
            "tie": count("cell_tie"),
            "outside": count("outside_cell"),
        },
        "boundary_count": count("on_boundary"),
        "interior_count": vocabulary - count("on_boundary"),
        "p1_asymptotic_certified_count": count("p1_asymptotic_certified"),
        "finite_optimum_certified_count": count("finite_optimum_certified"),
        "finite_interior_numerical_is_exact_certificate": False,
        "cell_unbounded_certified_count": count("cell_unbounded_certified"),
        "cell_bounded_certified_count": 0,
        "cell_boundedness_unknown_count": count("cell_boundedness_unknown"),
        "outside_cell_unknown_boundedness_count": int(
            (per_token["outside_cell"] & per_token["cell_boundedness_unknown"]).sum()
        ),
        "fp64_refined_token_count": count("fp64_refined"),
        "gradient_difference_fallback_count": count("gradient_difference_fallback"),
        "maximum_iterations_used": int(per_token["iterations"].max()),
        "max_frank_wolfe_gap": float(per_token["frank_wolfe_gap"].max()),
        "max_norm_over_radius": float(per_token["norm_over_radius"].max()),
        "publication_decision": "caller",
    }
    return S0OptimizationResult(sites, per_token, summary)
