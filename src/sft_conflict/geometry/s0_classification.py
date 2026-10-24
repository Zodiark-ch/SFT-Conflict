"""Optional, independent certificates for unconstrained fixed-head S0 optima.

This module neither imports nor changes the solver, configuration, or cache.
Inputs are the actual finite CPU FP32 head and recorded FP32 sites. All labels
are retained; no site is optimized, projected, replaced, or rejected here.

For an exact equal-weight group G, strict separation from every OTHER group
proves sup p_j = exp(b_j) / sum_{i in G} exp(b_i), unattained at any finite h.
Every member has this infinite-optimum certificate, but only maximal-bias
members have nonempty unbounded cells. Lower-bias members have empty cells.
When ALL weights are equal, softmax is constant instead: every label has a
finite optimum, maximal-bias cells are the whole space, and the rest are empty.
No other finite-optimum or bounded-cell certificate is attempted.

Grouping uses signed-zero-canonical SHA256 buckets AND torch.equal, never a
distance tolerance. Computation uses unique original representatives, a CPU
FP64 head with a common row subtracted, and token/vocabulary tiles. Device
storage is O(C H + B(C + H)), not a V-by-V table or an unconditional GPU head
copy. CPU storage is O(V H + V), including possible external proof directions.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import IntEnum

import torch
from torch import Tensor

S0_CLASSIFICATION_VERSION = 1
# Part of classification v1, NOT the solver/cache's 1e-5 audit tolerance.
# This is a sufficient-certificate floor, not a duplicate/normal/finite test.
S0_CLASSIFICATION_MARGIN_FLOOR = 1e-10
_PROJECTION_RELAXATION = 1.5


class S0ClassificationStatus(IntEnum):
    """Disjoint labels, independent of finite-ball optimizer convergence."""

    UNKNOWN_UNRESOLVED = 0
    P1_ASYMPTOTIC_CERTIFIED = 1
    FINITE_CONSTANT_SOFTMAX = 2
    DUPLICATE_SUPREMUM_CERTIFIED = 3


S0_CLASSIFICATION_STATUS_NAMES = {
    int(S0ClassificationStatus.UNKNOWN_UNRESOLVED): "unknown_unresolved",
    int(S0ClassificationStatus.P1_ASYMPTOTIC_CERTIFIED): "p1_asymptotic_certified",
    int(S0ClassificationStatus.FINITE_CONSTANT_SOFTMAX): "finite_constant_softmax",
    int(S0ClassificationStatus.DUPLICATE_SUPREMUM_CERTIFIED): "duplicate_supremum_certified",
}


@dataclass(frozen=True)
class S0ClassificationResult:
    """Vocabulary-ordered, detached CPU vectors and optional external rays.

    All per_token names start with ``classification_``. IDs, status and counts
    are int64, flags bool, and real values finite float64. ``representative``
    is the group's FIRST original vocabulary ID. ``norm`` is the norm of the
    verified unit direction (zero without a ray), not the search iterate norm.
    ``lower_margin`` bounds separation from all OTHER equal-weight groups.
    ``log_supremum`` is meaningful ONLY when ``infinite_certified`` is true;
    other rows contain a finite zero placeholder, not a probability claim.

    ``witness_index`` is -2 without a ray, -1 for the token's own normalized
    stored site, or a row in certificates. Every certified duplicate group
    and every successful search uses one external CPU FP64 unit ray, shared
    by the group; its int64 token_id is the original representative. The
    all-identical case uses e_0 as a whole-space recession witness, with zero
    separation margin and NO infinite-optimum claim, even for empty cells.

    ``search_iterations`` counts candidate checks, including the initial zero
    iterate. Exhaustion never certifies an unchecked update or a finite optimum.
    """

    per_token: dict[str, Tensor]
    certificates: dict[str, Tensor] | None
    summary: dict[str, object]

    @property
    def summary_json(self) -> str:
        return json.dumps(self.summary, sort_keys=True, allow_nan=False)


@dataclass(frozen=True)
class _Groups:
    representatives: Tensor
    inverse: Tensor
    sizes: Tensor


@dataclass(frozen=True)
class _Head:
    weight: Tensor  # CPU FP64 unique rows, centered once; only tiles go to CUDA.
    column_abs_max: Tensor  # FP64 [H] on the computation device.
    gamma: float


def _positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _cpu_fp32(value: Tensor, name: str) -> None:
    if (
        not isinstance(value, Tensor)
        or value.layout != torch.strided
        or value.device.type != "cpu"
        or value.dtype != torch.float32
    ):
        raise ValueError(f"{name} must be a dense CPU float32 tensor; no implicit conversion")


def _validate_inputs(weight: Tensor, bias: Tensor | None, sites: Tensor, chunk_size: int) -> None:
    _cpu_fp32(weight, "weight")
    if weight.ndim != 2 or weight.shape[0] < 2 or weight.shape[1] < 1:
        raise ValueError("weight must have shape [V,H], V >= 2 and H >= 1")
    _cpu_fp32(sites, "sites")
    if sites.shape != weight.shape:
        raise ValueError("sites must have the same [V,H] shape as weight")
    if bias is not None:
        _cpu_fp32(bias, "bias")
        if bias.shape != (len(weight),):
            raise ValueError("bias must have shape [V] or be None")
    for name, value in (("weight", weight), ("bias", bias), ("sites", sites)):
        if value is not None:
            for start in range(0, len(value), chunk_size):
                if not bool(torch.isfinite(value[start : start + chunk_size]).all()):
                    raise ValueError(f"{name} must contain only finite values")


def _exact_groups(weight: Tensor) -> _Groups:
    """SHA collisions only add exact comparisons; they can never merge rows."""
    buckets: dict[bytes, list[int]] = {}
    representatives: list[int] = []
    inverse: list[int] = []
    sizes: list[int] = []
    for token, row in enumerate(weight):
        canonical = row.resolve_neg().contiguous().clone()
        canonical[canonical == 0] = 0.0  # torch.equal equates both signed zeros.
        array = canonical.numpy()
        array = array.astype(array.dtype.newbyteorder("<"), copy=False)
        digest = hashlib.sha256(memoryview(array).cast("B")).digest()
        bucket = buckets.setdefault(digest, [])
        group = next(
            (index for index in bucket if torch.equal(row, weight[representatives[index]])),
            None,
        )
        if group is None:
            group = len(representatives)
            representatives.append(token)
            sizes.append(0)
            bucket.append(group)
        inverse.append(group)
        sizes[group] += 1
    return _Groups(
        torch.tensor(representatives, dtype=torch.int64, device="cpu"),
        torch.tensor(inverse, dtype=torch.int64, device="cpu"),
        torch.tensor(sizes, dtype=torch.int64, device="cpu"),
    )


def _prepare_head(weight: Tensor, groups: _Groups, chunk_size: int, device: torch.device) -> _Head:
    hidden = weight.shape[1]
    centered = torch.empty((len(groups.sizes), hidden), dtype=torch.float64, device="cpu")
    origin = weight[groups.representatives[0]].double()
    column_abs_max = torch.zeros(hidden, dtype=torch.float64, device="cpu")
    for start in range(0, len(centered), chunk_size):
        stop = min(start + chunk_size, len(centered))
        tile = weight[groups.representatives[start:stop]].double() - origin
        centered[start:stop] = tile
        column_abs_max = torch.maximum(column_abs_max, tile.abs().amax(dim=0))
    gamma = (hidden + 4) * torch.finfo(torch.float64).eps
    gamma /= 1.0 - gamma
    return _Head(centered, column_abs_max.to(device=device), gamma)


def _norm(values: Tensor) -> Tensor:
    """Scaled FP64 norm without an epsilon floor or small-normal underflow."""
    scale = values.abs().amax(dim=-1)
    divisor = torch.where(scale == 0, 1.0, scale)
    return scale * (values / divisor.unsqueeze(-1)).square().sum(dim=-1).sqrt()


def _linear_gaps(
    head: _Head, points: Tensor, labels: Tensor, chunk_size: int
) -> tuple[Tensor, Tensor, Tensor]:
    """Full unique-head GEMMs; capture the target BEFORE masking its group.

    Strict running comparisons preserve first-original-ID argmax tie breaking
    across chunks. Only the returned O(B) statistics outlive a vocabulary tile.
    """
    target = points.new_zeros(len(points))
    maximum = points.new_full((len(points),), -torch.finfo(torch.float64).max)
    worst = torch.zeros(len(points), dtype=torch.int64, device=points.device)
    finite = torch.ones(len(points), dtype=torch.bool, device=points.device)
    for start in range(0, len(head.weight), chunk_size):
        stop = min(start + chunk_size, len(head.weight))
        tile = head.weight[start:stop].to(device=points.device)
        logits = torch.mm(points, tile.T)
        finite &= torch.isfinite(logits).all(dim=1)
        rows = torch.nonzero((labels >= start) & (labels < stop)).flatten()
        columns = labels[rows] - start
        target[rows] = logits[rows, columns].clone()
        logits[rows, columns] = -torch.finfo(torch.float64).max
        values, indices = logits.max(dim=1)
        improve = values > maximum
        maximum = torch.where(improve, values, maximum)
        worst = torch.where(improve, indices + start, worst)
        del tile, logits
    gap = target - maximum
    return gap, worst, finite & torch.isfinite(gap)


def _roundoff(head: _Head, unit: Tensor) -> Tensor:
    return 2.0 * head.gamma * (unit.abs() * head.column_abs_max).sum(dim=-1)


def _verify_rays(
    head: _Head, candidates: Tensor, labels: Tensor, chunk_size: int
) -> tuple[Tensor, Tensor, Tensor]:
    """Reverify the ACTUAL normalized FP64 ray against every unique other row."""
    norm = _norm(candidates)
    valid = torch.isfinite(candidates).all(dim=1) & torch.isfinite(norm) & (norm > 0)
    unit = torch.zeros_like(candidates)
    unit[valid] = candidates[valid] / norm[valid, None]
    gap, _, finite = _linear_gaps(head, unit, labels, chunk_size)
    lower = gap - _roundoff(head, unit)
    certified = valid & finite & torch.isfinite(lower) & (lower > S0_CLASSIFICATION_MARGIN_FLOOR)
    return unit, torch.where(certified, lower, 0.0), certified


def _search_rays(
    head: _Head, labels: Tensor, chunk_size: int, max_iterations: int
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Bounded relaxed maximum-violation projections, starting at x=0.

    At each check, k maximizes OTHER-group W_k.x and gap=(W_j-W_k).x.
    Unless certified, x += 1.5 * (1-gap)/||W_j-W_k||^2 * (W_j-W_k).
    Failure, numerical ambiguity and exhaustion all remain unknown. No LP or
    stationarity inference is made, and an unchecked final update is not used.
    """
    count, hidden = len(labels), head.weight.shape[1]
    directions = torch.zeros((count, hidden), dtype=torch.float64, device=labels.device)
    lower_margin = directions.new_zeros(count)
    certified = torch.zeros(count, dtype=torch.bool, device=labels.device)
    iterations = torch.zeros(count, dtype=torch.int64, device=labels.device)
    active = torch.arange(count, device=labels.device)
    points = torch.zeros_like(directions)
    for iteration in range(1, max_iterations + 1):
        if not len(active):
            break
        iterations[active] = iteration
        active_labels = labels[active]
        gap, worst, finite = _linear_gaps(head, points, active_labels, chunk_size)
        norm = _norm(points)
        finite &= torch.isfinite(norm)
        divisor = torch.where((norm > 0) & finite, norm, 1.0)
        unit = points / divisor.unsqueeze(-1)
        lower = gap / divisor - _roundoff(head, unit)
        possible = finite & (norm > 0) & (lower > S0_CLASSIFICATION_MARGIN_FLOOR)
        if bool(possible.any()):
            rays, bounds, verified = _verify_rays(
                head, points[possible], active_labels[possible], chunk_size
            )
            chosen = active[possible][verified]
            directions[chosen] = rays[verified]
            lower_margin[chosen] = bounds[verified]
            certified[chosen] = True
        keep = ~certified[active] & finite
        if iteration == max_iterations or not bool(keep.any()):
            break
        active = active[keep]
        points, gap, worst = points[keep], gap[keep], worst[keep]
        # Only selected [B,H] differences, never an all-pairs difference table.
        delta = (head.weight[labels[active].cpu()] - head.weight[worst.cpu()]).to(
            device=labels.device
        )
        squared_norm = delta.square().sum(dim=1)
        usable = torch.isfinite(squared_norm) & (squared_norm > 0)
        active, points, gap, delta, squared_norm = (
            value[usable] for value in (active, points, gap, delta, squared_norm)
        )
        eta = (1.0 - gap) / squared_norm
        updated = points + (_PROJECTION_RELAXATION * eta).unsqueeze(-1) * delta
        usable = torch.isfinite(updated).all(dim=1)
        active, points = active[usable], updated[usable]
    return directions, lower_margin, certified, iterations


def _bias_facts(bias: Tensor | None, groups: _Groups) -> tuple[Tensor, Tensor]:
    values = (
        torch.zeros(len(groups.inverse), dtype=torch.float64, device="cpu")
        if bias is None
        else bias.double()
    )
    maximum = values.new_full((len(groups.sizes),), -torch.finfo(torch.float64).max)
    maximum.scatter_reduce_(0, groups.inverse, values, reduce="amax", include_self=True)
    shifted = values - maximum[groups.inverse]
    mass = values.new_zeros(len(groups.sizes))
    mass.scatter_add_(0, groups.inverse, shifted.exp())
    # Subtract the maximum FIRST, so even enormous common biases retain log |G|.
    log_cap = shifted - mass.log()[groups.inverse]
    return shifted < 0, log_cap


@torch.no_grad()
def classify_s0_optima(
    weight: Tensor,
    bias: Tensor | None,
    sites: Tensor,
    *,
    device: str | torch.device = "cpu",
    token_batch_size: int = 128,
    vocabulary_chunk_size: int = 32768,
    max_iterations: int = 1000,
) -> S0ClassificationResult:
    """Classify all V labels without changing sites or optimizer acceptance.

    First verify each representative's normalized stored site. Unresolved
    groups receive at most max_iterations zero-initialized separation checks.
    A one-check budget can therefore leave even exposed zero-site rows unknown.
    Only exact equal weights are grouped, independently of biases. Successful
    searched rays are normalized and fully reverified before being published.

    No RNG or global precision flags are changed. Autograd and autocast are
    locally disabled; all separation/search arithmetic is FP64. CPU mode never
    probes or initializes CUDA. CUDA mode transfers head tiles, not the model.
    """
    for name, value in (
        ("token_batch_size", token_batch_size),
        ("vocabulary_chunk_size", vocabulary_chunk_size),
        ("max_iterations", max_iterations),
    ):
        _positive_integer(name, value)
    if not isinstance(device, str | torch.device):
        raise ValueError("device must name a CPU or CUDA device")
    try:
        compute_device = torch.device(device)
    except (RuntimeError, ValueError) as error:
        raise ValueError("device must name a CPU or CUDA device") from error
    if compute_device.type not in {"cpu", "cuda"}:
        raise ValueError("device must name a CPU or CUDA device")
    _validate_inputs(weight, bias, sites, vocabulary_chunk_size)
    weight, sites = weight.detach(), sites.detach()
    bias = None if bias is None else bias.detach()

    with (
        torch.autocast(device_type="cpu", enabled=False),
        torch.autocast(device_type=compute_device.type, enabled=False),
    ):
        groups = _exact_groups(weight)
        unique = len(groups.sizes)
        vocabulary, hidden = weight.shape
        group_certified = torch.zeros(unique, dtype=torch.bool, device="cpu")
        group_margin = torch.zeros(unique, dtype=torch.float64, device="cpu")
        group_norm = torch.zeros_like(group_margin)
        group_iterations = torch.zeros(unique, dtype=torch.int64, device="cpu")
        group_witness = torch.full((unique,), -2, dtype=torch.int64, device="cpu")
        external: dict[int, Tensor] = {}

        def record(
            ids: Tensor, rays: Tensor, bounds: Tensor, proven: Tensor, searched: bool
        ) -> None:
            good = proven.cpu()
            chosen = ids[good]
            verified_rays = rays[proven].cpu()
            group_certified[chosen] = True
            group_margin[chosen] = bounds[proven].cpu()
            group_norm[chosen] = _norm(verified_rays)
            for offset, group in enumerate(chosen.tolist()):
                if searched or groups.sizes[group] > 1:
                    external[group] = verified_rays[offset].clone()
                else:
                    group_witness[group] = -1

        if unique == 1:
            # No outside group: softmax is constant, NOT an unattained supremum.
            ray = torch.zeros(hidden, dtype=torch.float64, device="cpu")
            ray[0] = 1.0
            external[0] = ray
            group_norm[0] = 1.0
        else:
            head = _prepare_head(weight, groups, vocabulary_chunk_size, compute_device)
            for start in range(0, unique, token_batch_size):
                ids = torch.arange(start, min(start + token_batch_size, unique), device="cpu")
                candidates = sites[groups.representatives[ids]].to(
                    device=compute_device, dtype=torch.float64
                )
                rays, bounds, proven = _verify_rays(
                    head, candidates, ids.to(device=compute_device), vocabulary_chunk_size
                )
                record(ids, rays, bounds, proven, False)
            pending = torch.nonzero(~group_certified).flatten()
            for start in range(0, len(pending), token_batch_size):
                ids = pending[start : start + token_batch_size]
                rays, bounds, proven, iterations = _search_rays(
                    head, ids.to(device=compute_device), vocabulary_chunk_size, max_iterations
                )
                group_iterations[ids] = iterations.cpu()
                record(ids, rays, bounds, proven, True)

        certificates: dict[str, Tensor] | None = None
        if external:
            ordered = sorted(external)
            certificate_groups = torch.tensor(ordered, dtype=torch.int64, device="cpu")
            certificates = {
                "directions": torch.stack([external[group] for group in ordered]),
                "token_ids": groups.representatives[certificate_groups].clone(),
            }
            group_witness[certificate_groups] = torch.arange(len(ordered), device="cpu")
        infinite = group_certified[groups.inverse]
        finite = torch.full((vocabulary,), unique == 1, dtype=torch.bool, device="cpu")
        sizes = groups.sizes[groups.inverse]
        p1 = infinite & (sizes == 1)
        empty, log_cap = _bias_facts(bias, groups)
        unbounded = (infinite | finite) & ~empty
        status = torch.full(
            (vocabulary,),
            int(S0ClassificationStatus.UNKNOWN_UNRESOLVED),
            dtype=torch.int64,
            device="cpu",
        )
        status[finite] = int(S0ClassificationStatus.FINITE_CONSTANT_SOFTMAX)
        status[infinite & ~p1] = int(S0ClassificationStatus.DUPLICATE_SUPREMUM_CERTIFIED)
        status[p1] = int(S0ClassificationStatus.P1_ASYMPTOTIC_CERTIFIED)
        per_token = {
            "classification_representative": groups.representatives[groups.inverse],
            "classification_group_size": sizes,
            "classification_status": status,
            "classification_finite_certified": finite,
            "classification_infinite_certified": infinite,
            "classification_p1_certified": p1,
            "classification_cell_unbounded_certified": unbounded,
            "classification_cell_empty_certified": empty,
            "classification_cell_bounded_certified": torch.zeros_like(finite),
            "classification_lower_margin": group_margin[groups.inverse],
            "classification_norm": group_norm[groups.inverse],
            "classification_witness_index": group_witness[groups.inverse],
            "classification_log_supremum": torch.where(infinite, log_cap, 0.0),
            "classification_search_iterations": group_iterations[groups.inverse],
        }

    finite_count, infinite_count = int(finite.sum()), int(infinite.sum())
    unbounded_count, empty_count = int(unbounded.sum()), int(empty.sum())
    summary: dict[str, object] = {
        "classification_version": S0_CLASSIFICATION_VERSION,
        "classification_config": {
            "version": S0_CLASSIFICATION_VERSION,
            "device": str(compute_device),
            "token_batch_size": token_batch_size,
            "vocabulary_chunk_size": vocabulary_chunk_size,
            "max_iterations": max_iterations,
            "separation_margin_floor": S0_CLASSIFICATION_MARGIN_FLOOR,
            "algorithm": "bounded_relaxed_max_violation_projection",
            "initialization": "zero",
            "projection_relaxation": _PROJECTION_RELAXATION,
            "projection_target_margin": 1.0,
            "iteration_count": "candidate_checks_including_initial_zero",
            "roundoff": "2*gamma_(H+4)*sum(abs(unit)*column_abs_max)",
            "gamma": "(H+4)*eps64/(1-(H+4)*eps64)",
            "floor_is_solver_audit_tolerance": False,
        },
        "vocabulary_size": vocabulary,
        "hidden_size": hidden,
        "head_dtype": "float32",
        "site_dtype": "float32",
        "audit_dtype": "float64",
        "bias_supplied": bias is not None,
        "all_weights_identical": unique == 1,
        "finite_certified_count": finite_count,
        "infinite_certified_count": infinite_count,
        "unknown_count": vocabulary - finite_count - infinite_count,
        "p1_certified_count": int(p1.sum()),
        "duplicate_supremum_certified_count": int((infinite & ~p1).sum()),
        "unique_group_count": unique,
        "duplicate_group_count": int((groups.sizes > 1).sum()),
        "duplicate_row_count": int(groups.sizes[groups.sizes > 1].sum()),
        "duplicate_excess_row_count": vocabulary - unique,
        "cell_counts": {
            "nonempty_unbounded": unbounded_count,
            "empty": empty_count,
            "bounded": 0,
            "unknown": vocabulary - unbounded_count - empty_count,
        },
        "status_codes": {name: code for code, name in S0_CLASSIFICATION_STATUS_NAMES.items()},
        "status_counts": {
            name: int((status == code).sum())
            for code, name in S0_CLASSIFICATION_STATUS_NAMES.items()
        },
        "external_witness_count": len(external),
        "site_witness_token_count": int((per_token["classification_witness_index"] == -1).sum()),
        "search_group_count": int((group_iterations > 0).sum()),
        "searched_certified_group_count": int(((group_iterations > 0) & group_certified).sum()),
        "maximum_search_iterations": int(group_iterations.max()),
        "stored_site_selection": "first_original_representative_per_exact_weight_group",
        "duplicate_equality": "signed_zero_canonical_sha256_then_torch.equal_on_original_fp32",
        "certificate_scope": "all_unique_other_groups_then_exact_membership_expansion_to_all_V",
        "log_supremum_valid_when": "classification_infinite_certified",
        "constant_group_witness": "e_0_recession_ray_only_not_strict_separation",
        "finite_classification_scope": "all_weights_identical_only",
        "general_lp_solved": False,
        "sites_modified": False,
        "training_eligibility_modified": False,
        "limitations": [
            "Sufficient certificates only; exhaustion or failed separation means unknown, "
            "not finite.",
            "Small gradients and ball boundaries are not finite/infinite-optimum certificates.",
            "Cell emptiness is proved only by equal-weight, strictly greater-bias domination.",
            "Only maximal-bias members of exposed groups have nonempty unbounded cells.",
            "No general finite-optimum, cell-feasibility, or cell-boundedness LP is solved.",
            "Rounding an FP32 gauge shift can change the input head; "
            "such shifts are not invariances.",
        ],
    }
    return S0ClassificationResult(per_token, certificates, summary)
