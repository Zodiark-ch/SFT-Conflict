"""Immutable, head-bound, vocabulary-ordered S0 Safetensors publications.

This module deliberately has no dataset, model-loading, or solver dependency.
Callers supply the *actual* frozen FP32 head and the solver's post-FP32-export
audit. Required ``per_token`` vectors are ``converged`` (bool), ``cell_margin``,
``optimality_gap``, ``site_norm``, ``log_probability`` (float64), and
``unbounded_certified`` (bool). Optional ``token_ids`` must be int64 arange(V);
otherwise it is added. Extra finite, one-dimensional numeric/bool vectors are
preserved. Solver names ``frank_wolfe_gap``, ``p1_asymptotic_certified``, and
``token_id`` acquire these canonical aliases without deleting the original keys.
Optional bool ``finite_values`` (legacy alias ``finite_numerical``) reports
numerical finiteness, never a certificate of an attained finite optimum. Payload
values are always checked independently; a supplied false flag also blocks training.
A true unbounded certificate needs a supplied float64
``p1_directional_margin_lower_bound`` > audit_tolerance; optional float64
``p1_direction_norm`` must be positive and approximately one. This proves
p_j tends to one along a separating ray, NOT finite stationarity or convergence
on the ball. No gradient-norm threshold applies, and it never enables training.

Optional independent classification rays live in ``certificates.safetensors``,
not JSON or replacement sites: ``directions`` is CPU float64 [M, H], with finite,
nonzero approximately unit rows; ``token_ids`` is unique CPU int64 [M] in [0, V).
Only supplied rows are stored, in their supplied order. ``None``, an empty dict,
or a valid zero-row table preserves the original three-file publication.
Optional int64 ``classification_witness_index`` audit vectors use -1 for the
normalized recorded site, -2 for no witness, and nonnegative certificate row
indices. Duplicate-group tokens may share an external row. When supplied,
``classification_representative`` must match that row's certificate token ID,
not necessarily the audited token ID. These are structural bindings only: the
classifier must verify rays against the head. Its version/evidence in ``summary``
is manifest-bound; none of these fields change site-only P1 checks or training.

``solver_config`` is a JSON dict with radius > 0, gap_tolerance >= 0, and
audit_tolerance >= 0 (normally ``asdict(S0Config)``). Every setting is hashed
together with the versioned solver contract for binding. Actual backend/solver
details belong in ``summary`` and are covered by the manifest digest, not merged
into the caller's configuration. The objective is full-vocabulary log-softmax
maximization in a closed L2 ball.
``cell_margin`` is the minimum *unnormalized logit difference* to other labels;
``optimality_gap`` is the nonnegative Frank-Wolfe gap for negative log probability.

All row norms and eligibility masks are independently recomputed. Small heads
also receive an independent full-vocabulary audit of the original FP32 W/b at
the stored sites: FP64 explicit differences, stable other-mass gradients, and
scaled norms/gaps. Separation bounds use the normalized site (or the solver's
centered-row direction at a zero site), with exact arithmetic only to resolve
FP64 roundoff-ambiguous bounds. Zero directions cannot certify separation. Large
heads retain the solver's numerical evidence without repeating O(V**2 H) work;
the manifest explicitly states that limitation. Checksums bind recorded evidence
to inputs and detect corruption, not maliciously fabricated solver evidence.
Callers needing independent large-head certification must run it separately.

``complete`` means all V rows are present, NOT that optimization succeeded.
Contract v3 training requires ALL rows to have finite values, ball feasibility,
reported convergence, and a Frank-Wolfe gap <= gap_tolerance. Cell type is ONLY
diagnostic: strict_inside is margin > tol, tolerance_band is abs(margin) <= tol,
and outside_cell is margin < -tol. These three counts partition all rows.
in_closed_cell is margin >= 0 (membership, NOT a claim of cell boundedness).
Tied and outside anchors use the exact recorded sites, without filtering,
replacement, or imputation. Numerically unresolved rows remain diagnostic-only.
Earlier contracts are rejected, not reinterpreted under the v3 policy.
``validate_binding`` must succeed before a caller uses sites for training.
"""

from __future__ import annotations

import copy
import ctypes
import hashlib
import json
import math
import os
import re
import shutil
import stat
import tempfile
from fractions import Fraction
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch
from safetensors.torch import load, save_file

S0_CACHE_SCHEMA_VERSION = 1
# Acceptance semantics, independent of the compatible tensor schema and optimizer v1.
S0_CACHE_CONTRACT_VERSION = 3
S0_REQUIRED_PER_TOKEN_FIELDS = frozenset(
    {
        "converged",
        "cell_margin",
        "optimality_gap",
        "site_norm",
        "log_probability",
        "unbounded_certified",
    }
)

_SOLVER_CONTRACT = {
    "version": S0_CACHE_CONTRACT_VERSION,
    "solver_version": 1,
    "objective": "maximize_full_vocabulary_log_softmax_j(W @ h + b)",
    "domain": "closed_l2_ball_centered_at_zero",
    "gap": "grad(-log_p_j).dot(h) + radius * norm(grad(-log_p_j))",
    "cell_margin": "min_over_k_ne_j((W_j-W_k).dot(h)+b_j-b_k)",
    "training_cell": "diagnostic_only_not_a_training_requirement",
    "cell_counts": {
        "strict_inside": "cell_margin > audit_tolerance",
        "tolerance_band": "abs(cell_margin) <= audit_tolerance",
        "outside_cell": "cell_margin < -audit_tolerance",
        "in_closed_cell": "cell_margin >= 0; membership_not_cell_boundedness",
    },
    "training_acceptance": "all_rows_finite_feasible_reported_converged_and_gap_within_tolerance",
    "ball_feasibility": "fp64_site_norm <= radius + audit_tolerance",
    "convergence": "reported_converged and optimality_gap <= gap_tolerance",
    "unbounded_certificate": (
        "supplied_fp64_p1_directional_margin_lower_bound > audit_tolerance; "
        "min_over_k_ne_j((W_j-W_k).dot(unit_direction)) >= lower_bound"
    ),
    "certificate_direction": "normalized_site_or_centered_head_row_when_site_is_zero",
    "certificate_direction_norm": "positive_and_approximately_one_if_supplied",
    "certificate_convergence": "independent_of_gradient_norm_and_ball_convergence",
    "finite_values": "finite_values_not_an_exact_finite_optimum_certificate",
    "audit_at": "stored_float32_sites",
    "audit_scope": "all_vocabulary_competitors_no_topk",
}
_INDEPENDENT_AUDIT_WORK_LIMIT = 2_000_000  # V * V * H; no large CPU solver replay.
_MANIFEST_LIMIT = 16 * 1024**2
_FLOAT_FIELDS = {"cell_margin", "optimality_gap", "site_norm", "log_probability"}
_BOOL_FIELDS = {"converged", "unbounded_certified", "finite_values", "finite_numerical"}
_PER_TOKEN_ALIASES = {
    "frank_wolfe_gap": "optimality_gap",
    "p1_asymptotic_certified": "unbounded_certified",
    "token_id": "token_ids",
    "finite_numerical": "finite_values",
}
_OPTIONAL_FLOAT_FIELDS = {
    "gradient_norm",
    "normal_norm",
    "projected_gradient_norm",
    "p1_directional_margin_lower_bound",
    "p1_direction_norm",
}
_OPTIONAL_INT64_FIELDS = {"classification_witness_index", "classification_representative"}
_DTYPES = {
    torch.bool,
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.float32,
    torch.float64,
}
_DERIVED_FIELDS = {
    "cache_site_norm",
    "cache_in_ball",
    "cache_closed_cell",
    "cache_strict_inside",
    "cache_converged",
    "cache_usable",
}
_FILE_NAMES = {"sites.safetensors", "audit.safetensors"}
_CERTIFICATES_FILE = "certificates.safetensors"
_CERTIFICATE_FIELDS = {"directions", "token_ids"}
_MANIFEST_KEYS = {
    "schema_version",
    "anchor_mode",
    "format",
    "complete",
    "usable_for_training",
    "vocabulary_size",
    "hidden_size",
    "row_order",
    "storage_dtype",
    "model_id",
    "model_revision",
    "head_sha256",
    "head",
    "solver_contract",
    "solver_config",
    "solver_config_sha256",
    "solver_sha256",
    "sites_sha256",
    "audit_provenance",
    "counts",
    "solver_summary",
    "files",
    "manifest_sha256",
}


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")


def _json_digest(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _json_copy(value: Any, name: str) -> Any:
    """Reject lossy JSON coercions (tuple, integer keys, NaN, tensors, etc.)."""
    if value is None or type(value) in (str, bool):
        return value
    if type(value) in (int, float):
        try:
            finite = math.isfinite(value)
        except OverflowError:
            finite = False
        if not finite:
            raise ValueError(f"{name} must contain only finite JSON numbers")
        return value
    if isinstance(value, list):
        return [_json_copy(item, name) for item in value]
    if isinstance(value, dict) and all(type(key) is str for key in value):
        return {key: _json_copy(item, f"{name}.{key}") for key, item in value.items()}
    raise ValueError(f"{name} must be JSON data with string object keys")


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _config(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("solver_config must be a JSON dict")
    result = _json_copy(value, "solver_config")
    for key in ("radius", "gap_tolerance", "audit_tolerance"):
        item = result.get(key)
        if type(item) not in (int, float) or item < 0 or (key == "radius" and item == 0):
            bound = "positive" if key == "radius" else "non-negative"
            raise ValueError(f"solver_config.{key} must be finite and {bound}")
    if not math.isfinite(result["radius"] + result["audit_tolerance"]):
        raise ValueError("solver_config.radius + audit_tolerance must be finite")
    if "config_version" in result and (
        type(result["config_version"]) is not int or result["config_version"] != 1
    ):
        raise ValueError("solver_config.config_version must be the integer 1")
    for key in ("token_batch_size", "vocabulary_chunk_size", "max_iterations"):
        if key in result and (type(result[key]) is not int or result[key] < 1):
            raise ValueError(f"solver_config.{key} must be a positive integer")
    if "initial_step_size" in result and (
        type(result["initial_step_size"]) not in (int, float) or result["initial_step_size"] <= 0
    ):
        raise ValueError("solver_config.initial_step_size must be a positive finite number")
    return result


def _tensor(value: Any, name: str, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.layout != torch.strided:
        raise ValueError(f"{name} must be a dense Tensor")
    if dtype is not None and value.dtype != dtype:
        raise ValueError(f"{name} must be {dtype}; implicit dtype conversion is forbidden")
    if value.device.type == "meta" or value.dtype not in _DTYPES:
        raise ValueError(f"{name} has an unsupported device or dtype")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")
    return value


def _head(weight: torch.Tensor, bias: torch.Tensor | None) -> tuple[int, int]:
    _tensor(weight, "weight", dtype=torch.float32)
    if weight.ndim != 2 or weight.shape[0] < 2 or weight.shape[1] < 1:
        raise ValueError("weight must have shape [V, H] with V >= 2 and H >= 1")
    if bias is not None:
        _tensor(bias, "bias", dtype=torch.float32)
        if bias.shape != (weight.shape[0],):
            raise ValueError("bias must have shape [V]")
    return int(weight.shape[0]), int(weight.shape[1])


def _update_tensor_digest(digest: Any, tensor: torch.Tensor) -> None:
    tensor = tensor.detach().resolve_conj().resolve_neg().cpu().contiguous()
    metadata = _json_bytes({"dtype": str(tensor.dtype), "shape": list(tensor.shape)})
    digest.update(len(metadata).to_bytes(8, "little"))
    digest.update(metadata)
    array = tensor.numpy()
    # Normalize byte order, not precision. The logical dtype was checked above.
    array = array.astype(array.dtype.newbyteorder("<"), copy=False)
    digest.update(memoryview(array).cast("B"))


def head_fingerprint(weight: torch.Tensor, bias: torch.Tensor | None) -> str:
    """Hash exact FP32 logical values, dtype, shape, and explicit bias presence.

    Noncontiguous views have the same fingerprint as their contiguous copy.
    None and an explicit zero bias differ; lower/higher precision heads are
    rejected rather than silently rounded. CPU input never initializes CUDA.
    """
    _head(weight, bias)
    digest = hashlib.sha256(b"sft-conflict:s0-head:fp32-le:v1\0weight\0")
    _update_tensor_digest(digest, weight)
    digest.update(b"\0bias:none\0" if bias is None else b"\0bias:tensor\0")
    if bias is not None:
        _update_tensor_digest(digest, bias)
    return digest.hexdigest()


def _tensor_digest(tensor: torch.Tensor) -> str:
    digest = hashlib.sha256(b"sft-conflict:s0-tensor:le:v1\0")
    _update_tensor_digest(digest, tensor)
    return digest.hexdigest()


def _tensor_spec(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "shape": list(tensor.shape),
        "sha256": _tensor_digest(tensor),
    }


def _scaled_norm(value: torch.Tensor) -> torch.Tensor:
    """FP64 Euclidean norm without squaring away tiny nonzero components."""
    value = value.to(torch.float64)
    scale = value.abs().amax(dim=-1)
    divisor = torch.where(scale == 0, 1.0, scale)
    return scale * (value / divisor.unsqueeze(-1)).square().sum(dim=-1).sqrt()


def _site_norms(sites: torch.Tensor) -> torch.Tensor:
    result = torch.empty(sites.shape[0], dtype=torch.float64)
    for start in range(0, sites.shape[0], 256):
        result[start : start + 256] = _scaled_norm(sites[start : start + 256])
    return result


def _certificate_vectors(
    certificates: Any, vocab: int, hidden: int
) -> dict[str, torch.Tensor] | None:
    """Validate and snapshot external witnesses without normalizing or expanding them."""
    if certificates is None:
        return None
    if not isinstance(certificates, dict):
        raise ValueError("certificates must be a dict of directions and token_ids")
    if not certificates:
        return None
    if set(certificates) != _CERTIFICATE_FIELDS:
        raise ValueError("certificates must contain only directions and token_ids")
    for name, dtype in (("directions", torch.float64), ("token_ids", torch.int64)):
        value = certificates[name]
        if not isinstance(value, torch.Tensor) or value.device.type != "cpu":
            raise ValueError(f"certificates.{name} must be a CPU Tensor")
        _tensor(value, f"certificates.{name}", dtype=dtype)
    directions, token_ids = certificates["directions"], certificates["token_ids"]
    if directions.ndim != 2 or directions.shape[1] != hidden or directions.shape[0] > vocab:
        raise ValueError("certificates.directions must have shape [M, H] with 0 <= M <= V")
    rows = directions.shape[0]
    if token_ids.shape != (rows,):
        raise ValueError("certificates.token_ids must have shape [M]")
    if bool(((token_ids < 0) | (token_ids >= vocab)).any()):
        raise ValueError("certificates.token_ids must be in [0, V)")
    if token_ids.unique().numel() != rows:
        raise ValueError("certificates.token_ids must be unique")
    if rows == 0:
        return None
    norms = _site_norms(directions.detach())
    if bool((norms <= 0).any()) or not torch.allclose(
        norms, torch.ones_like(norms), rtol=1e-10, atol=1e-12
    ):
        raise ValueError("certificates.directions must be nonzero and approximately unit norm")
    return {name: value.detach().contiguous().clone() for name, value in certificates.items()}


def _check_classification_witnesses(
    audit: dict[str, torch.Tensor], certificates: dict[str, torch.Tensor] | None
) -> None:
    """Bind external row references, allowing duplicate-group representative reuse."""
    indices = audit.get("classification_witness_index")
    if indices is None:
        return
    if bool((indices < -2).any()):
        raise ValueError("per_token.classification_witness_index must be -2, -1, or a row index")
    external = indices >= 0
    if not bool(external.any()):
        return
    if certificates is None:
        raise ValueError("per_token.classification_witness_index requires external certificates")
    rows = indices[external]
    if bool((rows >= certificates["directions"].shape[0]).any()):
        raise ValueError("per_token.classification_witness_index is out of certificates range")
    representatives = audit.get("classification_representative")
    if representatives is not None and not torch.equal(
        certificates["token_ids"][rows], representatives[external]
    ):
        raise ValueError(
            "per_token.classification_representative disagrees with certificates.token_ids "
            "at classification_witness_index"
        )


def _audit_vectors(per_token: Any, vocab: int) -> dict[str, torch.Tensor]:
    if not isinstance(per_token, dict):
        raise ValueError("per_token must be a dict of one vector per vocabulary token")
    per_token = dict(per_token)
    for source, target in _PER_TOKEN_ALIASES.items():
        if source in per_token and target not in per_token:
            per_token[target] = per_token[source]
    if missing := S0_REQUIRED_PER_TOKEN_FIELDS - per_token.keys():
        raise ValueError(f"per_token is missing required fields: {sorted(missing)}")
    result = {}
    for name, value in per_token.items():
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):
            raise ValueError("per_token field names must be simple identifiers")
        if name.startswith("cache_") or name == "sites":
            raise ValueError(f"per_token field {name} is reserved")
        _tensor(value, f"per_token.{name}")
        if value.device.type != "cpu" or value.shape != (vocab,):
            raise ValueError(f"per_token.{name} must be a CPU vector of length V")
        if name in _FLOAT_FIELDS | _OPTIONAL_FLOAT_FIELDS and value.dtype != torch.float64:
            raise ValueError(f"per_token.{name} must be float64")
        if name in _BOOL_FIELDS and value.dtype != torch.bool:
            raise ValueError(f"per_token.{name} must be bool")
        if name in _OPTIONAL_INT64_FIELDS and value.dtype != torch.int64:
            raise ValueError(f"per_token.{name} must be int64")
        result[name] = value.detach().contiguous().clone()
    for source, target in _PER_TOKEN_ALIASES.items():
        if source in result and (
            result[source].dtype != result[target].dtype
            or not torch.equal(result[source], result[target])
        ):
            raise ValueError(f"per_token aliases disagree: {source} and {target}")
    if "token_ids" not in result:
        result["token_ids"] = torch.arange(vocab, dtype=torch.int64)
    if result["token_ids"].dtype != torch.int64 or not torch.equal(
        result["token_ids"], torch.arange(vocab, dtype=torch.int64)
    ):
        raise ValueError("per_token.token_ids must be int64 arange(V) in vocabulary order")
    representatives = result.get("classification_representative")
    if representatives is not None and bool(
        ((representatives < 0) | (representatives >= vocab)).any()
    ):
        raise ValueError("per_token.classification_representative must be in [0, V)")
    for name in (
        "optimality_gap",
        "site_norm",
        "gradient_norm",
        "normal_norm",
        "projected_gradient_norm",
        "p1_direction_norm",
    ):
        if name in result and bool((result[name] < 0).any()):
            raise ValueError(f"per_token.{name} must be non-negative")
    if bool((result["log_probability"] > 0).any()):
        raise ValueError("per_token.log_probability must be <= 0")
    return result


def _derived_audit(
    sites: torch.Tensor, audit: dict[str, torch.Tensor], config: dict[str, Any]
) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    norms = _site_norms(sites)
    if not torch.allclose(audit["site_norm"], norms, rtol=1e-10, atol=0):
        raise ValueError("per_token.site_norm disagrees with the actual FP32 sites")
    certified = audit["unbounded_certified"]
    if bool(certified.any()):
        bound = audit.get("p1_directional_margin_lower_bound")
        if bound is None or bool((bound[certified] <= config["audit_tolerance"]).any()):
            raise ValueError(
                "unbounded_certified requires float64 p1_directional_margin_lower_bound "
                "> audit_tolerance"
            )
        if "p1_direction_norm" in audit:
            direction_norm = audit["p1_direction_norm"][certified]
            if bool((direction_norm <= 0).any()) or not torch.allclose(
                direction_norm, torch.ones_like(direction_norm), rtol=1e-10, atol=1e-12
            ):
                raise ValueError("certified p1_direction_norm must be positive and near one")
    in_ball = norms <= config["radius"] + config["audit_tolerance"]
    margin = audit["cell_margin"]
    closed = margin >= 0
    strict = margin > config["audit_tolerance"]
    tolerance_band = margin.abs() <= config["audit_tolerance"]
    outside = margin < -config["audit_tolerance"]
    gap_ok = audit["optimality_gap"] <= config["gap_tolerance"]
    converged = audit["converged"] & gap_ok
    # _tensor has already rejected nonfinite sites and every supplied audit value.
    # A solver's explicit negative finiteness report still fails closed. Cell
    # membership and unconstrained optimum certificates never gate this mask.
    finite = audit.get("finite_values", torch.ones(sites.shape[0], dtype=torch.bool))
    usable = converged & in_ball & finite
    derived = {
        "cache_site_norm": norms,
        "cache_in_ball": in_ball,
        "cache_closed_cell": closed,
        "cache_strict_inside": strict,
        "cache_converged": converged,
        "cache_usable": usable,
    }
    masks = {
        "reported_converged": audit["converged"],
        "converged": converged,
        "gap_within_tolerance": gap_ok,
        "in_ball": in_ball,
        "in_closed_cell": closed,
        "strict_inside": strict,
        "tolerance_band": tolerance_band,
        "outside_cell": outside,
        "finite_values": finite,
        "unbounded_certified": certified,
        "usable": usable,
    }
    counts = {"rows": int(sites.shape[0]), **{key: int(mask.sum()) for key, mask in masks.items()}}
    return derived, counts


def _provenance(vocab: int, hidden: int) -> dict[str, Any]:
    return {
        "numerical_evidence": "solver_supplied_post_fp32_export_full_vocabulary",
        "site_norms": "independent_cpu_float64_all_rows",
        "small_head_full_reaudit": vocab * vocab * hidden <= _INDEPENDENT_AUDIT_WORK_LIMIT,
        "small_head_gradient": "fp64_explicit_weight_differences_log_other_mass_scaled_norm",
        "small_head_certificate": "original_fp32_head_separation_with_exact_roundoff_fallback",
        "full_reaudit_work_limit": _INDEPENDENT_AUDIT_WORK_LIMIT,
        "authenticity_signature": False,
    }


def _check_separation_bound(
    target: torch.Tensor,
    competitors: torch.Tensor,
    direction: torch.Tensor,
    lower_bound: torch.Tensor,
) -> None:
    """Check the supplied lower bound, not just the sign of a rounded GEMM.

    Fast FP64 intervals suffice for conservative solver bounds. In the narrow
    roundoff band, exact FP32 weights and the actual FP64 direction distinguish
    a legitimate equality from an overclaim by even one ULP. No audit-tolerance
    slack can turn a zero/negative separating margin into a certificate.
    """
    if not bool(_scaled_norm(direction) > 0):
        raise ValueError("p1_direction_norm failed independent audit: zero direction")
    differences = target - competitors
    margins = differences @ direction
    gamma = (direction.numel() + 2) * torch.finfo(torch.float64).eps
    gamma /= 1.0 - gamma
    roundoff = gamma * (differences.abs() @ direction.abs())
    if bool((lower_bound > margins + roundoff).any()):
        raise ValueError("p1_directional_margin_lower_bound failed independent separation audit")
    uncertain = torch.nonzero(lower_bound > margins - roundoff).flatten().tolist()
    if not uncertain:
        return
    exact_target = [Fraction(value) for value in target.tolist()]
    exact_direction = [Fraction(value) for value in direction.tolist()]
    exact_bound = Fraction(float(lower_bound))
    for row in uncertain:
        exact_margin = sum(
            (gold - Fraction(other)) * component
            for gold, other, component in zip(
                exact_target, competitors[row].tolist(), exact_direction, strict=True
            )
        )
        if exact_bound > exact_margin:
            raise ValueError(
                "p1_directional_margin_lower_bound is not a conservative separation bound"
            )


def _independent_small_audit(
    sites: torch.Tensor,
    audit: dict[str, torch.Tensor],
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    config: dict[str, Any],
) -> None:
    vocab, hidden = sites.shape
    if not _provenance(vocab, hidden)["small_head_full_reaudit"]:
        return
    w = weight.detach().cpu().to(torch.float64)
    b = torch.zeros(vocab, dtype=torch.float64) if bias is None else bias.detach().cpu().double()
    ids = torch.arange(vocab)
    for token in range(vocab):
        x = sites[token].to(torch.float64)
        others = ids != token
        competitors = w[others]
        differences = competitors - w[token]
        relative = differences @ x + (b[others] - b[token])
        log_odds = torch.logsumexp(relative, dim=0)
        loss = torch.logaddexp(torch.zeros_like(log_odds), log_odds)
        log_conditional = relative - log_odds
        weighted = differences * log_conditional.exp().unsqueeze(-1)
        subnormal = log_conditional < math.log(torch.finfo(torch.float64).tiny)
        if bool(subnormal.any()):
            values = differences[subnormal]
            weighted[subnormal] = (
                values.sign() * (values.abs().log() + log_conditional[subnormal, None]).exp()
            )
        conditional_gradient = weighted.sum(dim=0)
        normal = _scaled_norm(conditional_gradient)
        norm = _scaled_norm(x)
        unit_h = x / torch.where(norm == 0, 1.0, norm)
        unit_g = conditional_gradient / torch.where(normal == 0, 1.0, normal)
        # Recover representable norms/gaps even if the other mass underflows;
        # genuine final underflow may be recorded as zero, never as exact proof.
        log_gradient_norm = (log_odds - loss) + normal.log()
        distance = (
            (config["radius"] - norm) + 0.5 * norm * (unit_h + unit_g).square().sum()
        ).clamp_min(0)
        expected = {
            "cell_margin": -relative.max(),
            "log_probability": -loss,
            "optimality_gap": (log_gradient_norm + distance.log()).exp(),
        }
        if "gradient_norm" in audit:
            expected["gradient_norm"] = log_gradient_norm.exp()
        # Cancellation between genuinely opposing differences can round an
        # interior residual differently. Scale the comparison allowance by
        # the actual other-mass contributions, never an absolute 1e-10 floor.
        # This retains tiny confident derivatives/gaps while tolerating FP64
        # summation roundoff near a numerical stationary point.
        roundoff_factor = 32 * (vocab + hidden + 4) * torch.finfo(torch.float64).eps
        log_error = (
            (log_odds - loss)
            + _scaled_norm(weighted.abs().sum(dim=0)).log()
            + math.log(roundoff_factor)
        )
        gradient_error = float(log_error.exp())
        gap_error = float((log_error + math.log(config["radius"] + float(norm))).exp())
        for name, actual in expected.items():
            atol = {
                "cell_margin": 1e-10,
                "log_probability": 8 * math.ulp(0.0),
                "gradient_norm": max(8 * math.ulp(0.0), gradient_error),
                "optimality_gap": max(8 * math.ulp(0.0), gap_error),
            }[name]
            if not torch.allclose(audit[name][token], actual, rtol=1e-8, atol=atol):
                raise ValueError(f"per_token.{name} failed independent FP64 full-vocabulary audit")
        if bool(audit["unbounded_certified"][token]):
            direction = unit_h
            if bool(norm == 0):
                centered = w - w[0]
                normal_row = centered[token] - centered.mean(dim=0)
                normal_length = _scaled_norm(normal_row)
                direction = normal_row / torch.where(normal_length == 0, 1.0, normal_length)
            _check_separation_bound(
                w[token],
                competitors,
                direction,
                audit["p1_directional_margin_lower_bound"][token],
            )


def _safe_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if ".." in path.parts:
        raise ValueError("S0 cache paths must not contain '..'")
    path = path.absolute()
    for component in (path, *path.parents):
        if component.is_symlink():
            raise ValueError(f"S0 cache paths must not contain symlinks: {component}")
    return path


def _read_regular(path: Path, *, size: int | None = None, limit: int | None = None) -> bytes:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, "rb") as handle:
            info = os.fstat(handle.fileno())
            if not stat.S_ISREG(info.st_mode):
                raise ValueError(f"S0 cache member must be a regular file: {path.name}")
            if size is not None and info.st_size != size:
                raise ValueError(f"S0 cache file size mismatch: {path.name}")
            if limit is not None and info.st_size > limit:
                raise ValueError(f"S0 cache member is too large: {path.name}")
            return handle.read()
    except OSError as error:
        raise ValueError(f"missing or unsafe S0 cache file: {path.name}") from error


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate manifest key: {key}")
        result[key] = value
    return result


def _certificate_file_limit(entry: dict[str, Any], vocab: int, hidden: int) -> int:
    """Bound an untrusted witness file before reading or loading its payload."""
    specs = entry["tensors"]
    if set(specs) != _CERTIFICATE_FIELDS:
        raise ValueError("certificates.safetensors must contain only directions and token_ids")
    for name, dtype, rank in (("directions", "float64", 2), ("token_ids", "int64", 1)):
        spec = specs[name]
        if not isinstance(spec, dict) or set(spec) != {"dtype", "shape", "sha256"}:
            raise ValueError(f"invalid S0 cache certificates.{name} metadata")
        shape = spec["shape"]
        if (
            spec["dtype"] != dtype
            or not isinstance(shape, list)
            or len(shape) != rank
            or any(type(dimension) is not int or dimension < 0 for dimension in shape)
            or not isinstance(spec["sha256"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", spec["sha256"])
        ):
            raise ValueError(f"invalid S0 cache certificates.{name} spec")
    rows, width = specs["directions"]["shape"]
    if not 1 <= rows <= vocab or width != hidden:
        raise ValueError("certificates.directions must have shape [M, H] with 1 <= M <= V")
    if specs["token_ids"]["shape"] != [rows]:
        raise ValueError("certificates.token_ids must have shape [M]")
    # Eight header-length bytes, at most the existing JSON limit for the header,
    # and exactly two shape-bounded FP64/int64 payloads; never trust size_bytes alone.
    return 8 + _MANIFEST_LIMIT + 8 * rows * (hidden + 1)


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(
            _read_regular(path / "manifest.json", limit=_MANIFEST_LIMIT),
            object_pairs_hook=_unique_json_object,
        )
        manifest = _json_copy(manifest, "manifest")
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid S0 cache manifest JSON") from error
    if not isinstance(manifest, dict) or set(manifest) != _MANIFEST_KEYS:
        raise ValueError("invalid S0 cache manifest fields")
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if manifest["manifest_sha256"] != _json_digest(unsigned):
        raise ValueError("S0 cache manifest SHA256 mismatch")
    constants = {
        "schema_version": S0_CACHE_SCHEMA_VERSION,
        "anchor_mode": "recorded_s0",
        "format": "safetensors",
        "complete": True,
        "row_order": "token_id_0_to_V_minus_1",
        "storage_dtype": "float32",
        "solver_contract": _SOLVER_CONTRACT,
    }
    for name, value in constants.items():
        if _json_bytes(manifest[name]) != _json_bytes(value):
            raise ValueError(f"invalid S0 cache {name}")
    for name, lower in (("vocabulary_size", 2), ("hidden_size", 1)):
        if type(manifest[name]) is not int or manifest[name] < lower:
            raise ValueError(f"invalid S0 cache {name}")
    if type(manifest["usable_for_training"]) is not bool:
        raise ValueError("invalid S0 cache usable_for_training flag")
    for name in ("model_id", "model_revision"):
        _text(manifest[name], name)
    for name in ("head_sha256", "sites_sha256"):
        if not isinstance(manifest[name], str) or not re.fullmatch(r"[0-9a-f]{64}", manifest[name]):
            raise ValueError(f"invalid S0 cache {name}")
    config = _config(manifest["solver_config"])
    if manifest["solver_config_sha256"] != _json_digest(config) or manifest[
        "solver_sha256"
    ] != _json_digest({"contract": _SOLVER_CONTRACT, "config": config}):
        raise ValueError("S0 cache solver SHA256 mismatch")
    vocab, hidden = manifest["vocabulary_size"], manifest["hidden_size"]
    head = manifest["head"]
    if not isinstance(head, dict) or set(head) != {"weight_shape", "bias_shape", "dtype"}:
        raise ValueError("invalid S0 cache head metadata")
    if _json_bytes(head) not in (
        _json_bytes({"weight_shape": [vocab, hidden], "bias_shape": None, "dtype": "float32"}),
        _json_bytes({"weight_shape": [vocab, hidden], "bias_shape": [vocab], "dtype": "float32"}),
    ):
        raise ValueError("invalid S0 cache head dimensions or dtype")
    if _json_bytes(manifest["audit_provenance"]) != _json_bytes(_provenance(vocab, hidden)):
        raise ValueError("invalid S0 cache audit provenance")
    if not isinstance(manifest["solver_summary"], dict):
        raise ValueError("invalid S0 cache solver_summary")
    if not isinstance(manifest["files"], dict) or not (
        _FILE_NAMES <= manifest["files"].keys() <= _FILE_NAMES | {_CERTIFICATES_FILE}
    ):
        raise ValueError(
            "S0 cache files must be the fixed local Safetensors members "
            "with optional certificates.safetensors"
        )
    for name, entry in manifest["files"].items():
        if not isinstance(entry, dict) or set(entry) != {"sha256", "size_bytes", "tensors"}:
            raise ValueError(f"invalid S0 cache file metadata: {name}")
        if type(entry["size_bytes"]) is not int or entry["size_bytes"] < 1:
            raise ValueError(f"invalid S0 cache file size: {name}")
        if not isinstance(entry["sha256"], str) or not re.fullmatch(
            r"[0-9a-f]{64}", entry["sha256"]
        ):
            raise ValueError(f"invalid S0 cache file SHA256: {name}")
        if not isinstance(entry["tensors"], dict):
            raise ValueError(f"invalid S0 cache tensor metadata: {name}")
        if name == _CERTIFICATES_FILE and entry["size_bytes"] > _certificate_file_limit(
            entry, vocab, hidden
        ):
            raise ValueError(f"S0 cache member is too large: {name}")
    return manifest


class S0Cache:
    """Read-only publication handle; construction reads the manifest, not tensors."""

    def __init__(self, path: str | Path) -> None:
        self.path = _safe_path(path)
        if not self.path.is_dir():
            raise ValueError("S0 cache path must be an existing directory")
        self._manifest = _read_manifest(self.path)
        self._sites: torch.Tensor | None = None
        self._certificates: dict[str, torch.Tensor] | None = None

    @property
    def manifest(self) -> dict[str, Any]:
        return copy.deepcopy(self._manifest)

    @property
    def sites(self) -> torch.Tensor:
        """Lazy CPU FP32 table; return a copy so callers cannot mutate the handle."""
        if self._sites is None:
            self.validate_integrity()
        assert self._sites is not None
        return self._sites.clone()

    @property
    def certificates(self) -> dict[str, torch.Tensor] | None:
        """Lazy integrity-checked CPU witnesses, defensively copied; absent means None.

        Like ``sites``, this caches a validated snapshot. Explicit integrity/binding
        validation always rereads the files. No missing ray is inferred from sites.
        """
        if self._sites is None:
            self.validate_integrity()
        if self._certificates is None:
            return None
        return {name: value.clone() for name, value in self._certificates.items()}

    def identity(self) -> dict[str, Any]:
        """Stable identity of this snapshot (not a substitute for binding validation)."""
        return {
            key: self._manifest[key]
            for key in (
                "anchor_mode",
                "head_sha256",
                "solver_sha256",
                "sites_sha256",
                "manifest_sha256",
                "schema_version",
            )
        }

    def validate_integrity(self) -> None:
        """Recheck files, tensors, row order, masks, norms, and authoritative counts.

        Does not need W/b and cannot independently recalculate cell/gap evidence.
        It rereads even after a previous successful validation; no stale OK flag.
        """
        path = _safe_path(self.path)
        manifest = _read_manifest(path)
        if _json_bytes(manifest) != _json_bytes(self._manifest):
            raise ValueError("S0 cache manifest changed after the handle was opened")
        if {member.name for member in path.iterdir()} != set(manifest["files"]) | {"manifest.json"}:
            raise ValueError("S0 cache has missing or unexpected files")
        tensors = {}
        vocab, hidden = manifest["vocabulary_size"], manifest["hidden_size"]
        for name, entry in manifest["files"].items():
            limit = (
                _certificate_file_limit(entry, vocab, hidden)
                if name == _CERTIFICATES_FILE
                else None
            )
            raw = _read_regular(path / name, size=entry["size_bytes"], limit=limit)
            if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
                raise ValueError(f"S0 cache file SHA256 mismatch: {name}")
            try:
                values = load(raw)
            except Exception as error:
                raise ValueError(f"invalid S0 cache Safetensors member: {name}") from error
            del raw
            if set(values) != set(entry["tensors"]):
                raise ValueError(f"S0 cache tensor keys mismatch: {name}")
            for key, value in values.items():
                _tensor(value, key)
                if _json_bytes(_tensor_spec(value)) != _json_bytes(entry["tensors"][key]):
                    raise ValueError(f"S0 cache tensor metadata/hash mismatch: {key}")
            tensors[name] = values
        if set(tensors["sites.safetensors"]) != {"sites"}:
            raise ValueError("sites.safetensors must contain only the sites table")
        sites = tensors["sites.safetensors"]["sites"]
        _tensor(sites, "sites", dtype=torch.float32)
        if sites.shape != (vocab, hidden) or _tensor_digest(sites) != manifest["sites_sha256"]:
            raise ValueError("S0 cache sites shape or SHA256 mismatch")
        certificates = _certificate_vectors(tensors.get(_CERTIFICATES_FILE), vocab, hidden)
        if _CERTIFICATES_FILE in tensors and certificates is None:
            raise ValueError("certificates.safetensors must contain at least one witness")
        stored_audit = tensors["audit.safetensors"]
        if not stored_audit.keys() >= _DERIVED_FIELDS or "token_ids" not in stored_audit:
            raise ValueError("S0 cache is missing derived audit fields or token_ids")
        audit = _audit_vectors(
            {key: value for key, value in stored_audit.items() if key not in _DERIVED_FIELDS}, vocab
        )
        _check_classification_witnesses(audit, certificates)
        derived, counts = _derived_audit(sites, audit, manifest["solver_config"])
        for key, expected in derived.items():
            actual = stored_audit[key]
            if (
                actual.dtype != expected.dtype
                or actual.shape != expected.shape
                or not torch.equal(actual, expected)
            ):
                raise ValueError(f"S0 cache derived audit mismatch: {key}")
        if _json_bytes(counts) != _json_bytes(manifest["counts"]):
            raise ValueError("S0 cache counts disagree with per-token evidence")
        if manifest["usable_for_training"] != (counts["usable"] == vocab):
            raise ValueError("S0 cache usable_for_training disagrees with per-token evidence")
        self._sites = sites
        self._certificates = certificates

    def validate_binding(
        self,
        *,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        model_id: str,
        model_revision: str,
        solver_config: dict[str, Any],
    ) -> None:
        """Fail closed for stale heads/config/revisions and diagnostic-only tables."""
        config = _config(solver_config)
        expected = {
            "model_id": _text(model_id, "model_id"),
            "model_revision": _text(model_revision, "model_revision"),
            "head_sha256": head_fingerprint(weight, bias),
            "solver_sha256": _json_digest({"contract": _SOLVER_CONTRACT, "config": config}),
        }
        for name, value in expected.items():
            if self._manifest[name] != value:
                raise ValueError(f"S0 cache binding mismatch: {name}")
        self.validate_integrity()
        if not self._manifest["usable_for_training"]:
            raise ValueError(
                "S0 cache is diagnostic-only: numerical acceptance failed; every token must "
                "have finite values, be within the ball, report convergence, and have a "
                "Frank-Wolfe gap <= gap_tolerance"
            )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_no_replace(source: Path, destination: Path) -> None:
    """Linux atomic directory publication, including refusal to replace EMPTY dirs.

    os.rename's check-then-rename pattern can clobber an intervening empty
    destination. Fail closed if the filesystem lacks RENAME_NOREPLACE instead.
    """
    libc = ctypes.CDLL(None, use_errno=True)
    rename = getattr(libc, "renameat2", None)
    if rename is None:
        raise OSError("atomic create-only S0 publication requires Linux renameat2")
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(source), -100, os.fsencode(destination), 1) != 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), str(destination))


def _reuse(destination: Path, expected: dict[str, Any]) -> dict[str, Any]:
    try:
        existing = S0Cache(destination)
        existing.validate_integrity()
        if _json_bytes(existing.manifest) != _json_bytes(expected):
            raise ValueError("publication content differs")
    except (ValueError, OSError) as error:
        raise FileExistsError(
            f"refusing to overwrite existing S0 destination: {destination}"
        ) from error
    return existing.manifest


def write_s0_cache(
    destination: str | Path,
    *,
    sites: torch.Tensor,
    per_token: dict[str, torch.Tensor],
    summary: dict[str, Any],
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    model_id: str,
    model_revision: str,
    solver_config: dict[str, Any],
    certificates: dict[str, torch.Tensor] | None = None,
) -> dict[str, Any]:
    """Create an immutable complete artifact, or reuse an exactly matching one.

    Return the authoritative manifest. ``summary`` is stored separately as
    ``solver_summary`` and never authorizes training. No row filtering, default
    sites, resume shards, pickle, or distributed checkpoint loading is supported.
    ``certificates`` optionally snapshots independent CPU FP64 unit directions
    and their unique int64 head-row IDs, without converting/normalizing them.
    Empty input omits the extra member; classifier version/evidence belongs in
    ``summary``. External witnesses never replace the legacy site-only P1 audit.
    """
    destination = _safe_path(destination)
    vocab, hidden = _head(weight, bias)
    config = _config(solver_config)
    _tensor(sites, "sites", dtype=torch.float32)
    if sites.device.type != "cpu" or sites.shape != (vocab, hidden):
        raise ValueError("sites must be a complete CPU float32 [V, H] table")
    sites = sites.detach().contiguous().clone()
    audit = _audit_vectors(per_token, vocab)
    certificates = _certificate_vectors(certificates, vocab, hidden)
    _check_classification_witnesses(audit, certificates)
    derived, counts = _derived_audit(sites, audit, config)
    _independent_small_audit(sites, audit, weight, bias, config)
    if not isinstance(summary, dict):
        raise ValueError("summary must be a JSON dict")
    manifest = {
        "schema_version": S0_CACHE_SCHEMA_VERSION,
        "anchor_mode": "recorded_s0",
        "format": "safetensors",
        "complete": True,
        "usable_for_training": counts["usable"] == vocab,
        "vocabulary_size": vocab,
        "hidden_size": hidden,
        "row_order": "token_id_0_to_V_minus_1",
        "storage_dtype": "float32",
        "model_id": _text(model_id, "model_id"),
        "model_revision": _text(model_revision, "model_revision"),
        "head_sha256": head_fingerprint(weight, bias),
        "head": {
            "weight_shape": [vocab, hidden],
            "bias_shape": None if bias is None else [vocab],
            "dtype": "float32",
        },
        "solver_contract": copy.deepcopy(_SOLVER_CONTRACT),
        "solver_config": config,
        "solver_config_sha256": _json_digest(config),
        "solver_sha256": _json_digest({"contract": _SOLVER_CONTRACT, "config": config}),
        "sites_sha256": _tensor_digest(sites),
        "audit_provenance": _provenance(vocab, hidden),
        "counts": counts,
        "solver_summary": _json_copy(summary, "summary"),
        "files": {},
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    _safe_path(destination)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.{uuid4().hex}.tmp-", dir=destination.parent)
    )
    try:
        members = {
            "sites.safetensors": {"sites": sites},
            "audit.safetensors": {**audit, **derived},
        }
        if certificates is not None:
            members[_CERTIFICATES_FILE] = certificates
        for name, values in members.items():
            file = temporary / name
            save_file(dict(sorted(values.items())), str(file))
            with file.open("rb") as handle:
                os.fsync(handle.fileno())
                digest = hashlib.sha256()
                while chunk := handle.read(1024**2):
                    digest.update(chunk)
            manifest["files"][name] = {
                "sha256": digest.hexdigest(),
                "size_bytes": file.stat().st_size,
                "tensors": {key: _tensor_spec(value) for key, value in sorted(values.items())},
            }
        manifest["manifest_sha256"] = _json_digest(manifest)
        encoded = _json_bytes(manifest) + b"\n"
        if len(encoded) > _MANIFEST_LIMIT:
            raise ValueError("S0 cache manifest is too large")
        with (temporary / "manifest.json").open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        S0Cache(temporary).validate_integrity()
        _fsync_directory(temporary)
        _safe_path(destination)
        try:
            _publish_no_replace(temporary, destination)
        except FileExistsError:
            return _reuse(destination, manifest)
        _fsync_directory(destination.parent)
        return copy.deepcopy(manifest)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
