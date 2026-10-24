"""Gold-label LVD+ with the legacy outer moment and an opt-in boundary risk.

An absent Anchor denotes a target-only phase on ALL ranks and skips pressure
collectives. An Anchor rank with zero supervised labels must instead provide
its all-masked batch: it still participates in the global moment, including
its graph-connected zero local gradient. No reference model or d0 is used.
Hybrid mode adds a separate global position mean, never a moment of risk rows.
"""

from __future__ import annotations

import copy
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as distributed
from torch import Tensor

from sft_conflict.config import LVDNeighborStrategy, LVDPlusAnchorMode, LVDPlusConfig
from sft_conflict.geometry import (
    LVDPlusPressureResult,
    chunked_lvd_plus_pressure,
    distributed_second_moment_frobenius_loss_and_global_pressure,
    pressure_spectrum_statistics,
    select_lvd_competitors,
)
from sft_conflict.geometry.boundary_risk import distributed_global_mean, winner_risk_rows
from sft_conflict.models import get_unembedding

from .base import LossContext, LossResult
from .plugins import _anchor_loss_result

_BOUNDARY_SUMMARIES = ("signed_distance", "center_margin", "current_margin")
_POSITION_SUMMARIES = ("center_distance_norm", "hidden_norm", "pressure_norm", "pressure_energy")
_SPECTRUM_FIELDS = {
    "global_pressure_positions": "sample_count",
    "uncentered_or_centered_trace": "trace",
    "moment_frobenius_squared": "frobenius_squared",
    "maximum_eigenvalue": "maximum_eigenvalue",
    "effective_rank": "effective_rank",
    "normalized_frobenius": "normalized_frobenius",
    "maximum_eigenvalue_share": "maximum_eigenvalue_share",
}
_LVD_PLUS_RATIO_COUNTS = {
    **{
        f"lvd_plus/response_region_{region}_rate": (
            f"lvd_plus/response_region_{region}_count",
            "lvd_plus/valid_boundary_count",
        )
        for region in range(3)
    },
    **{
        f"lvd_plus/{name}_rate": (f"lvd_plus/{name}_count", "lvd_plus/valid_boundary_count")
        for name in (
            "crossing",
            "ignored_crossing",
            "active_exponent_tail",
            "response_active_boundary",
        )
    },
    "lvd_plus/response_active_position_rate": (
        "lvd_plus/response_active_position_count",
        "lvd_plus/position_count",
    ),
    "lvd_plus/label_accuracy": ("lvd_plus/label_correct_count", "lvd_plus/label_token_count"),
    "lvd_plus/competitors_per_position": (
        "lvd_plus/selected_competitor_count",
        "lvd_plus/position_count",
    ),
    "lvd_plus/degenerate_pair_rate": (
        "lvd_plus/degenerate_pair_count",
        "lvd_plus/selected_competitor_count",
    ),
    **{
        f"lvd_plus/{name}_mean": (f"lvd_plus/{name}_sum", "lvd_plus/valid_boundary_count")
        for name in _BOUNDARY_SUMMARIES
    },
    **{
        f"lvd_plus/{name}_mean": (f"lvd_plus/{name}_sum", "lvd_plus/position_count")
        for name in _POSITION_SUMMARIES
    },
}

# Reporting schema only, independent of the legacy LVD/pilot acceptance maps.
# Crossing includes every margin <= 0, including s <= 0 (ignored crossings).
# Both crossing rates use all valid boundaries, not only active responses.
# Exact duplicate rows remain selected competitors, but are not boundaries.
LVD_PLUS_CALIBRATION_METRIC_COUNTS = {
    "anchor/teacher_forced_accuracy": (
        "anchor/teacher_forced_correct_count",
        "anchor/teacher_forced_token_count",
    ),
    **_LVD_PLUS_RATIO_COUNTS,
}


@dataclass(frozen=True)
class LVDPlusRawComponents:
    """Unweighted, already globally normalized calibration/training components.

    ``outer_loss`` is the original pressure second moment, including zero
    pressure rows. ``risk_loss`` is the global supervised-position mean in
    hybrid mode (a connected zero in pressure-only mode). Neither contains
    lambda, risk_weight, S_O, or S_R. Counts/diagnostics are detached; only
    counts belong to SUM aggregation. Reading raw components does not consume
    the context's once-only Anchor accuracy metrics or require scale binding.
    """

    outer_loss: Tensor
    risk_loss: Tensor
    global_position_count: int
    local_count: int
    anchor_present: bool
    connected_logits_zero: Tensor
    counts: dict[str, Tensor | float] = field(default_factory=dict)
    diagnostics: dict[str, Tensor | float] = field(default_factory=dict)


def _finite_hybrid_number(name: str, value: float, *, allow_zero: bool = False) -> float:
    try:
        valid = (
            isinstance(value, int | float)
            and not isinstance(value, bool)
            and math.isfinite(value)
            and (value >= 0 if allow_zero else value > 0)
        )
    except OverflowError:
        valid = False
    if not valid:
        bound = "non-negative" if allow_zero else "positive"
        raise ValueError(f"lvd_plus.{name} must be a {bound} finite number")
    return float(value)


def _divide_by_scale(value: Tensor, scale: float) -> Tensor:
    """Do not round an otherwise valid Python scale to FP32 infinity or zero."""
    while scale > 1e30:
        value = value / 1e30
        scale /= 1e30
    while scale < 1e-30:
        value = value / 1e-30
        scale /= 1e-30
    return value / scale


def _finite_lambda(value: float) -> float:
    try:
        valid = (
            isinstance(value, int | float)
            and not isinstance(value, bool)
            and math.isfinite(value)
            and value >= 0
        )
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError("resolved LVD+ lambda must be a finite non-negative number")
    return float(value)


@torch.no_grad()
def _local_diagnostics(
    result: LVDPlusPressureResult,
    *,
    competitor_mask: Tensor,
    current_hidden: Tensor,
    current_logits: Tensor,
    label_ids: Tensor,
) -> tuple[dict[str, Tensor | float], dict[str, Tensor | float]]:
    """Detached local sums; global spectra are already global and never SUMmed."""
    boundary_mask = competitor_mask
    degenerate_count: Tensor | float = 0.0
    if result.degenerate_pair_mask is not None:
        degenerate_pairs = result.degenerate_pair_mask & competitor_mask
        boundary_mask = competitor_mask & ~degenerate_pairs
        degenerate_count = degenerate_pairs.sum()
    signed = result.signed_distances.masked_select(boundary_mask)
    margins = result.current_margins.masked_select(boundary_mask)
    regions = result.response.regions.masked_select(boundary_mask)
    active = result.signed_distances.gt(0) & boundary_mask
    pressure_norms = result.pressure_vectors.detach().double().norm(dim=-1)
    counts: dict[str, Tensor | float] = {
        "lvd_plus/position_count": float(label_ids.numel()),
        "lvd_plus/label_token_count": float(label_ids.numel()),
        "lvd_plus/label_correct_count": current_logits.argmax(dim=-1).eq(label_ids).sum(),
        "lvd_plus/selected_competitor_count": competitor_mask.sum(),
        "lvd_plus/degenerate_pair_count": degenerate_count,
        "lvd_plus/valid_boundary_count": boundary_mask.sum(),
        "lvd_plus/crossing_count": margins.le(0).sum(),
        "lvd_plus/ignored_crossing_count": (signed.le(0) & margins.le(0)).sum(),
        "lvd_plus/response_active_boundary_count": active.sum(),
        "lvd_plus/response_active_position_count": active.any(dim=-1).sum(),
        # Geometry counts only the active tangent tail, strictly beyond C/mu.
        "lvd_plus/active_exponent_tail_count": result.response.exponent_clamp_count,
    }
    counts.update(
        {
            f"lvd_plus/response_region_{region}_count": regions.eq(region).sum()
            for region in range(3)
        }
    )
    diagnostics: dict[str, Tensor | float] = {
        "lvd_plus/maximum_unclamped_exponent": result.response.maximum_unclamped_exponent,
    }
    for name, values in (
        ("signed_distance", signed),
        ("center_margin", result.center_margins.masked_select(boundary_mask)),
        ("current_margin", margins),
    ):
        counts[f"lvd_plus/{name}_sum"] = values.double().sum()
        # Like the existing losses, quantiles remain rank/microbatch summaries;
        # only count-aware means and rates are pooled over unequal observations.
        for suffix, quantile in (("q05", 0.05), ("q50", 0.5), ("q95", 0.95)):
            diagnostics[f"lvd_plus/{name}_{suffix}"] = (
                torch.quantile(values.float(), quantile) if values.numel() else 0.0
            )
    for name, values in (
        ("center_distance_norm", result.center_distance_norms),
        ("hidden_norm", current_hidden.detach().double().norm(dim=-1)),
        ("pressure_norm", pressure_norms),
        ("pressure_energy", pressure_norms.square()),
    ):
        counts[f"lvd_plus/{name}_sum"] = values.double().sum()
    return counts, diagnostics


@torch.no_grad()
def _loss_result(
    context: LossContext,
    *,
    loss: Tensor,
    unscaled_loss: Tensor,
    effective_lambda: float,
    counts: dict[str, Tensor | float] | None = None,
    diagnostics: dict[str, Tensor | float] | None = None,
) -> LossResult:
    sums: dict[str, Tensor | float] = {
        name: 0.0 for pair in _LVD_PLUS_RATIO_COUNTS.values() for name in pair
    }
    sums.update(counts or {})
    numerators = {name: sums[pair[0]] for name, pair in _LVD_PLUS_RATIO_COUNTS.items()}
    denominators = {name: sums[pair[1]] for name, pair in _LVD_PLUS_RATIO_COUNTS.items()}
    metrics: dict[str, Tensor | float] = {
        "loss/lvd_plus": unscaled_loss.detach(),
        "loss/lvd_plus_weighted": loss.detach(),
        "lvd_plus/lambda": effective_lambda,
        "lvd_plus/response_region_count": 3.0,  # Metadata, NOT an observation counter.
        "lvd_plus/maximum_unclamped_exponent": 0.0,
        **{f"lvd_plus/{name}": 0.0 for name in _SPECTRUM_FIELDS},
        **{
            f"lvd_plus/{name}_{suffix}": 0.0
            for name in _BOUNDARY_SUMMARIES
            for suffix in ("q05", "q50", "q95")
        },
        **(diagnostics or {}),
        **sums,
    }
    for name, numerator in numerators.items():
        denominator = denominators[name]
        metrics[name] = numerator / (
            denominator.clamp_min(1) if isinstance(denominator, Tensor) else max(denominator, 1.0)
        )
    return _anchor_loss_result(
        context,
        loss=loss,
        metrics=metrics,
        ratio_numerators=numerators,
        ratio_denominators=denominators,
        sum_metrics=sums,
    )


def _validated_s0_identity(
    config: LVDPlusConfig, identity: dict[str, Any] | None
) -> dict[str, Any]:
    """Check provenance structure; actual artifact/head binding belongs to S0Cache.

    A dictionary alone is not evidence of a valid cache. Direct callers, like
    the trainer, must first run S0Cache.validate_binding with their actual head.
    No head/table hash or whole-table copy is repeated in the loss hot path.
    """
    if not isinstance(identity, dict) or identity.get("anchor_mode") != "recorded_s0":
        raise ValueError("recorded_s0 requires a verified S0 cache identity")
    if type(identity.get("schema_version")) is not int or identity["schema_version"] != 1:
        raise ValueError("S0 identity schema_version must be 1")
    for name in ("head_sha256", "solver_sha256", "sites_sha256", "manifest_sha256"):
        if (
            not isinstance(identity.get(name), str)
            or re.fullmatch(r"[0-9a-f]{64}", identity[name]) is None
        ):
            raise ValueError(f"S0 identity {name} must be a lowercase SHA256 digest")
    if config.s0_cache_sha256 is not None and config.s0_cache_sha256 != identity["manifest_sha256"]:
        raise ValueError("S0 identity does not match lvd_plus.s0_cache_sha256")
    try:
        json.dumps(identity, allow_nan=False)
    except (TypeError, ValueError, OverflowError, RecursionError) as error:
        raise ValueError("S0 identity must contain finite JSON provenance") from error
    return copy.deepcopy(identity)


def _validated_scale_identity(identity: dict[str, Any]) -> dict[str, Any]:
    """Validate metadata structure, not artifact provenance (owned by the workflow)."""
    if not isinstance(identity, dict) or identity.get("kind") != "lvd_plus_scales":
        raise ValueError("scale identity kind must be lvd_plus_scales")
    if type(identity.get("schema_version")) is not int or identity["schema_version"] != 1:
        raise ValueError("scale identity schema_version must be 1")
    for name in ("manifest_sha256", "binding_sha256"):
        if (
            not isinstance(identity.get(name), str)
            or re.fullmatch(r"[0-9a-f]{64}", identity[name]) is None
        ):
            raise ValueError(f"scale identity {name} must be a lowercase SHA256 digest")
    try:
        json.dumps(identity, allow_nan=False)
    except (TypeError, ValueError, OverflowError, RecursionError) as error:
        raise ValueError("scale identity must contain finite JSON provenance") from error
    return copy.deepcopy(identity)


@torch.no_grad()
def _hybrid_metrics(
    raw: LVDPlusRawComponents,
    outer_normalized: Tensor,
    risk_normalized: Tensor,
    risk_weighted_normalized: Tensor,
) -> dict[str, Tensor | float]:
    # All five are ordinary global scalars, NOT entries in sum_metrics.
    return {
        "loss/lvd_plus_outer_raw": raw.outer_loss.detach(),
        "loss/lvd_plus_risk_raw": raw.risk_loss.detach(),
        "loss/lvd_plus_outer_normalized": outer_normalized.detach(),
        "loss/lvd_plus_risk_normalized": risk_normalized.detach(),
        "loss/lvd_plus_risk_weighted_normalized": risk_weighted_normalized.detach(),
    }


class AnchorLVDPlusLoss:
    name = "anchor_lvd_plus"
    requires_anchor_hidden = True

    def __init__(
        self,
        config: LVDPlusConfig,
        *,
        lambda_value: float,
        unembedding_weight: Tensor | None = None,
        unembedding_bias: Tensor | None = None,
        process_group: distributed.ProcessGroup | None = None,
        anchor_centers: Tensor | None = None,
        s0_identity: dict[str, Any] | None = None,
    ) -> None:
        if not isinstance(config, LVDPlusConfig):
            raise ValueError("AnchorLVDPlusLoss requires LVDPlusConfig, not legacy LVDConfig")
        config.validate()
        if unembedding_bias is not None and unembedding_weight is None:
            raise ValueError("LVD+ unembedding_bias requires an explicit unembedding_weight")
        for name, value in (("unembedding_weight", unembedding_weight), ("bias", unembedding_bias)):
            if value is not None and (not isinstance(value, Tensor) or value.requires_grad):
                raise ValueError(f"LVD+ {name} must be a frozen tensor")
        if config.anchor_mode is LVDPlusAnchorMode.HALF_UNEMBEDDING:
            if anchor_centers is not None or s0_identity is not None:
                raise ValueError("half_unembedding forbids anchor_centers and s0_identity")
        else:
            if (
                unembedding_weight is None
                or unembedding_weight.device.type != "cpu"
                or unembedding_weight.ndim != 2
                or not unembedding_weight.is_floating_point()
                or (unembedding_bias is not None and unembedding_bias.device.type != "cpu")
            ):
                raise ValueError("recorded_s0 requires an explicit frozen CPU Unembedding snapshot")
            if (
                not isinstance(anchor_centers, Tensor)
                or anchor_centers.layout != torch.strided
                or anchor_centers.shape != unembedding_weight.shape
                or anchor_centers.dtype != torch.float32
                or anchor_centers.device.type != "cpu"
                or anchor_centers.requires_grad
            ):
                raise ValueError("recorded_s0 requires frozen CPU FP32 anchor_centers shaped [V,H]")
            if not bool(torch.isfinite(anchor_centers).all()):
                raise ValueError("recorded_s0 anchor_centers must be finite")
            s0_identity = _validated_s0_identity(config, s0_identity)
        self.config = config
        self.set_lambda(lambda_value)
        # The trainer owns the immutable CPU snapshot. Do not copy or transfer
        # the whole head here; geometry transfers indexed rows per chunk only.
        self.unembedding_weight = unembedding_weight
        self.unembedding_bias = unembedding_bias
        self.process_group = process_group
        self.anchor_centers = anchor_centers
        self.s0_identity = s0_identity
        # Resolved scalars / a saved SHA alone are not a verified workflow
        # binding. Even restored configurations must explicitly bind identity.
        self._bound_scales: tuple[float, float] | None = None
        self._scale_identity: dict[str, Any] | None = None

    @property
    def scale_identity(self) -> dict[str, Any] | None:
        """A defensive metadata copy; this property performs no cache I/O."""
        return copy.deepcopy(self._scale_identity)

    def bind_scales(self, outer_scale: float, risk_scale: float, identity: dict[str, Any]) -> None:
        """Bind nontrainable scales after the workflow verifies the actual cache.

        Synthetic well-formed identities are useful for direct unit callers,
        but are not artifact verification. Rebinding is idempotent only for
        the same scales and identity; stale config values are never replaced.
        """
        if not self.config.uses_boundary_risk:
            raise ValueError("bind_scales requires pressure_plus_risk")
        outer = _finite_hybrid_number("resolved_outer_scale", outer_scale)
        risk = _finite_hybrid_number("resolved_risk_scale", risk_scale)
        verified = _validated_scale_identity(identity)
        for name, value in (("resolved_outer_scale", outer), ("resolved_risk_scale", risk)):
            configured = getattr(self.config, name)
            if configured is not None and _finite_hybrid_number(name, configured) != value:
                raise ValueError(f"bound scale does not match lvd_plus.{name}")
        configured_sha = self.config.scale_cache_sha256
        if configured_sha is not None and (
            not isinstance(configured_sha, str) or configured_sha != verified["manifest_sha256"]
        ):
            raise ValueError("scale identity does not match lvd_plus.scale_cache_sha256")
        if self._bound_scales is not None and self._bound_scales != (outer, risk):
            raise ValueError("LVD+ scales are already bound to different values")
        if self._scale_identity is not None and self._scale_identity != verified:
            raise ValueError("LVD+ scales are already bound to a different identity")
        self.config.resolved_outer_scale = outer
        self.config.resolved_risk_scale = risk
        self.config.scale_cache_sha256 = verified["manifest_sha256"]
        self._bound_scales = outer, risk
        self._scale_identity = verified

    def _scales_for_forward(self) -> tuple[float, float]:
        if self._bound_scales is None or self._scale_identity is None:
            raise ValueError("pressure_plus_risk requires bind_scales before an Anchor loss")
        for name, bound in zip(
            ("resolved_outer_scale", "resolved_risk_scale"), self._bound_scales, strict=True
        ):
            if _finite_hybrid_number(name, getattr(self.config, name)) != bound:
                raise ValueError(f"lvd_plus.{name} changed after bind_scales")
        if (
            not isinstance(self.config.scale_cache_sha256, str)
            or self.config.scale_cache_sha256 != self._scale_identity["manifest_sha256"]
        ):
            raise ValueError("lvd_plus.scale_cache_sha256 changed after bind_scales")
        return self._bound_scales

    def set_lambda(self, value: float) -> None:
        self.lambda_value = _finite_lambda(value)

    def _effective_lambda(self, global_step: int) -> float:
        value = _finite_lambda(self.lambda_value)
        if type(global_step) is not int or global_step < 0:
            raise ValueError("LVD+ global_step must be a non-negative integer")
        warmup = self.config.lambda_warmup_steps
        if type(warmup) is not int or warmup < 0:
            raise ValueError("LVD+ lambda_warmup_steps must be a non-negative integer")
        if warmup and global_step + 1 < warmup:
            value *= (global_step + 1) / warmup
        return _finite_lambda(value)

    def raw_components(self, context: LossContext) -> LVDPlusRawComponents:
        """Compute raw global O and R without scales, lambda, or metric side effects.

        This is also the no-grad scale-recording API. An all-masked Anchor is
        still present and participates in both hybrid collectives; only a
        rank-consistent absent Anchor skips them. Use a fresh LossContext for
        gradient calibration/training after a cached no-grad recording forward.
        """
        if context.reference_model is not None:
            raise ValueError("Anchor LVD+ does not accept a reference model")
        if context.stream_batch("anchor") is None:
            # Rank-consistent target-only evaluation: no geometry/head access
            # or collective. All-masked Anchor batches take the normal path.
            zero = context.connected_zero()
            return LVDPlusRawComponents(
                outer_loss=zero,
                risk_loss=zero,
                global_position_count=0,
                local_count=0,
                anchor_present=False,
                connected_logits_zero=zero,
            )
        # Capture settings BEFORE the model forward. Checkpoint reevaluation
        # must not observe subsequent mutations of the caller's config.
        config = copy.copy(self.config)
        uses_boundary_risk = config.uses_boundary_risk
        risk_temperature = 1.0
        if uses_boundary_risk:
            if config.post_crossing_slope_gain != 1.0:
                raise ValueError("pressure_plus_risk requires post_crossing_slope_gain=1")
            risk_temperature = _finite_hybrid_number("risk_temperature", config.risk_temperature)
        if not context.capture_anchor_hidden or not context.capture_anchor_label_positions:
            raise ValueError(
                "Anchor LVD+ requires capture_anchor_hidden=True and "
                "capture_anchor_label_positions=True before the first forward"
            )
        source_weight, source_bias = self.unembedding_weight, self.unembedding_bias
        if source_weight is None:
            head = get_unembedding(context.model)
            if any(parameter.requires_grad for parameter in head.parameters()):
                raise ValueError("Anchor LVD+ requires a frozen Unembedding")
            source_weight = head.weight.detach()
            source_bias = getattr(head, "bias", None)
        outputs = context.forward_current("anchor")
        hidden = context.hidden_states.get("anchor")
        positions, mask, gold_ids = context.anchor_label_selection()
        if hidden is None or hidden.ndim != 3 or hidden.shape[:2] != positions.shape:
            raise RuntimeError("LVD+ loss did not receive gold-position Anchor hidden states")

        with torch.autocast(device_type=hidden.device.type, enabled=False):
            positions = positions.to(hidden.device)
            mask = mask.to(hidden.device)
            label_ids = gold_ids.to(hidden.device)[mask]
            batch_indices, _ = mask.nonzero(as_tuple=True)
            current_hidden = hidden[mask].float()
            current_logits = outputs.logits.detach()[batch_indices, positions[mask]].float()
            if uses_boundary_risk and label_ids.numel() == 0:
                # There is no winner to select on an empty rank, even in a
                # one-token toy vocabulary. Both global losses still run.
                columns = min(config.top_k, max(current_logits.shape[1] - 1, 0))
                competitor_ids = label_ids.new_empty((0, columns))
                competitor_mask = torch.ones_like(competitor_ids, dtype=torch.bool)
            else:
                competitor_ids, competitor_mask = select_lvd_competitors(
                    current_logits=current_logits,
                    reference_token_ids=label_ids,  # The selector excludes GOLD, not Base.
                    top_k=config.top_k,
                    strategy=LVDNeighborStrategy.CURRENT_TOPK,
                )
            center_kwargs = (
                {"anchor_centers": self.anchor_centers}
                if config.anchor_mode is LVDPlusAnchorMode.RECORDED_S0
                else {}
            )
            result = chunked_lvd_plus_pressure(
                current_hidden=current_hidden,
                unembedding_weight=source_weight,
                unembedding_bias=source_bias,
                label_token_ids=label_ids,
                competitor_ids=competitor_ids,
                competitor_mask=competitor_mask,
                config=config,
                **center_kwargs,
            )
            # ONE unchanged outer moment over all pressure rows, including
            # zero responses. Never a mean of directional potentials/chunks.
            outer_loss, global_pressure = (
                distributed_second_moment_frobenius_loss_and_global_pressure(
                    result.pressure_vectors,
                    mode=config.moment_mode,
                    compensate_distributed_gradient_average=(
                        context.compensate_distributed_gradient_average
                    ),
                    process_group=self.process_group,
                )
            )
            connected_logits_zero = outputs.logits.sum() * 0.0
            risk_loss = connected_logits_zero
            if uses_boundary_risk:
                if label_ids.numel():
                    if competitor_ids.shape[1] == 0 or not bool(competitor_mask.any(dim=1).all()):
                        raise ValueError("winner risk requires a valid non-gold competitor per row")
                    # CURRENT_TOPK with positive K includes a full-vocabulary
                    # non-gold maximizer. Do not assume the IDs are sorted;
                    # masked padding IDs must never participate in gather/max.
                    safe_ids = competitor_ids.masked_fill(~competitor_mask, 0)
                    selected_logits = current_logits.gather(1, safe_ids).masked_fill(
                        ~competitor_mask, -torch.inf
                    )
                    winner_columns = selected_logits.argmax(dim=1, keepdim=True)
                    winner_ids = safe_ids.gather(1, winner_columns).squeeze(1)
                else:
                    winner_ids = label_ids.new_empty((0,))
                risk_error = None
                try:
                    risk_rows = winner_risk_rows(
                        current_hidden=current_hidden,
                        unembedding_weight=source_weight,
                        label_token_ids=label_ids,
                        winner_token_ids=winner_ids,
                        unembedding_bias=source_bias,
                        temperature=risk_temperature,
                        position_chunk_size=config.position_chunk_size,
                    )
                except ValueError as exc:
                    risk_error = str(exc)
                    risk_rows = current_hidden.sum(dim=1) * 0.0
                # Synchronize recoverable numeric/input failures BEFORE the
                # mean's collectives. Do not catch fatal CUDA RuntimeErrors.
                if distributed.is_available() and distributed.is_initialized():
                    failed = torch.tensor(
                        int(risk_error is not None), device=current_hidden.device
                    )
                    distributed.all_reduce(
                        failed, op=distributed.ReduceOp.MAX, group=self.process_group
                    )
                    if bool(failed):
                        raise ValueError(risk_error or "winner risk failed on another rank")
                elif risk_error is not None:
                    raise ValueError(risk_error)
                risk_loss = distributed_global_mean(
                    risk_rows,
                    compensate_distributed_gradient_average=(
                        context.compensate_distributed_gradient_average
                    ),
                    process_group=self.process_group,
                )
            spectrum = pressure_spectrum_statistics(
                global_pressure, mode=config.moment_mode, rows_are_global=True
            )
            counts, diagnostics = _local_diagnostics(
                result,
                competitor_mask=competitor_mask,
                current_hidden=current_hidden,
                current_logits=current_logits,
                label_ids=label_ids,
            )
            diagnostics.update(
                {
                    f"lvd_plus/{name}": float(getattr(spectrum, field))
                    for name, field in _SPECTRUM_FIELDS.items()
                }
            )
        return LVDPlusRawComponents(
            outer_loss=outer_loss,
            risk_loss=risk_loss,
            global_position_count=int(global_pressure.shape[0]),
            local_count=int(label_ids.numel()),
            anchor_present=True,
            connected_logits_zero=connected_logits_zero,
            counts=counts,
            diagnostics=diagnostics,
        )

    def __call__(self, context: LossContext) -> LossResult:
        if context.reference_model is not None:
            raise ValueError("Anchor LVD+ does not accept a reference model")
        effective_lambda = self._effective_lambda(context.global_step)
        uses_boundary_risk = self.config.uses_boundary_risk
        anchor_present = context.stream_batch("anchor") is not None
        # Snapshot nontrainable scalars before any model/geometry computation.
        # Config mutations after this point cannot alter this graph's backward.
        outer_scale, risk_scale, risk_weight = 1.0, 1.0, 0.0
        if uses_boundary_risk and anchor_present:
            outer_scale, risk_scale = self._scales_for_forward()
            risk_weight = _finite_hybrid_number(
                "risk_weight", self.config.risk_weight, allow_zero=True
            )
        raw = self.raw_components(context)
        diagnostics = dict(raw.diagnostics)
        if not raw.anchor_present:
            zero = raw.connected_logits_zero
            if uses_boundary_risk:
                diagnostics.update(_hybrid_metrics(raw, zero, zero, zero))
            return _loss_result(
                context,
                loss=zero,
                unscaled_loss=zero,
                effective_lambda=effective_lambda,
                diagnostics=diagnostics,
            )
        with torch.autocast(device_type=raw.outer_loss.device.type, enabled=False):
            unscaled_loss = raw.outer_loss
            if uses_boundary_risk:
                outer_normalized = _divide_by_scale(raw.outer_loss, outer_scale)
                risk_normalized = _divide_by_scale(raw.risk_loss, risk_scale)
                risk_weighted_normalized = risk_weight * risk_normalized
                unscaled_loss = outer_normalized + risk_weighted_normalized
                diagnostics.update(
                    _hybrid_metrics(
                        raw, outer_normalized, risk_normalized, risk_weighted_normalized
                    )
                )
            # Preserve the original pressure-only arithmetic, including its
            # logits connection. Lambda multiplies the hybrid sum exactly once.
            weighted_loss = effective_lambda * unscaled_loss + raw.connected_logits_zero
        return _loss_result(
            context,
            loss=weighted_loss,
            unscaled_loss=unscaled_loss,
            effective_lambda=effective_lambda,
            counts=raw.counts,
            diagnostics=diagnostics,
        )
