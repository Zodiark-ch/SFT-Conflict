"""M1 target, anchor, distillation, and null-LVD loss plugins."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as functional

from sft_conflict.config import (
    AnchorGeometryConfig,
    LVDConfig,
    LVDPlusConfig,
    LVDPressureMode,
    LVDResponseMode,
    MethodConfig,
    MethodName,
)
from sft_conflict.data import ReferenceCache
from sft_conflict.geometry import (
    LVDPressureResult,
    chunked_lvd_pressure_loss,
    distributed_second_moment_frobenius_loss_and_global_pressure,
    pressure_spectrum_statistics,
    select_lvd_competitors,
)
from sft_conflict.models import get_unembedding

from .base import (
    LossContext,
    LossPlugin,
    LossResult,
    causal_lm_accuracy_counts,
    causal_lm_cross_entropy,
    stsb_numeric_mse,
)


@torch.no_grad()
def _anchor_loss_result(
    context: LossContext,
    *,
    loss: torch.Tensor,
    metrics: dict[str, torch.Tensor | float],
    ratio_numerators: dict[str, torch.Tensor | float] | None = None,
    ratio_denominators: dict[str, torch.Tensor | float] | None = None,
    sum_metrics: dict[str, torch.Tensor | float] | None = None,
) -> LossResult:
    numerators = dict(ratio_numerators or {})
    denominators = dict(ratio_denominators or {})
    sums = dict(sum_metrics or {})
    metrics = dict(metrics)
    if not context.anchor_accuracy_recorded:
        batch = context.stream_batch("anchor")
        if batch is None:
            correct, tokens, accuracy = 0.0, 0.0, 0.0
        else:
            # Every caller has already forwarded Anchor. Never request another
            # current/reference forward, even for LVD's empty-cache branch.
            correct, tokens = causal_lm_accuracy_counts(
                context.current_outputs["anchor"].logits, batch["labels"]
            )
            accuracy = correct / tokens.clamp_min(1)
        name = "anchor/teacher_forced_accuracy"
        numerators[name] = correct
        denominators[name] = tokens
        sums.update(
            {
                "anchor/teacher_forced_correct_count": correct,
                "anchor/teacher_forced_token_count": tokens,
                "tokens/anchor_supervised": tokens,
            }
        )
        metrics.update(sums)
        metrics[name] = accuracy
        context.anchor_accuracy_recorded = True
    elif context.capture_anchor_label_positions:
        # LVD+ may run before Anchor CE in a manually composed plugin list.
        # Its first observation already SUMmed this count; do not reintroduce
        # the later CE scalar as a conflicting mean. Legacy outputs stay as-is.
        metrics.pop("tokens/anchor_supervised", None)
    return LossResult(loss, metrics, numerators, denominators, sums)


_LVD_RATIO_COUNTS = {
    **{
        f"lvd/response_region_{region}_rate": (
            f"lvd/response_region_{region}_count",
            "lvd/valid_boundary_count",
        )
        for region in range(4)
    },
    "lvd/safety_core_violation_rate": (
        "lvd/safety_core_violation_count",
        "lvd/valid_boundary_count",
    ),
    "lvd/crossing_rate": ("lvd/crossing_count", "lvd/valid_boundary_count"),
    "lvd/tiny_baseline_margin_rate": (
        "lvd/tiny_baseline_margin_count",
        "lvd/valid_boundary_count",
    ),
    "lvd/active_exponent_cap_rate": (
        "lvd/active_exponent_cap_count",
        "lvd/valid_boundary_count",
    ),
    "lvd/zero_response_gradient_crossing_rate": (
        "lvd/zero_response_gradient_crossing_count",
        "lvd/crossing_count",
    ),
    "lvd/capacity_effective_neighbors_mean": (
        "lvd/capacity_effective_neighbors_sum",
        "lvd/capacity_position_count",
    ),
    "lvd/capacity_max_mean": ("lvd/capacity_max_sum", "lvd/capacity_position_count"),
    **{
        f"lvd/{name}_mean": (f"lvd/{name}_sum", "anchor/cached_position_count")
        for name in ("hidden_norm", "reference_hidden_norm", "hidden_to_reference_norm_ratio")
    },
    "lvd/competitors_per_position": ("lvd/valid_boundary_count", "anchor/cached_position_count"),
    "lvd/actual_flip_rate": ("lvd/actual_flip_count", "anchor/cached_position_count"),
    "lvd/neighbor_recall_on_actual_flips": ("lvd/recalled_flip_count", "lvd/actual_flip_count"),
    "anchor/base_top1_agreement_cached": (
        "anchor/base_top1_agreement_cached_count",
        "anchor/cached_position_count",
    ),
    "anchor/base_correct_retention": (
        "anchor/base_correct_retained_count",
        "anchor/base_correct_cached_count",
    ),
    "anchor/label_accuracy_cached": (
        "anchor/cached_label_correct_count",
        "anchor/cached_label_token_count",
    ),
}


_LVD_RAW_PRESSURE_RATIO_COUNTS = {
    f"lvd/raw_pressure_{name}_mean": (
        f"lvd/raw_pressure_{name}_sum",
        "lvd/raw_pressure_position_count",
    )
    for name in ("norm", "energy")
}


@torch.no_grad()
def _lvd_raw_pressure_counts(raw_pressure: torch.Tensor | None) -> dict[str, torch.Tensor | float]:
    """Local raw-P sums, reduced by the existing log-time count accumulator only."""
    if raw_pressure is None:
        return {
            "lvd/raw_pressure_norm_sum": 0.0,
            "lvd/raw_pressure_energy_sum": 0.0,
            "lvd/raw_pressure_position_count": 0.0,
        }
    norms = raw_pressure.detach().double().norm(dim=-1)
    return {
        "lvd/raw_pressure_norm_sum": norms.sum(),
        "lvd/raw_pressure_energy_sum": norms.square().sum(),
        "lvd/raw_pressure_position_count": float(len(norms)),
    }


# Calibration uses the same count schema, including full-response accuracy.
# This is a reporting map, not a pilot acceptance policy or an objective change.
LVD_CALIBRATION_METRIC_COUNTS = {
    "anchor/teacher_forced_accuracy": (
        "anchor/teacher_forced_correct_count",
        "anchor/teacher_forced_token_count",
    ),
    **_LVD_RATIO_COUNTS,
}


def lvd_ratio_counts(mode: LVDResponseMode | str) -> dict[str, tuple[str, str]]:
    """Copy the legacy map, replacing only its response-region entries."""
    mode = LVDResponseMode(mode)
    return {
        **{
            f"lvd/response_region_{region}_rate": (
                f"lvd/response_region_{region}_count",
                "lvd/valid_boundary_count",
            )
            for region in range(mode.region_count)
        },
        **{
            name: pair
            for name, pair in _LVD_RATIO_COUNTS.items()
            if not name.startswith("lvd/response_region_")
        },
    }


def lvd_calibration_metric_counts(mode: LVDResponseMode | str) -> dict[str, tuple[str, str]]:
    """Mode-aware reporting schema; the exported legacy constant stays unchanged."""
    return {
        "anchor/teacher_forced_accuracy": LVD_CALIBRATION_METRIC_COUNTS[
            "anchor/teacher_forced_accuracy"
        ],
        **lvd_ratio_counts(mode),
    }


@torch.no_grad()
def _lvd_health_counts(
    result: LVDPressureResult,
    *,
    config: LVDConfig,
    competitor_mask: torch.Tensor,
    current_hidden: torch.Tensor,
    reference_hidden: torch.Tensor,
) -> dict[str, torch.Tensor | float]:
    """Local, detached sums only; no forwards, collectives or loss modifications.

    Cap exceedances use only the selected exponential branch, strictly above
    the cap (the derivative at equality is nonzero). The zero-gradient counter
    means legacy FOUR_SEGMENT saturation among actual d_new <= 0 crossings,
    NOT zero gradients of the full LVD objective or all zero response regions.
    A linear tail can exceed the cap without losing its response derivative.
    """
    ratios = result.erosion_ratios.detach().float()
    mode = LVDResponseMode(config.response_mode)
    active_capped = torch.zeros_like(competitor_mask)
    if mode is LVDResponseMode.THREE_SEGMENT:
        boundary = 1.0 / config.gamma0
        exponent = config.mu3 * (ratios - boundary)
        active_capped = competitor_mask & ratios.ge(boundary) & exponent.gt(config.exponent_clamp)
    elif mode in {LVDResponseMode.FOUR_SEGMENT, LVDResponseMode.FOUR_SEGMENT_LINEAR_TAIL}:
        boundary = 1.0 / config.gamma0
        exponent3 = config.mu3 * (ratios - 1.0)
        exponent4 = config.mu4 * (ratios - boundary)
        active_capped = competitor_mask & (
            (ratios.ge(1) & ratios.lt(boundary) & exponent3.gt(config.exponent_clamp))
            | (ratios.ge(boundary) & exponent4.gt(config.exponent_clamp))
        )
    saturated_crossings = active_capped & result.current_margins.detach().le(0)
    if mode is not LVDResponseMode.FOUR_SEGMENT:
        saturated_crossings = torch.zeros_like(competitor_mask)

    hidden_norms = current_hidden.detach().double().norm(dim=-1)
    reference_norms = reference_hidden.detach().double().norm(dim=-1)
    counts: dict[str, torch.Tensor | float] = {
        "lvd/active_exponent_cap_count": active_capped.sum(),
        "lvd/zero_response_gradient_crossing_count": saturated_crossings.sum(),
        "lvd/hidden_norm_sum": hidden_norms.sum(),
        "lvd/reference_hidden_norm_sum": reference_norms.sum(),
        "lvd/hidden_to_reference_norm_ratio_sum": (
            hidden_norms / reference_norms.clamp_min(config.normal_epsilon)
        ).sum(),
        "lvd/capacity_effective_neighbors_sum": 0.0,
        "lvd/capacity_max_sum": 0.0,
        "lvd/capacity_position_count": 0.0,
    }
    # Real LVDPressureResult always has weights. Older lightweight test doubles
    # may omit them: report no observations, never invent uniform alpha/effK=1.
    weights = getattr(result, "capacity_weights", None)
    if weights is not None:
        weights = weights.detach().double().masked_fill(~competitor_mask, 0.0)
        effective_neighbors = weights.square().sum(dim=-1).reciprocal()
        counts.update(
            {
                "lvd/capacity_effective_neighbors_sum": effective_neighbors.sum(),
                "lvd/capacity_max_sum": weights.amax(dim=-1).sum(),
                "lvd/capacity_position_count": float(weights.shape[0]),
            }
        )
    if LVDPressureMode(config.pressure_mode) is LVDPressureMode.ASINH_WEIGHTED_SUM:
        # Never substitute effective pressure for unavailable raw P in old fakes.
        counts.update(_lvd_raw_pressure_counts(getattr(result, "raw_pressure_vectors", None)))
    return counts


@torch.no_grad()
def _lvd_loss_result(
    context: LossContext,
    *,
    loss: torch.Tensor,
    metrics: dict[str, torch.Tensor | float],
    counts: dict[str, torch.Tensor | float] | None = None,
    response_mode: LVDResponseMode | str = LVDResponseMode.FOUR_SEGMENT,
) -> LossResult:
    # Give absent Anchor and empty-cache ranks the same zero-denominator schema.
    # Only local counts belong here; already-global pressure statistics stay means.
    response_mode = LVDResponseMode(response_mode)
    ratio_counts = lvd_ratio_counts(response_mode)
    if counts is not None and "lvd/raw_pressure_position_count" in counts:
        ratio_counts = {**ratio_counts, **_LVD_RAW_PRESSURE_RATIO_COUNTS}
    sums: dict[str, torch.Tensor | float] = {
        name: 0.0 for pair in ratio_counts.values() for name in pair
    }
    for name in (
        "lvd/requested_positions",
        "lvd/cached_positions",
        "lvd/uncached_positions",
        "lvd/exponent_clamp_count",
    ):
        sums[name] = metrics.get(name, 0.0)
    sums["anchor/cached_position_count"] = sums["lvd/cached_positions"]
    sums.update(counts or {})
    numerators = {name: sums[pair[0]] for name, pair in ratio_counts.items()}
    denominators = {name: sums[pair[1]] for name, pair in ratio_counts.items()}
    metrics = {**metrics, **sums}
    if response_mode is LVDResponseMode.THREE_SEGMENT:
        # Mode metadata is a scalar, not a counter to SUM across ranks/steps.
        metrics["lvd/response_region_count"] = float(response_mode.region_count)
    for name, numerator in numerators.items():
        denominator = denominators[name]
        # Denominators are counts; numerators may also be per-position sums.
        # No observations (also no actual flips) is 0/0 -> 0, never perfect recall.
        metrics[name] = numerator / (
            denominator.clamp_min(1)
            if isinstance(denominator, torch.Tensor)
            else max(denominator, 1.0)
        )
    return _anchor_loss_result(
        context,
        loss=loss,
        metrics=metrics,
        ratio_numerators=numerators,
        ratio_denominators=denominators,
        sum_metrics=sums,
    )


@dataclass(frozen=True)
class TargetCrossEntropy:
    name: str = "target_ce"
    requires_anchor_hidden: bool = False

    def __call__(self, context: LossContext) -> LossResult:
        batch = context.stream_batch("target")
        if batch is None:
            raise ValueError("target CE requires a target batch")
        outputs = context.forward_current("target")
        if "regression_labels" in batch:
            loss, predictions = stsb_numeric_mse(
                outputs.logits,
                batch["regression_labels"],
                batch["regression_positions"],
                batch["regression_integer_token_ids"],
                batch["regression_fraction_token_ids"],
            )
            return LossResult(
                loss=loss,
                metrics={
                    "loss/target_mse": loss.detach(),
                    "stsb/prediction_mean": predictions.detach().mean(),
                    "stsb/label_mean": batch["regression_labels"].detach().float().mean(),
                    "samples/target_regression": float(len(predictions)),
                },
            )
        loss, valid_count = causal_lm_cross_entropy(outputs.logits, batch["labels"])
        return LossResult(
            loss=loss,
            metrics={"loss/target_ce": loss.detach(), "tokens/target_supervised": valid_count},
            ratio_numerators={"loss/target_ce_token_mean": loss.detach().double() * valid_count},
            ratio_denominators={"loss/target_ce_token_mean": valid_count},
            sum_metrics={"tokens/target_supervised": valid_count},
        )


@dataclass(frozen=True)
class AnchorCrossEntropy:
    weight: float
    name: str = "anchor_ce"
    requires_anchor_hidden: bool = False

    def __call__(self, context: LossContext) -> LossResult:
        batch = context.stream_batch("anchor")
        if batch is None:
            zero = context.connected_zero()
            return _anchor_loss_result(
                context,
                loss=zero,
                metrics={"loss/anchor_ce": 0.0, "tokens/anchor_supervised": 0.0},
            )
        outputs = context.forward_current("anchor")
        unscaled_loss, valid_count = causal_lm_cross_entropy(outputs.logits, batch["labels"])
        return _anchor_loss_result(
            context,
            loss=self.weight * unscaled_loss,
            metrics={
                "loss/anchor_ce": unscaled_loss.detach(),
                "loss/anchor_ce_weighted": (self.weight * unscaled_loss).detach(),
                "tokens/anchor_supervised": valid_count,
            },
        )


@dataclass(frozen=True)
class AnchorKLDistillation:
    weight: float
    temperature: float
    name: str = "anchor_kl"
    requires_anchor_hidden: bool = False

    def __call__(self, context: LossContext) -> LossResult:
        batch = context.stream_batch("anchor")
        if batch is None:
            zero = context.connected_zero()
            return _anchor_loss_result(
                context,
                loss=zero,
                metrics={"loss/anchor_kl": 0.0, "tokens/anchor_supervised": 0.0},
            )
        current_logits = context.forward_current("anchor").logits[:, :-1, :].float()
        reference_logits = context.forward_reference("anchor").logits[:, :-1, :].float()
        prediction_mask = batch["labels"][:, 1:].ne(-100)
        valid_count = prediction_mask.sum()
        if not bool(valid_count):
            unscaled_loss = current_logits.sum() * 0.0
        else:
            temperature = self.temperature
            token_kl = functional.kl_div(
                functional.log_softmax(current_logits / temperature, dim=-1),
                functional.softmax(reference_logits / temperature, dim=-1),
                reduction="none",
            ).sum(dim=-1)
            unscaled_loss = (
                token_kl.masked_select(prediction_mask).sum() * temperature**2 / valid_count
            )
        return _anchor_loss_result(
            context,
            loss=self.weight * unscaled_loss,
            metrics={
                "loss/anchor_kl": unscaled_loss.detach(),
                "loss/anchor_kl_weighted": (self.weight * unscaled_loss).detach(),
                "tokens/anchor_supervised": valid_count,
            },
        )


@dataclass(frozen=True)
class NullLVDLoss:
    name: str = "null_lvd"
    requires_anchor_hidden: bool = True

    def __call__(self, context: LossContext) -> LossResult:
        if context.stream_batch("anchor") is None:
            return _anchor_loss_result(
                context,
                loss=context.connected_zero(),
                metrics={"loss/lvd": 0.0, "lvd/selected_positions": 0.0},
            )
        context.forward_current("anchor")
        hidden_states = context.hidden_states.get("anchor")
        if hidden_states is None:
            raise RuntimeError("the null LVD plugin did not receive captured anchor hidden states")
        zero = hidden_states.sum() * 0.0 + context.current_outputs["anchor"].logits.sum() * 0.0
        return _anchor_loss_result(
            context,
            loss=zero,
            metrics={
                "loss/lvd": zero.detach(),
                "lvd/selected_positions": float(hidden_states.numel() // hidden_states.shape[-1]),
            },
        )


class AnchorLVDLoss:
    name = "anchor_lvd"
    requires_anchor_hidden = True

    def __init__(
        self,
        config: LVDConfig,
        reference_cache: ReferenceCache,
        *,
        lambda_value: float,
        unembedding_weight: torch.Tensor | None = None,
        unembedding_bias: torch.Tensor | None = None,
        process_group: torch.distributed.ProcessGroup | None = None,
    ) -> None:
        config.validate()
        if lambda_value < 0:
            raise ValueError("resolved LVD lambda must be non-negative")
        self.config = config
        self.reference_cache = reference_cache
        self.lambda_value = float(lambda_value)
        self.unembedding_weight = unembedding_weight
        self.unembedding_bias = unembedding_bias
        self.process_group = process_group

    def set_lambda(self, value: float) -> None:
        if value < 0 or not torch.isfinite(torch.tensor(value)):
            raise ValueError("resolved LVD lambda must be finite and non-negative")
        self.lambda_value = float(value)

    def _effective_lambda(self, global_step: int) -> float:
        warmup = self.config.lambda_warmup_steps
        if warmup <= 0:
            return self.lambda_value
        return self.lambda_value * min(1.0, (global_step + 1) / warmup)

    def __call__(self, context: LossContext) -> LossResult:
        report_raw_pressure = (
            LVDPressureMode(self.config.pressure_mode) is LVDPressureMode.ASINH_WEIGHTED_SUM
        )
        response_mode = LVDResponseMode(self.config.response_mode)
        batch = context.stream_batch("anchor")
        if batch is None:
            zero = context.connected_zero()
            return _lvd_loss_result(
                context,
                loss=zero,
                metrics={"loss/lvd": 0.0, "lvd/cached_positions": 0.0},
                counts=_lvd_raw_pressure_counts(None) if report_raw_pressure else None,
                response_mode=self.config.response_mode,
            )
        outputs = context.forward_current("anchor")
        hidden = context.hidden_states.get("anchor")
        if hidden is None:
            raise RuntimeError("LVD loss did not receive captured Anchor hidden states")
        required = {
            "selected_anchor_positions",
            "selected_anchor_position_mask",
            "sample_ids",
        }
        missing = sorted(required - set(batch))
        if missing:
            raise ValueError(f"LVD Anchor batch is missing position fields: {missing}")
        positions = batch["selected_anchor_positions"]
        position_mask = batch["selected_anchor_position_mask"]
        sample_ids = batch["sample_ids"]
        requested_positions = int(position_mask.sum().item())
        selected_hidden = []
        selected_logits = []
        cached_rows: list[dict[str, object]] = []
        for batch_index, sample_id in enumerate(sample_ids):
            for position_index in range(positions.shape[1]):
                if not bool(position_mask[batch_index, position_index]):
                    continue
                token_position = int(positions[batch_index, position_index].item())
                cached = self.reference_cache.get_position(str(sample_id), token_position)
                if cached is None:
                    continue
                selected_hidden.append(hidden[batch_index, position_index])
                selected_logits.append(outputs.logits[batch_index, token_position])
                cached_rows.append(cached)

        hidden_size = hidden.shape[-1]
        if not cached_rows:
            local_pressure = hidden.reshape(-1, hidden_size)[:0]
            distributed_loss, global_pressure = (
                distributed_second_moment_frobenius_loss_and_global_pressure(
                    local_pressure,
                    mode=self.config.moment_mode,
                    compensate_distributed_gradient_average=(
                        context.compensate_distributed_gradient_average
                    ),
                    process_group=self.process_group,
                )
            )
            spectrum = pressure_spectrum_statistics(
                global_pressure,
                mode=self.config.moment_mode,
                rows_are_global=True,
            )
            effective_lambda = self._effective_lambda(context.global_step)
            zero = (
                hidden.sum() * 0.0
                + outputs.logits.sum() * 0.0
                + effective_lambda * distributed_loss
            )
            return _lvd_loss_result(
                context,
                loss=zero,
                metrics={
                    "loss/lvd": distributed_loss.detach(),
                    "loss/lvd_weighted": zero.detach(),
                    "lvd/lambda": effective_lambda,
                    "lvd/requested_positions": float(requested_positions),
                    "lvd/cached_positions": 0.0,
                    "lvd/uncached_positions": float(requested_positions),
                    "lvd/global_pressure_positions": float(spectrum.sample_count),
                    "lvd/uncentered_or_centered_trace": spectrum.trace,
                    "lvd/moment_frobenius_squared": spectrum.frobenius_squared,
                },
                counts=_lvd_raw_pressure_counts(None) if report_raw_pressure else None,
                response_mode=self.config.response_mode,
            )

        device = hidden.device
        current_hidden = torch.stack(selected_hidden).float()
        current_logits = torch.stack([logits.detach() for logits in selected_logits]).float()
        reference_hidden = torch.stack(
            [row["hidden_states"] for row in cached_rows]  # type: ignore[list-item]
        ).to(device=device, dtype=torch.float32)
        reference_ids = torch.stack(
            [row["reference_token_ids"] for row in cached_rows]  # type: ignore[list-item]
        ).to(device=device, dtype=torch.long)
        all_cached_margins = torch.stack(
            [row["baseline_margins"] for row in cached_rows]  # type: ignore[list-item]
        )
        if all_cached_margins.shape != current_logits.shape:
            raise ValueError(
                "the d0 cache must contain one vocabulary-indexed margin per model logit"
            )
        competitor_ids, competitor_mask = select_lvd_competitors(
            current_logits=current_logits,
            reference_token_ids=reference_ids,
            top_k=self.config.top_k,
            strategy=self.config.neighbor_strategy,
        )
        cached_margins = all_cached_margins.gather(
            1, competitor_ids.to(all_cached_margins.device)
        ).to(device=device, dtype=torch.float32)
        del all_cached_margins
        if self.unembedding_weight is None:
            head = get_unembedding(context.model)
            source_weight = head.weight.detach()
            source_bias = getattr(head, "bias", None)
        else:
            source_weight = self.unembedding_weight
            source_bias = self.unembedding_bias
        storage_dtype = str(self.reference_cache.manifest["storage_dtype"])
        tolerance = (
            1e-4
            if storage_dtype == "float32"
            else float(
                self.reference_cache.manifest["precision_budget"]["margin_max_absolute_error"]
            )
        )
        result = chunked_lvd_pressure_loss(
            current_hidden=current_hidden,
            reference_hidden=reference_hidden,
            unembedding_weight=source_weight,
            unembedding_bias=source_bias,
            reference_token_ids=reference_ids,
            competitor_ids=competitor_ids,
            baseline_margins=cached_margins,
            competitor_mask=competitor_mask,
            config=self.config,
            margin_tolerance=tolerance,
        )
        unscaled_loss, global_pressure = (
            distributed_second_moment_frobenius_loss_and_global_pressure(
                result.pressure_vectors,
                mode=self.config.moment_mode,
                compensate_distributed_gradient_average=(
                    context.compensate_distributed_gradient_average
                ),
                process_group=self.process_group,
            )
        )
        effective_lambda = self._effective_lambda(context.global_step)
        weighted_loss = effective_lambda * unscaled_loss
        weighted_loss = weighted_loss + outputs.logits.sum() * 0.0
        pressure_norms = result.pressure_vectors.detach().float().norm(dim=-1)
        spectrum = pressure_spectrum_statistics(
            global_pressure,
            mode=self.config.moment_mode,
            rows_are_global=True,
        )
        current_winners = current_logits.detach().argmax(dim=-1)
        flipped = current_winners.ne(reference_ids)
        winner_in_neighbors = (
            competitor_ids.eq(current_winners.unsqueeze(1)).logical_and(competitor_mask).any(dim=-1)
        )
        cached_target_ids = torch.stack(
            [row["target_token_ids"] for row in cached_rows]  # type: ignore[list-item]
        ).to(device=device, dtype=torch.long)
        cached_base_correct = torch.stack(
            [row["base_correct"] for row in cached_rows]  # type: ignore[list-item]
        ).to(device=device, dtype=torch.bool)
        cached_label_mask = cached_target_ids.ne(-100)
        valid_d0 = result.baseline_margins_raw.masked_select(competitor_mask).detach()
        valid_dnew = result.current_margins.masked_select(competitor_mask).detach()
        valid_erosion = result.erosion_ratios.masked_select(competitor_mask).detach()
        metrics: dict[str, torch.Tensor | float] = {
            "loss/lvd": unscaled_loss.detach(),
            "loss/lvd_weighted": weighted_loss.detach(),
            "lvd/lambda": effective_lambda,
            "lvd/requested_positions": float(requested_positions),
            "lvd/cached_positions": float(len(cached_rows)),
            "lvd/uncached_positions": float(requested_positions - len(cached_rows)),
            "lvd/global_pressure_positions": float(spectrum.sample_count),
            "lvd/competitors_per_position": float(competitor_mask.sum(dim=-1).float().mean()),
            "lvd/d0_mean": result.baseline_margins_raw.masked_select(competitor_mask).mean(),
            "lvd/dnew_mean": result.current_margins.masked_select(competitor_mask).mean(),
            "lvd/erosion_ratio_mean": result.erosion_ratios.masked_select(competitor_mask).mean(),
            "lvd/safety_core_violation_rate": result.safety_core_violation_rate,
            "lvd/crossing_rate": result.crossing_rate,
            "lvd/tiny_baseline_margin_rate": result.tiny_baseline_margin_rate,
            "lvd/pressure_norm_mean": pressure_norms.mean(),
            "lvd/pressure_energy_mean": pressure_norms.square().mean(),
            "lvd/uncentered_or_centered_trace": spectrum.trace,
            "lvd/moment_frobenius_squared": spectrum.frobenius_squared,
            "lvd/maximum_eigenvalue": spectrum.maximum_eigenvalue,
            "lvd/effective_rank": spectrum.effective_rank,
            "lvd/normalized_frobenius": spectrum.normalized_frobenius,
            "lvd/maximum_eigenvalue_share": spectrum.maximum_eigenvalue_share,
            "lvd/exponent_clamp_count": result.response.exponent_clamp_count,
            "lvd/maximum_unclamped_exponent": result.response.maximum_unclamped_exponent,
        }
        for name, values in (
            ("d0", valid_d0),
            ("dnew", valid_dnew),
            ("erosion_ratio", valid_erosion),
        ):
            metrics[f"lvd/{name}_q05"] = torch.quantile(values.float(), 0.05)
            metrics[f"lvd/{name}_q50"] = torch.quantile(values.float(), 0.50)
            metrics[f"lvd/{name}_q95"] = torch.quantile(values.float(), 0.95)
        valid_regions = result.response.regions.masked_select(competitor_mask)
        core_violations = (
            valid_erosion.gt(1)
            if response_mode is LVDResponseMode.THREE_SEGMENT
            else valid_erosion.ge(1)
        )
        counts: dict[str, torch.Tensor | float] = {
            "lvd/valid_boundary_count": competitor_mask.sum(),
            "lvd/safety_core_violation_count": core_violations.sum(),
            # Floored d0 can cross before rho reaches the response's boundary.
            "lvd/crossing_count": valid_dnew.le(0).sum(),
            "lvd/tiny_baseline_margin_count": valid_d0.le(self.config.baseline_margin_floor).sum(),
            "lvd/actual_flip_count": flipped.sum(),
            "lvd/recalled_flip_count": flipped.logical_and(winner_in_neighbors).sum(),
            "anchor/base_top1_agreement_cached_count": (~flipped).sum(),
            "anchor/base_correct_retained_count": (~flipped).logical_and(cached_base_correct).sum(),
            "anchor/base_correct_cached_count": cached_base_correct.sum(),
            "anchor/cached_label_correct_count": (
                current_winners.eq(cached_target_ids).logical_and(cached_label_mask).sum()
            ),
            "anchor/cached_label_token_count": cached_label_mask.sum(),
        }
        for region in range(response_mode.region_count):
            counts[f"lvd/response_region_{region}_count"] = valid_regions.eq(region).sum()
        counts.update(
            _lvd_health_counts(
                result,
                config=self.config,
                competitor_mask=competitor_mask,
                current_hidden=current_hidden,
                reference_hidden=reference_hidden,
            )
        )
        return _lvd_loss_result(
            context,
            loss=weighted_loss,
            metrics=metrics,
            counts=counts,
            response_mode=self.config.response_mode,
        )


def build_loss_plugins(
    config: MethodConfig,
    *,
    lvd_config: LVDConfig | None = None,
    reference_cache: ReferenceCache | None = None,
    resolved_lvd_lambda: float | None = None,
    unembedding_weight: torch.Tensor | None = None,
    unembedding_bias: torch.Tensor | None = None,
    lvd_process_group: torch.distributed.ProcessGroup | None = None,
    lvd_plus_config: LVDPlusConfig | None = None,
    resolved_lvd_plus_lambda: float | None = None,
    lvd_plus_anchor_centers: torch.Tensor | None = None,
    lvd_plus_s0_identity: dict[str, Any] | None = None,
    anchor_geometry_config: AnchorGeometryConfig | None = None,
    anchor_geometry_reference: Any | None = None,
) -> tuple[LossPlugin, ...]:
    plugins: list[LossPlugin] = [TargetCrossEntropy()]
    if config.name.uses_dynamic_geometry:
        from .anchor_geometry import AnchorGeometryLoss

        if anchor_geometry_config is None or anchor_geometry_reference is None:
            raise ValueError("dynamic geometry requires its own config and output reference")
        plugins.append(AnchorCrossEntropy(weight=config.anchor_weight))
        geometry = AnchorGeometryLoss(anchor_geometry_config, anchor_geometry_reference)
        geometry.anchor_weight = float(config.anchor_weight)
        plugins.append(geometry)
    elif config.name is MethodName.TARGET_PLUS_ANCHOR_CE:
        plugins.append(AnchorCrossEntropy(weight=config.anchor_weight))
    elif config.name is MethodName.TARGET_PLUS_ANCHOR_KL:
        plugins.append(
            AnchorKLDistillation(
                weight=config.anchor_weight,
                temperature=config.kl_temperature,
            )
        )
    elif config.name is MethodName.TARGET_PLUS_NULL_LVD:
        plugins.append(NullLVDLoss())
    elif config.name.uses_lvd_plus:
        # Keep the independent label-site method out of every legacy d0 path.
        # The late import lets LVD+ reuse the teacher-forced count helper above.
        from .lvd_plus import AnchorLVDPlusLoss

        if reference_cache is not None:
            raise ValueError("target_plus_anchor_lvd+ forbids a reference/d0 cache")
        if lvd_plus_config is None:
            raise ValueError("target_plus_anchor_lvd+ requires LVDPlusConfig")
        s0_kwargs = {}
        if lvd_plus_anchor_centers is not None:
            s0_kwargs["anchor_centers"] = lvd_plus_anchor_centers
        if lvd_plus_s0_identity is not None:
            s0_kwargs["s0_identity"] = lvd_plus_s0_identity
        plugins.append(
            AnchorLVDPlusLoss(
                lvd_plus_config,
                lambda_value=1.0 if resolved_lvd_plus_lambda is None else resolved_lvd_plus_lambda,
                unembedding_weight=unembedding_weight,
                unembedding_bias=unembedding_bias,
                process_group=lvd_process_group,
                **s0_kwargs,
            )
        )
    elif config.name.uses_lvd:
        if lvd_config is None or reference_cache is None:
            raise ValueError("target_plus_anchor_lvd requires LVD config and d0 cache")
        if config.name is MethodName.TARGET_PLUS_ANCHOR_CE_LVD:
            plugins.append(AnchorCrossEntropy(weight=config.anchor_weight))
        initial_lambda = 1.0 if resolved_lvd_lambda is None else resolved_lvd_lambda
        plugins.append(
            AnchorLVDLoss(
                lvd_config,
                reference_cache,
                lambda_value=initial_lambda,
                unembedding_weight=unembedding_weight,
                unembedding_bias=unembedding_bias,
                process_group=lvd_process_group,
            )
        )
    elif config.name is not MethodName.TARGET_ONLY:
        raise ValueError(f"unsupported method: {config.name}")
    return tuple(plugins)
