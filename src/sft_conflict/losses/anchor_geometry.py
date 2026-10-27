"""Anchor CE's auxiliary: unequal signed-margin drift with a live output head."""

from __future__ import annotations

import math

import torch
import torch.distributed as distributed
from torch import Tensor

from sft_conflict.config import AnchorGeometryConfig
from sft_conflict.data.anchor_geometry_reference import AnchorGeometryReference, json_digest
from sft_conflict.geometry.boundary_risk import distributed_global_mean
from sft_conflict.geometry.dynamic import (
    base_conditioned_band_widths,
    chunked_centered_huber_rows,
    dynamic_competitors,
    immediate_hidden_gradients,
    normalized_margins,
    relative_margin_gradients,
)

from .base import LossContext, LossResult


def _collective_geometry_error(error: Exception | None, device) -> None:
    active = distributed.is_available() and distributed.is_initialized()
    if active:
        failed = torch.tensor(error is not None, dtype=torch.long, device=device)
        distributed.all_reduce(failed, op=distributed.ReduceOp.MAX)
        if failed.item():
            raise ValueError(
                str(error) if error is not None else "geometry failure on another rank"
            ) from error
    elif error is not None:
        raise error


class AnchorGeometryLoss:
    name = "anchor_dynamic_geometry"
    requires_anchor_hidden = True

    def __init__(self, config: AnchorGeometryConfig, reference: AnchorGeometryReference) -> None:
        config.validate()
        self.config = config
        self.reference = reference
        self.calibrating = False
        self.max_frequency = int(reference.frequencies.max())
        # Weight of the Anchor CE term this auxiliary accompanies (set by the plugin
        # builder); only used to express the immediate-gradient diagnostic ratio.
        self.anchor_weight = 1.0

    def _band_widths(self, record, column: int, label: int, ids: Tensor) -> Tensor:
        initial_hidden = record["hidden"][column : column + 1]
        scale_ids = record["top_ids"][column, : self.config.band_scale_top_k]
        lengths = (self.reference.weight[scale_ids] - self.reference.weight[label]).norm(dim=-1)
        valid_scale_ids = scale_ids[lengths > self.config.normal_epsilon]
        if len(valid_scale_ids):
            scale_margins = normalized_margins(
                initial_hidden,
                self.reference.weight,
                torch.tensor([label]),
                valid_scale_ids[None],
                bias=self.reference.bias,
                normal_epsilon=self.config.normal_epsilon,
                hidden_epsilon=self.config.hidden_epsilon,
            )
        else:
            scale_margins = torch.zeros(1, 1)
        peak_id = record["top_ids"][column, 0]
        base_logits = initial_hidden @ self.reference.weight[ids].t()
        peak_logits = initial_hidden @ self.reference.weight[peak_id]
        if self.reference.bias is not None:
            base_logits += self.reference.bias[ids][None]
            peak_logits += self.reference.bias[peak_id]
        return base_conditioned_band_widths(
            base_logits,
            peak_logits,
            scale_margins,
            self.reference.frequencies[ids][None],
            max_frequency=self.max_frequency,
            scale_ratio=self.config.band_scale_ratio,
            temperature=self.config.band_temperature,
            frequency_strength=self.config.band_frequency_strength,
            epsilon=self.config.hidden_epsilon,
        )[0]

    @torch.no_grad()
    def _immediate_hidden_statistics(
        self,
        weight: Tensor,
        logits: Tensor,
        positions: Tensor,
        mask: Tensor,
        labels: Tensor,
        selected_hidden: Tensor,
        live_weight: Tensor,
        local_ids: Tensor,
        margins: Tensor,
        reference_values: Tensor,
        weights: Tensor,
    ) -> tuple[Tensor, Tensor, int, int]:
        """Squared immediate hidden-gradient norms (before the model Jacobian).

        Geometry: the relative objective's per-position rows on the eligible positions;
        Anchor CE: unit-weight token CE on every supervised Anchor token of this batch.
        Both exclude their mean normalisation, which the caller applies.
        """
        margin_gradient = relative_margin_gradients(
            margins,
            reference_values,
            weights,
            tolerance=self.config.relative_tolerance,
            delta=self.config.relative_delta,
            floor=self.config.margin_floor,
        )
        geometry_gradients = immediate_hidden_gradients(
            selected_hidden.detach(),
            live_weight.detach(),
            local_ids[:, 0],
            local_ids[:, 1:],
            margin_gradient,
            normal_epsilon=self.config.normal_epsilon,
            hidden_epsilon=self.config.hidden_epsilon,
            position_chunk_size=self.config.position_chunk_size,
        )
        batch_rows = torch.arange(positions.shape[0], device=positions.device)[:, None]
        rows = batch_rows.expand_as(positions)
        token_logits = logits.detach()[rows[mask], positions[mask]].float()
        token_labels = labels[mask]
        full_weight = weight.detach().float()
        ce_gradients = torch.softmax(token_logits, dim=-1) @ full_weight - full_weight[token_labels]
        return (
            geometry_gradients.float().square().sum(),
            ce_gradients.square().sum(),
            len(selected_hidden),
            int(token_labels.numel()),
        )

    def coefficient(self, step: int) -> float:
        if step < self.config.geometry_start_step:
            return 0.0
        if self.config.resolved_lambda is None:
            raise RuntimeError("dynamic geometry must calibrate after CE warmup before training")
        progress = (step - self.config.geometry_start_step + 1) / self.config.geometry_ramp_steps
        return self.config.resolved_lambda * min(1.0, progress)

    def observe(
        self,
        context: LossContext,
        weight: Tensor,
        bias: Tensor | None,
        hidden: Tensor,
        logits: Tensor,
    ) -> LossResult:
        zero = logits.sum() * 0
        if not self.calibrating and context.global_step < self.config.geometry_start_step:
            return LossResult(zero, {"geometry/lambda": 0.0, "geometry/active": 0.0})
        error = None
        rows, common, counts = [], [], {}
        immediate = torch.zeros(4, dtype=torch.float64, device=logits.device)
        band_mode = self.config.objective == "base_band_v2"
        relative_mode = self.config.objective == "relative_band_v3"
        if band_mode:
            statistic_names = (
                "geometry/band_width_mean",
                "geometry/band_log_width_ratio_mean",
                "geometry/band_center_mean",
                "geometry/band_active_rate",
            )
        elif relative_mode:
            statistic_names = (
                "geometry/relative_erosion_mean",
                "geometry/relative_excess_mean",
                "geometry/relative_active_rate",
            )
        else:
            statistic_names = ()
        band_statistics = dict.fromkeys(statistic_names, zero.detach())
        try:
            positions, mask, labels = context.anchor_label_selection()
            batch = context.stream_batch("anchor")
            if weight.shape != self.reference.weight.shape:
                raise ValueError("live head does not match the Base reference shape")
            if not self.calibrating:
                self.coefficient(context.global_step)
            counts = {
                "geometry/base_correct_count": 0,
                "geometry/base_correct_retained_count": 0,
                "geometry/base_wrong_count": 0,
                "geometry/base_wrong_recovered_count": 0,
                "geometry/current_direction_count": 0,
                "geometry/base_direction_count": 0,
                "geometry/coverage_direction_count": 0,
                "geometry/valid_pair_count": 0,
                "geometry/crossing_count": 0,
                "geometry/skipped_reference_position_count": 0,
            }
            current_predictions = logits.detach().argmax(dim=-1)
            selected_rows = []
            for batch_row, sample_id in enumerate(batch["sample_ids"]):
                record = self.reference.record(sample_id)
                local_positions = positions[batch_row, mask[batch_row]].detach().cpu()
                local_labels = labels[batch_row, mask[batch_row]].detach().cpu()
                if not torch.equal(local_positions, record["positions"]) or not torch.equal(
                    local_labels, record["labels"]
                ):
                    raise ValueError("Anchor causal positions/labels disagree with Base reference")
                for column, (position, label) in enumerate(
                    zip(
                        local_positions.tolist(),
                        local_labels.tolist(),
                        strict=True,
                    )
                ):
                    retained = int(current_predictions[batch_row, position].item() == label)
                    if not record["correct"][column]:
                        counts["geometry/base_wrong_count"] += 1
                        counts["geometry/base_wrong_recovered_count"] += retained
                        continue
                    counts["geometry/base_correct_count"] += 1
                    counts["geometry/base_correct_retained_count"] += retained
                    if record["hidden"][column].norm() <= self.config.hidden_epsilon:
                        counts["geometry/skipped_reference_position_count"] += 1
                        continue
                    seed = int(
                        json_digest(
                            [
                                "anchor-geometry-coverage-v1",
                                self.config.coverage_seed,
                                sample_id,
                                position,
                                context.anchor_geometry_epoch,
                            ]
                        )[:16],
                        16,
                    ) % (2**63 - 1)
                    ids, weights, sizes = dynamic_competitors(
                        logits[batch_row, position],
                        record["top_ids"][column],
                        self.reference.weight,
                        label,
                        self.reference.frequencies,
                        self.config,
                        seed=seed,
                    )
                    if not len(ids):
                        counts["geometry/skipped_reference_position_count"] += 1
                        continue
                    for group, size in zip(("current", "base", "coverage"), sizes, strict=True):
                        counts[f"geometry/{group}_direction_count"] += size
                    counts["geometry/valid_pair_count"] += len(ids)
                    with torch.no_grad():
                        reference = normalized_margins(
                            record["hidden"][column : column + 1],
                            self.reference.weight,
                            torch.tensor([label]),
                            ids[None, :],
                            bias=self.reference.bias,
                            normal_epsilon=self.config.normal_epsilon,
                            hidden_epsilon=self.config.hidden_epsilon,
                        )[0]
                        band_widths = (
                            self._band_widths(record, column, label, ids) if band_mode else None
                        )
                    selected_rows.append(
                        (hidden[batch_row, column], label, ids, weights, reference, band_widths)
                    )
            if selected_rows:
                width = max(len(entry[2]) for entry in selected_rows)
                candidate_ids = torch.stack(
                    [
                        torch.cat((entry[2], entry[2][-1:].expand(width - len(entry[2]))))
                        for entry in selected_rows
                    ]
                ).to(hidden.device)
                weights = torch.stack(
                    [
                        torch.nn.functional.pad(entry[3], (0, width - len(entry[3])))
                        for entry in selected_rows
                    ]
                ).to(hidden.device)
                reference_values = torch.stack(
                    [
                        torch.nn.functional.pad(entry[4], (0, width - len(entry[4])))
                        for entry in selected_rows
                    ]
                ).to(hidden.device)
                selected_hidden = torch.stack([entry[0] for entry in selected_rows])
                selected_labels = torch.tensor(
                    [entry[1] for entry in selected_rows],
                    device=hidden.device,
                )
                # The head observer merges these row gradients into its dense linear
                # gradient; ordinary autograd sparse/dense addition can copy the head.
                vocabulary_ids, local_ids = torch.unique(
                    torch.cat((selected_labels[:, None], candidate_ids), dim=1),
                    return_inverse=True,
                )
                live_weight = torch.nn.functional.embedding(vocabulary_ids, weight, sparse=True)
                live_bias = None if bias is None else bias[vocabulary_ids]
                band_widths = (
                    torch.stack(
                        [
                            torch.nn.functional.pad(entry[5], (0, width - len(entry[5])))
                            for entry in selected_rows
                        ]
                    ).to(hidden.device)
                    if band_mode
                    else None
                )
                values, shift, margins = chunked_centered_huber_rows(
                    selected_hidden,
                    live_weight,
                    local_ids[:, 0],
                    local_ids[:, 1:],
                    reference_values,
                    weights,
                    bias=live_bias,
                    delta=self.config.geometry_delta,
                    normal_epsilon=self.config.normal_epsilon,
                    hidden_epsilon=self.config.hidden_epsilon,
                    position_chunk_size=self.config.position_chunk_size,
                    widths=band_widths,
                    solver_iterations=self.config.band_solver_iterations,
                    relative=(
                        {
                            "tolerance": self.config.relative_tolerance,
                            "delta": self.config.relative_delta,
                            "floor": self.config.margin_floor,
                            "surrogate": (
                                {
                                    "saturated_slope": "all",
                                    "nearest_direction_slope": "nearest",
                                }.get(self.config.calibration_mode)
                                if self.calibrating
                                else None
                            ),
                        }
                        if relative_mode
                        else None
                    ),
                    head_gradient_scale=(self.config.head_gradient_scale if relative_mode else 1.0),
                    hidden_gradient_scale=(
                        self.config.hidden_gradient_scale if relative_mode else 1.0
                    ),
                )
                rows.append(values)
                if relative_mode:
                    signed_drift = margins.detach() - reference_values
                    common.append((signed_drift * weights).sum(dim=-1))
                    valid = weights > 0
                    erosion = -signed_drift / reference_values.clamp_min(self.config.margin_floor)
                    excess = (erosion - self.config.relative_tolerance).clamp_min(0)
                    band_statistics["geometry/relative_erosion_mean"] = (erosion * weights).sum()
                    band_statistics["geometry/relative_excess_mean"] = (excess * weights).sum()
                    band_statistics["geometry/relative_active_rate"] = ((excess > 0) & valid).sum()
                    geometry_sq, ce_sq, eligible, supervised = self._immediate_hidden_statistics(
                        weight,
                        logits,
                        positions,
                        mask,
                        labels,
                        selected_hidden,
                        live_weight,
                        local_ids,
                        margins,
                        reference_values,
                        weights,
                    )
                    immediate = torch.stack(
                        [
                            geometry_sq.double(),
                            ce_sq.double(),
                            torch.tensor(
                                float(eligible), dtype=torch.float64, device=geometry_sq.device
                            ),
                            torch.tensor(
                                float(supervised), dtype=torch.float64, device=geometry_sq.device
                            ),
                        ]
                    ).to(logits.device)
                elif band_mode:
                    signed_drift = margins.detach() - reference_values
                    common.append((signed_drift * weights).sum(dim=-1))
                    valid = weights > 0
                    band_statistics["geometry/band_width_mean"] = (band_widths * weights).sum()
                    minimum = band_widths.masked_fill(~valid, torch.inf).amin(dim=-1)
                    maximum = band_widths.masked_fill(~valid, 0).amax(dim=-1)
                    band_statistics["geometry/band_log_width_ratio_mean"] = (
                        maximum.log() - minimum.log()
                    ).sum()
                    band_statistics["geometry/band_center_mean"] = shift.detach().sum()
                    active = ((signed_drift - shift[:, None]).abs() > band_widths) | margins.lt(0)
                    band_statistics["geometry/band_active_rate"] = (active & valid).sum()
                else:
                    common.append(shift.detach())
                counts["geometry/crossing_count"] += int(
                    (margins.detach().lt(0) & weights.gt(0)).sum().item()
                )
        except (ValueError, RuntimeError, KeyError, IndexError) as exc:
            error = exc
        _collective_geometry_error(error, logits.device)
        coefficient = 1.0 if self.calibrating else self.coefficient(context.global_step)
        immediate_ratio, cap_active = 0.0, 0.0
        if relative_mode:
            # Global (all-rank) immediate pull ratio; every rank participates, also those
            # without eligible positions, so the cap is identical across ranks.
            if distributed.is_available() and distributed.is_initialized():
                distributed.all_reduce(immediate, op=distributed.ReduceOp.SUM)
            geometry_sq, ce_sq, eligible, supervised = immediate.tolist()
            geometry_norm = coefficient * math.sqrt(geometry_sq) / max(eligible, 1.0)
            ce_norm = self.anchor_weight * math.sqrt(ce_sq) / max(supervised, 1.0)
            immediate_ratio = geometry_norm / ce_norm if ce_norm > 0 else 0.0
            cap = self.config.pull_cap_ratio
            if (
                cap is not None
                and not self.calibrating
                and immediate_ratio > cap
                and math.isfinite(immediate_ratio)
            ):
                coefficient *= cap / immediate_ratio
                immediate_ratio = cap
                cap_active = 1.0
            counts["geometry/immediate_geometry_hidden_sq"] = coefficient**2 * float(immediate[0])
            counts["geometry/immediate_anchor_ce_hidden_sq"] = self.anchor_weight**2 * float(
                immediate[1]
            )
        local_rows = torch.cat(rows) if rows else zero.expand(0)
        raw = (
            distributed_global_mean(
                local_rows,
                compensate_distributed_gradient_average=(
                    context.compensate_distributed_gradient_average
                ),
            )
            + zero
        )
        common_values = torch.cat(common) if common else torch.empty(0, device=logits.device)
        counts["geometry/eligible_position_count"] = len(local_rows)
        numerators = {
            "geometry/common_drift_mean": common_values.sum(),
            "geometry/common_negative_rate": common_values.lt(0).sum(),
            "geometry/base_correct_retention": counts["geometry/base_correct_retained_count"],
            "geometry/base_wrong_recovery": counts["geometry/base_wrong_recovered_count"],
            "geometry/crossing_rate": counts["geometry/crossing_count"],
        }
        denominators = {
            "geometry/common_drift_mean": len(local_rows),
            "geometry/common_negative_rate": len(local_rows),
            "geometry/base_correct_retention": counts["geometry/base_correct_count"],
            "geometry/base_wrong_recovery": counts["geometry/base_wrong_count"],
            "geometry/crossing_rate": counts["geometry/valid_pair_count"],
        }
        numerators.update(band_statistics)
        denominators.update({name: len(local_rows) for name in band_statistics})
        if band_mode:
            denominators["geometry/band_active_rate"] = counts["geometry/valid_pair_count"]
        if relative_mode:
            denominators["geometry/relative_active_rate"] = counts["geometry/valid_pair_count"]
        metrics = {
            "loss/anchor_geometry_raw": raw.detach(),
            "geometry/lambda": coefficient,
            "geometry/active": 1.0,
        }
        if relative_mode:
            # Global ratio of the geometry's immediate hidden gradient (with its effective
            # coefficient and 1/N mean) to the weighted Anchor CE token-mean gradient.
            metrics["geometry/immediate_hidden_ratio"] = immediate_ratio
            metrics["geometry/pull_cap_active"] = cap_active
        return LossResult(raw * coefficient, metrics, numerators, denominators, counts)

    def __call__(self, context: LossContext) -> LossResult:
        if context.stream_batch("anchor") is None:
            zero = context.connected_zero()
            if not self.calibrating and context.global_step < self.config.geometry_start_step:
                return LossResult(zero, {"geometry/active": 0.0, "geometry/lambda": 0.0})
            _collective_geometry_error(None, zero.device)
            raw = (
                distributed_global_mean(
                    zero.expand(0),
                    compensate_distributed_gradient_average=context.compensate_distributed_gradient_average,
                )
                + zero
            )
            coefficient = 1.0 if self.calibrating else self.coefficient(context.global_step)
            return LossResult(
                raw * coefficient,
                {
                    "loss/anchor_geometry_raw": raw.detach(),
                    "geometry/active": 1.0,
                    "geometry/lambda": coefficient,
                },
            )
        context.forward_current("anchor")
        if context.anchor_geometry_result is None:
            raise RuntimeError("dynamic geometry requires a live-head forward observer")
        return context.anchor_geometry_result
