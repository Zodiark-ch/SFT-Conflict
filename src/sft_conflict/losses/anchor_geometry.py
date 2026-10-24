"""Anchor CE's auxiliary: unequal signed-margin drift with a live output head."""

from __future__ import annotations

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
    normalized_margins,
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
        band_mode = self.config.objective == "base_band_v2"
        band_statistics = (
            dict.fromkeys(
                (
                    "geometry/band_width_mean",
                    "geometry/band_log_width_ratio_mean",
                    "geometry/band_center_mean",
                    "geometry/band_active_rate",
                ),
                zero.detach(),
            )
            if band_mode
            else {}
        )
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
                )
                rows.append(values)
                if band_mode:
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
        coefficient = 1.0 if self.calibrating else self.coefficient(context.global_step)
        return LossResult(
            raw * coefficient,
            {
                "loss/anchor_geometry_raw": raw.detach(),
                "geometry/lambda": coefficient,
                "geometry/active": 1.0,
            },
            numerators,
            denominators,
            counts,
        )

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
