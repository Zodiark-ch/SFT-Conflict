"""Pure statistics used by post-SFT interference and crossing workflows."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class TokenDirectionMovementStatistics:
    mean_positive_movement: Tensor
    mean_absolute_movement: Tensor
    mean_signed_movement: Tensor
    root_mean_square_movement: Tensor
    maximum_absolute_movement: Tensor
    total_positive_movement: float
    total_absolute_movement: float
    mean_hidden_l2_displacement: float
    position_count: int


@torch.no_grad()
def token_direction_movement_statistics(
    displacements: Tensor,
    token_directions: Tensor,
    *,
    token_chunk_size: int = 256,
    position_chunk_size: int = 1024,
    normal_epsilon: float = 1e-12,
) -> TokenDirectionMovementStatistics:
    if displacements.ndim != 2 or token_directions.ndim != 2:
        raise ValueError("displacements and token directions must have shapes [N, H] and [V, H]")
    if displacements.shape[1] != token_directions.shape[1]:
        raise ValueError("displacement and token-direction hidden dimensions do not match")
    if not displacements.shape[0] or not token_directions.shape[0]:
        raise ValueError("movement statistics require at least one position and token direction")
    if token_chunk_size <= 0 or position_chunk_size <= 0:
        raise ValueError("movement statistic chunk sizes must be positive")
    if normal_epsilon <= 0:
        raise ValueError("normal_epsilon must be positive")

    movement = displacements.float()
    mean_positive_rows: list[Tensor] = []
    mean_absolute_rows: list[Tensor] = []
    mean_signed_rows: list[Tensor] = []
    root_mean_square_rows: list[Tensor] = []
    maximum_absolute_rows: list[Tensor] = []
    position_count = movement.shape[0]
    for token_start in range(0, token_directions.shape[0], token_chunk_size):
        token_end = min(token_start + token_chunk_size, token_directions.shape[0])
        directions = token_directions[token_start:token_end].to(
            device=movement.device,
            dtype=torch.float32,
        )
        direction_norms = torch.linalg.vector_norm(directions, dim=-1)
        if bool((direction_norms <= normal_epsilon).any()):
            raise ValueError("an Unembedding token direction has zero or near-zero norm")
        unit_directions = directions / direction_norms.unsqueeze(1)
        absolute_sum = torch.zeros(len(directions), device=movement.device)
        positive_sum = torch.zeros_like(absolute_sum)
        signed_sum = torch.zeros_like(absolute_sum)
        squared_sum = torch.zeros_like(absolute_sum)
        absolute_maximum = torch.zeros_like(absolute_sum)
        for position_start in range(0, position_count, position_chunk_size):
            position_end = min(position_start + position_chunk_size, position_count)
            projections = movement[position_start:position_end] @ unit_directions.transpose(0, 1)
            absolute = projections.abs()
            positive_sum += projections.clamp_min(0).sum(dim=0)
            absolute_sum += absolute.sum(dim=0)
            signed_sum += projections.sum(dim=0)
            squared_sum += projections.square().sum(dim=0)
            absolute_maximum = torch.maximum(absolute_maximum, absolute.amax(dim=0))
        mean_positive_rows.append((positive_sum / position_count).cpu())
        mean_absolute_rows.append((absolute_sum / position_count).cpu())
        mean_signed_rows.append((signed_sum / position_count).cpu())
        root_mean_square_rows.append((squared_sum / position_count).sqrt().cpu())
        maximum_absolute_rows.append(absolute_maximum.cpu())

    mean_positive = torch.cat(mean_positive_rows)
    mean_absolute = torch.cat(mean_absolute_rows)
    return TokenDirectionMovementStatistics(
        mean_positive_movement=mean_positive,
        mean_absolute_movement=mean_absolute,
        mean_signed_movement=torch.cat(mean_signed_rows),
        root_mean_square_movement=torch.cat(root_mean_square_rows),
        maximum_absolute_movement=torch.cat(maximum_absolute_rows),
        total_positive_movement=float(mean_positive.sum().item()),
        total_absolute_movement=float(mean_absolute.sum().item()),
        mean_hidden_l2_displacement=float(movement.norm(dim=-1).mean().item()),
        position_count=position_count,
    )


@dataclass(frozen=True)
class CrossingStatistics:
    token_count: int
    base_correct_count: int
    base_mismatch_count: int
    tuned_correct_count: int
    tuned_mismatch_count: int
    new_crossing_count: int
    recovered_count: int
    prediction_flip_count: int
    net_mismatch_increase: int
    base_accuracy: float
    tuned_accuracy: float


@torch.no_grad()
def crossing_statistics(
    base_predictions: Tensor,
    tuned_predictions: Tensor,
    target_token_ids: Tensor,
) -> CrossingStatistics:
    if (
        base_predictions.shape != tuned_predictions.shape
        or base_predictions.shape != target_token_ids.shape
    ):
        raise ValueError("base predictions, tuned predictions, and targets must have equal shapes")
    base = base_predictions.reshape(-1)
    tuned = tuned_predictions.reshape(-1)
    targets = target_token_ids.reshape(-1)
    if not targets.numel():
        raise ValueError("crossing statistics require at least one supervised token")
    base_correct = base.eq(targets)
    tuned_correct = tuned.eq(targets)
    base_correct_count = int(base_correct.sum().item())
    tuned_correct_count = int(tuned_correct.sum().item())
    token_count = targets.numel()
    base_mismatch_count = token_count - base_correct_count
    tuned_mismatch_count = token_count - tuned_correct_count
    return CrossingStatistics(
        token_count=token_count,
        base_correct_count=base_correct_count,
        base_mismatch_count=base_mismatch_count,
        tuned_correct_count=tuned_correct_count,
        tuned_mismatch_count=tuned_mismatch_count,
        new_crossing_count=int((base_correct & ~tuned_correct).sum().item()),
        recovered_count=int((~base_correct & tuned_correct).sum().item()),
        prediction_flip_count=int(base.ne(tuned).sum().item()),
        net_mismatch_increase=tuned_mismatch_count - base_mismatch_count,
        base_accuracy=base_correct_count / token_count,
        tuned_accuracy=tuned_correct_count / token_count,
    )
