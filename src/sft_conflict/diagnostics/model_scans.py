"""Model scans shared by post-SFT diagnostic workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import Dataset

from sft_conflict.data import CausalLMCollator, ReferenceCache
from sft_conflict.data.collator import model_tensor_inputs
from sft_conflict.models import UnembeddingInputCapture


@dataclass(frozen=True)
class CachedDisplacementCollection:
    displacements: Tensor
    sample_count: int
    position_count: int


@torch.no_grad()
def collect_cached_anchor_displacements(
    model: nn.Module,
    dataset: Dataset,
    reference_cache: ReferenceCache,
    *,
    pad_token_id: int,
    batch_size: int,
    device: torch.device | str,
) -> CachedDisplacementCollection:
    if batch_size <= 0:
        raise ValueError("diagnostic batch size must be positive")
    resolved_device = torch.device(device)
    collator = CausalLMCollator(pad_token_id, pad_to_multiple_of=None)
    displacement_rows: list[Tensor] = []
    was_training = model.training
    model.eval()
    try:
        for start in range(0, len(dataset), batch_size):
            examples = [
                dataset[index] for index in range(start, min(start + batch_size, len(dataset)))
            ]
            batch = collator(examples)
            positions = batch.get("selected_anchor_positions")
            position_mask = batch.get("selected_anchor_position_mask")
            if positions is None or position_mask is None:
                raise ValueError("interference scan requires positioned Anchor examples")
            if not bool(position_mask.any()):
                continue
            model_inputs = {
                key: value.to(resolved_device) for key, value in model_tensor_inputs(batch).items()
            }
            try:
                with UnembeddingInputCapture(
                    model,
                    positions=positions.to(resolved_device),
                    stop_forward=True,
                ) as capture:
                    model(**model_inputs)
            except UnembeddingInputCapture.ForwardStopped:
                pass
            if capture.hidden_states is None:
                raise RuntimeError("interference scan did not capture the Unembedding input")
            current_hidden = capture.hidden_states.float().cpu()
            for batch_index, sample_id in enumerate(batch["sample_ids"]):
                for position_index in range(positions.shape[1]):
                    if not bool(position_mask[batch_index, position_index]):
                        continue
                    token_position = int(positions[batch_index, position_index].item())
                    cached = reference_cache.get_position_fields(
                        str(sample_id),
                        token_position,
                        ("hidden_states",),
                    )
                    if cached is None:
                        raise ValueError(
                            "d0 cache is missing a selected Anchor position: "
                            f"sample_id={sample_id}, token_position={token_position}"
                        )
                    reference_hidden = cached["hidden_states"].float()
                    current_row = current_hidden[batch_index, position_index]
                    if reference_hidden.shape != current_row.shape:
                        raise ValueError("cached and current Anchor hidden-state shapes differ")
                    displacement_rows.append(current_row - reference_hidden)
    finally:
        model.train(was_training)
    if not displacement_rows:
        raise ValueError("interference scan found no cached Anchor positions")
    return CachedDisplacementCollection(
        displacements=torch.stack(displacement_rows),
        sample_count=len(dataset),
        position_count=len(displacement_rows),
    )


@dataclass(frozen=True)
class PredictionCollection:
    predictions: Tensor
    target_token_ids: Tensor
    sample_count: int
    token_count: int


@torch.no_grad()
def collect_supervised_next_token_predictions(
    model: nn.Module,
    dataset: Dataset,
    *,
    pad_token_id: int,
    batch_size: int,
    device: torch.device | str,
) -> PredictionCollection:
    if batch_size <= 0:
        raise ValueError("diagnostic batch size must be positive")
    resolved_device = torch.device(device)
    collator = CausalLMCollator(pad_token_id, pad_to_multiple_of=None)
    prediction_rows: list[Tensor] = []
    target_rows: list[Tensor] = []
    was_training = model.training
    model.eval()
    try:
        for start in range(0, len(dataset), batch_size):
            examples: list[dict[str, Any]] = [
                dataset[index] for index in range(start, min(start + batch_size, len(dataset)))
            ]
            batch = collator(examples)
            model_inputs = {
                key: value.to(resolved_device) for key, value in model_tensor_inputs(batch).items()
            }
            outputs = model(**model_inputs)
            targets = batch["labels"][:, 1:]
            supervised = targets.ne(-100)
            predictions = outputs.logits[:, :-1].argmax(dim=-1).cpu()
            prediction_rows.append(predictions.masked_select(supervised))
            target_rows.append(targets.masked_select(supervised))
    finally:
        model.train(was_training)
    if not target_rows or not sum(row.numel() for row in target_rows):
        raise ValueError("crossing scan found no supervised Anchor response tokens")
    predictions = torch.cat(prediction_rows)
    targets = torch.cat(target_rows)
    return PredictionCollection(
        predictions=predictions,
        target_token_ids=targets,
        sample_count=len(dataset),
        token_count=targets.numel(),
    )
