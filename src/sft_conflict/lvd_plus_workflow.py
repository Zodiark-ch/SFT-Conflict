"""Deterministic LVD+ calibration samples from the FULL training sources.

No DataLoader, held-out split, reserved coverage rows, pilot or publication is
involved. Sampling does not consume any global RNG. Training keeps every row.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any

import torch.distributed as distributed
from torch.utils.data import Dataset

from sft_conflict.data import CausalLMCollator


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _world() -> tuple[int, int]:
    if distributed.is_available() and distributed.is_initialized():
        rank, size = distributed.get_rank(), distributed.get_world_size()
    else:
        rank, size = int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))
    if size < 1 or not 0 <= rank < size:
        raise ValueError("LVD+ calibration requires 0 <= rank < world_size")
    return rank, size


def _source_identity(dataset: Dataset | None) -> dict[str, Any]:
    if dataset is None:
        return {"fingerprint": None, "sample_count": None}
    fingerprint = (
        getattr(dataset, "fingerprint", None)
        or getattr(dataset, "_fingerprint", None)
        or getattr(dataset, "stream_fingerprint", None)
    )
    return {
        "fingerprint": str(fingerprint) if fingerprint is not None else None,
        "sample_count": len(dataset),
    }


class LVDPlusCalibrationBatches(list[dict[str, Any]]):
    """Ordinary rank-local CPU batches carrying non-tensor provenance metadata."""

    def __init__(self, batches: list[dict[str, Any]], *, metadata: dict[str, Any]) -> None:
        super().__init__(batches)
        self.plan_metadata = metadata


def lvd_plus_calibration_batches(
    config: Any, target: Dataset, anchor: Dataset, pad_token_id: int
) -> LVDPlusCalibrationBatches:
    """Hash-order each stream independently; never cycle or touch a train loader.

    Each microbatch contains one configured Target batch and one configured
    Anchor batch. Global slots are ``microbatch * world_size + rank``. Both
    sources must supply enough unique IDs for the entire distributed plan;
    tiny callers must explicitly lower group/accumulation counts instead of
    silently repeating examples. The sources are training data, NOT holdouts.
    """
    plus = config.lvd_plus
    plus.validate()
    if not config.method.name.uses_lvd_plus:
        raise ValueError("LVD+ calibration batches require method=target_plus_anchor_lvd+")
    rank, world_size = _world()
    micros = plus.calibration_batches * plus.calibration_accumulation_steps
    sources = {"target": target, "anchor": anchor}
    orders: dict[str, list[int]] = {}
    identities = {}
    selected_ids = {}
    for stream, dataset in sources.items():
        batch_size = getattr(config.data, f"{stream}_batch_size")
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError(f"LVD+ {stream}_batch_size must be a positive integer")
        required = micros * world_size * batch_size
        if dataset is None or len(dataset) < required:
            available = 0 if dataset is None else len(dataset)
            raise ValueError(
                f"LVD+ {stream} requires {required} disjoint nonrepeating training samples, "
                f"received {available}; lower calibration groups/steps for tiny tests (no cycling)"
            )
        identity = _source_identity(dataset)
        if identity["fingerprint"] is None:
            raise ValueError(f"LVD+ {stream} source must expose a full dataset fingerprint")
        ids = [dataset[index].get("sample_id") for index in range(len(dataset))]
        if any(not isinstance(value, str) or not value for value in ids):
            raise ValueError(f"LVD+ {stream} source requires non-empty string sample IDs")
        if len(set(ids)) != len(ids):
            raise ValueError(f"LVD+ {stream} source has duplicate sample IDs")
        orders[stream] = sorted(
            range(len(dataset)),
            key=lambda index: _sha256_json(
                ["lvd-plus-calibration-v1", stream, plus.calibration_seed, ids[index]]
            ),
        )[:required]
        identities[stream] = identity
        selected_ids[stream] = [ids[index] for index in orders[stream]]

    collator = CausalLMCollator(pad_token_id, config.data.pad_to_multiple_of)
    batches, local_indices = [], []
    for micro in range(micros):
        slot = micro * world_size + rank
        batch, indices = {}, {}
        for stream, dataset in sources.items():
            batch_size = getattr(config.data, f"{stream}_batch_size")
            start = slot * batch_size
            indices[stream] = orders[stream][start : start + batch_size]
            batch[stream] = collator([dataset[index] for index in indices[stream]])
        batches.append(batch)
        local_indices.append(indices)
    return LVDPlusCalibrationBatches(
        batches,
        metadata={
            "version": 1,
            "source": "full_training_sources",
            "selection": "independent_stream_sha256_without_replacement",
            "seed": plus.calibration_seed,
            "rank": rank,
            "world_size": world_size,
            "source_datasets": identities,
            "global_selection_sha256": _sha256_json(selected_ids),
            "local_source_indices": local_indices,
            "rank_order": "global_slot = microbatch_index * world_size + rank",
            "anchor_batch_rule": "one configured Anchor batch per calibration microbatch",
            "training_rows_removed": 0,
            "held_out": False,
        },
    )
