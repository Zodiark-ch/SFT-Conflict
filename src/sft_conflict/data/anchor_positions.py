"""Deterministic selection of hidden-state positions used by Anchor protection."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

from torch.utils.data import Dataset


class AnchorPositionStrategy(str, Enum):
    LAST_NON_PADDING = "last_non_padding"
    ALL_SELECTED_TOKENS = "all_selected_tokens"
    SAMPLE_K_POSITIONS = "sample_k_positions"


@dataclass(frozen=True)
class AnchorPositionConfig:
    strategy: AnchorPositionStrategy = AnchorPositionStrategy.SAMPLE_K_POSITIONS
    positions_per_sample: int = 4
    seed: int = 42
    exclude_eos: bool = True
    exclude_pad: bool = True

    def validate(self) -> None:
        if self.positions_per_sample <= 0:
            raise ValueError("positions_per_sample must be positive")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["strategy"] = self.strategy.value
        return value


@dataclass(frozen=True)
class AnchorSelection:
    positions: tuple[int, ...]
    target_token_ids: tuple[int, ...]


class AnchorPositionSelector:
    def __init__(
        self,
        config: AnchorPositionConfig,
        *,
        pad_token_id: int | None,
        eos_token_id: int | None,
    ) -> None:
        config.validate()
        self.config = config
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id

    def _sample_seed(self, sample_id: str) -> int:
        digest = hashlib.sha256(f"{self.config.seed}\n{sample_id}".encode()).digest()
        return int.from_bytes(digest[:8], byteorder="big", signed=False)

    def _is_allowed_target(self, token_id: int) -> bool:
        if (
            self.config.exclude_pad
            and self.pad_token_id is not None
            and token_id == self.pad_token_id
        ):
            return False
        return not (
            self.config.exclude_eos
            and self.eos_token_id is not None
            and token_id == self.eos_token_id
        )

    def _response_candidates(self, record: dict[str, Any]) -> list[tuple[int, int]]:
        candidates: list[tuple[int, int]] = []
        input_ids = record["input_ids"]
        labels = record["labels"]
        attention_mask = record["attention_mask"]
        for label_index in range(1, len(labels)):
            target_token_id = int(labels[label_index])
            prediction_position = label_index - 1
            if target_token_id == -100 or not self._is_allowed_target(target_token_id):
                continue
            if not attention_mask[label_index] or not attention_mask[prediction_position]:
                continue
            if int(input_ids[label_index]) != target_token_id:
                raise ValueError(
                    f"supervised label/input mismatch for sample {record['sample_id']} "
                    f"at token {label_index}"
                )
            candidates.append((prediction_position, target_token_id))
        return candidates

    def select(self, record: dict[str, Any]) -> AnchorSelection:
        strategy = self.config.strategy
        if strategy is AnchorPositionStrategy.LAST_NON_PADDING:
            active = [index for index, selected in enumerate(record["attention_mask"]) if selected]
            if not active:
                return AnchorSelection((), ())
            return AnchorSelection((active[-1],), (-100,))

        candidates = self._response_candidates(record)
        if strategy is AnchorPositionStrategy.ALL_SELECTED_TOKENS:
            selected = candidates
        elif strategy is AnchorPositionStrategy.SAMPLE_K_POSITIONS:
            count = min(self.config.positions_per_sample, len(candidates))
            generator = random.Random(self._sample_seed(record["sample_id"]))
            selected = sorted(generator.sample(candidates, count), key=lambda item: item[0])
        else:
            raise ValueError(f"unsupported Anchor position strategy: {strategy}")
        return AnchorSelection(
            tuple(position for position, _ in selected),
            tuple(token_id for _, token_id in selected),
        )


class AnchorPositionDataset(Dataset[dict[str, Any]]):
    """Immutable view that makes selected Anchor positions part of dataset identity."""

    def __init__(self, dataset: Dataset, selector: AnchorPositionSelector) -> None:
        source_fingerprint = getattr(dataset, "fingerprint", None)
        if source_fingerprint is None:
            raise ValueError("Anchor position source dataset must expose a fingerprint")
        self.dataset = dataset
        self.selector = selector
        self.namespace = getattr(dataset, "namespace", type(dataset).__name__)
        self.preprocessing_stats = dict(getattr(dataset, "preprocessing_stats", {}))
        identity = {
            "source_fingerprint": str(source_fingerprint),
            "position_config": selector.config.to_dict(),
            "pad_token_id": selector.pad_token_id,
            "eos_token_id": selector.eos_token_id,
        }
        self.fingerprint = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        source_manifest = getattr(dataset, "manifest", {})
        self.manifest = {
            **dict(source_manifest),
            "fingerprint": self.fingerprint,
            "source_fingerprint": str(source_fingerprint),
            "anchor_positions": identity,
        }

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = dict(self.dataset[index])
        selection = self.selector.select(record)
        record["selected_anchor_positions"] = list(selection.positions)
        record["selected_anchor_target_ids"] = list(selection.target_token_ids)
        return record
