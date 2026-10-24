"""Deterministic target/anchor pairing without mutable loader cursors."""

from __future__ import annotations

import hashlib
import math
from fractions import Fraction
from typing import Any

import torch
from torch.utils.data import Dataset, Sampler

from sft_conflict.config import AnchorSamplingStrategy


def _fingerprint(dataset: Dataset) -> str:
    value = (
        getattr(dataset, "stream_fingerprint", None)
        or getattr(dataset, "fingerprint", None)
        or getattr(dataset, "_fingerprint", None)
    )
    if value is None:
        raise ValueError("dual-stream source datasets must expose a stable fingerprint")
    return str(value)


class DualStreamDataset(Dataset[dict[str, Any]]):
    """Virtual epoch plan whose item mapping is fully reconstructible from seed and epoch."""

    def __init__(
        self,
        target_dataset: Dataset,
        anchor_dataset: Dataset | None,
        *,
        target_batch_size: int,
        anchor_batch_size: int,
        anchor_batches_per_target: float,
        world_size: int,
        seed: int,
        steps_per_epoch: int | None = None,
        anchor_sampling_strategy: AnchorSamplingStrategy = AnchorSamplingStrategy.TASK_BALANCED,
    ) -> None:
        if len(target_dataset) == 0:
            raise ValueError("target dataset is empty")
        if target_batch_size <= 0 or anchor_batch_size <= 0 or world_size <= 0:
            raise ValueError("batch sizes and world size must be positive")
        if anchor_batches_per_target < 0:
            raise ValueError("anchor_batches_per_target must be non-negative")
        if anchor_batches_per_target > 0 and anchor_dataset is None:
            raise ValueError("a positive anchor ratio requires an anchor dataset")
        if anchor_dataset is not None and len(anchor_dataset) == 0:
            raise ValueError("anchor dataset is empty")

        self.target_dataset = target_dataset
        self.anchor_dataset = anchor_dataset
        self.target_batch_size = target_batch_size
        self.anchor_batch_size = anchor_batch_size
        self.anchor_ratio = Fraction(str(anchor_batches_per_target))
        self.world_size = world_size
        self.seed = seed
        self.anchor_sampling_strategy = AnchorSamplingStrategy(anchor_sampling_strategy)
        self.epoch = 0
        self._resume_epoch_guard: int | None = None
        if steps_per_epoch is None:
            per_rank_steps = math.ceil(len(target_dataset) / (target_batch_size * world_size))
        else:
            if steps_per_epoch <= 0:
                raise ValueError("steps_per_epoch must be positive")
            per_rank_steps = steps_per_epoch
        self.per_rank_steps = per_rank_steps
        self.global_batch_count = per_rank_steps * world_size
        self._target_permutation: list[int] | None = None
        self._anchor_permutation: list[int] | None = None
        self._anchor_sequence: list[int] | None = None
        self.target_fingerprint = _fingerprint(target_dataset)
        self.anchor_fingerprint = (
            _fingerprint(anchor_dataset) if anchor_dataset is not None else None
        )
        if (
            anchor_dataset is not None
            and self.anchor_sampling_strategy is AnchorSamplingStrategy.TASK_BALANCED
            and any(
                row.get("source_metadata", {}).get("task_metadata_available") is False
                for row in anchor_dataset
            )
        ):
            raise ValueError(
                "task_balanced Anchor sampling requires authoritative task metadata; "
                "provide data.anchor.metadata_path or use mixture_proportional"
            )

    def __len__(self) -> int:
        return self.global_batch_count * self.target_batch_size

    def preserve_epoch_for_resume(self, epoch: int) -> None:
        """Guard the first skipped-batch iterator against Accelerate 1.3's reset.

        skip_first_batches constructs a fresh DataLoaderShard with iteration=0,
        after Trainer has already selected the resumed epoch. Consume this guard
        on that stale reset, or discard it on advancement with newer Accelerate.
        Only the Trainer's mid-epoch resume path arms it; ordinary resets remain
        supported (including calibration followed by fresh training).
        """
        self.set_epoch(epoch)
        self._resume_epoch_guard = epoch if epoch > 0 else None

    def set_epoch(self, epoch: int) -> None:
        if self._resume_epoch_guard is not None:
            restored_epoch = self._resume_epoch_guard
            if epoch < restored_epoch:
                epoch = restored_epoch
                self._resume_epoch_guard = None
            elif epoch > restored_epoch:
                self._resume_epoch_guard = None
        if epoch != self.epoch:
            self.epoch = epoch
            self._target_permutation = None
            self._anchor_permutation = None
            self._anchor_sequence = None

    def _permutation(self, length: int, *, anchor: bool) -> list[int]:
        cached = self._anchor_permutation if anchor else self._target_permutation
        if cached is not None:
            return cached
        generator = torch.Generator()
        stream_offset = 1_000_003 if anchor else 0
        generator.manual_seed(self.seed + self.epoch * 10_000_019 + stream_offset)
        permutation = torch.randperm(length, generator=generator).tolist()
        if anchor:
            self._anchor_permutation = permutation
        else:
            self._target_permutation = permutation
        return permutation

    def _anchor_batch_count(self, global_batch_index: int) -> tuple[int, int]:
        numerator = self.anchor_ratio.numerator
        denominator = self.anchor_ratio.denominator
        consumed_before = global_batch_index * numerator // denominator
        consumed_after = (global_batch_index + 1) * numerator // denominator
        return consumed_before, consumed_after - consumed_before

    def _anchor_task(self, index: int) -> str:
        if self.anchor_dataset is None:
            raise RuntimeError("anchor task requested without an Anchor dataset")
        metadata = self.anchor_dataset[index].get("source_metadata", {})
        return str(
            metadata.get("task_name")
            or metadata.get("task")
            or metadata.get("task_source")
            or "unknown"
        )

    def _build_anchor_sequence(self) -> list[int]:
        if self.anchor_dataset is None:
            return []
        total_batches = (
            self.global_batch_count * self.anchor_ratio.numerator
        ) // self.anchor_ratio.denominator
        total_examples = total_batches * self.anchor_batch_size
        if total_examples == 0:
            return []
        permutation = self._permutation(len(self.anchor_dataset), anchor=True)
        if self.anchor_sampling_strategy is AnchorSamplingStrategy.MIXTURE_PROPORTIONAL:
            return [permutation[index % len(permutation)] for index in range(total_examples)]

        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch * 10_000_019 + 2_000_033)
        if self.anchor_sampling_strategy is AnchorSamplingStrategy.RANDOM:
            return torch.randint(
                len(self.anchor_dataset),
                (total_examples,),
                generator=generator,
            ).tolist()

        groups: dict[str, list[int]] = {}
        for index in permutation:
            groups.setdefault(self._anchor_task(index), []).append(index)
        task_names = sorted(groups)
        task_order = torch.randperm(len(task_names), generator=generator).tolist()
        task_names = [task_names[index] for index in task_order]
        offsets = {task_name: 0 for task_name in task_names}
        sequence: list[int] = []
        while len(sequence) < total_examples:
            for task_name in task_names:
                group = groups[task_name]
                sequence.append(group[offsets[task_name] % len(group)])
                offsets[task_name] += 1
                if len(sequence) == total_examples:
                    break
        return sequence

    def _anchor_indices(self) -> list[int]:
        if self._anchor_sequence is None:
            self._anchor_sequence = self._build_anchor_sequence()
        return self._anchor_sequence

    def __getitem__(self, virtual_index: int) -> dict[str, Any]:
        if virtual_index < 0 or virtual_index >= len(self):
            raise IndexError(virtual_index)
        global_batch_index, position_in_batch = divmod(virtual_index, self.target_batch_size)
        target_permutation = self._permutation(len(self.target_dataset), anchor=False)
        target_index = target_permutation[virtual_index % len(target_permutation)]

        anchors: list[dict[str, Any]] = []
        if self.anchor_dataset is not None and self.anchor_ratio > 0:
            consumed_batches, batch_count = self._anchor_batch_count(global_batch_index)
            total_anchor_examples = batch_count * self.anchor_batch_size
            local_start = total_anchor_examples * position_in_batch // self.target_batch_size
            local_end = total_anchor_examples * (position_in_batch + 1) // self.target_batch_size
            anchor_sequence = self._anchor_indices()
            sequence_start = consumed_batches * self.anchor_batch_size
            for offset in range(local_start, local_end):
                anchor_index = anchor_sequence[sequence_start + offset]
                anchors.append(self.anchor_dataset[anchor_index])
        return {"target": self.target_dataset[target_index], "anchor": anchors}

    def state_dict(self) -> dict[str, Any]:
        target_permutation = self._permutation(len(self.target_dataset), anchor=False)
        anchor_sequence = self._anchor_indices() if self.anchor_dataset is not None else None

        def permutation_hash(permutation: list[int] | None) -> str | None:
            if permutation is None:
                return None
            tensor = torch.tensor(permutation, dtype=torch.int64)
            return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()

        return {
            "target_fingerprint": self.target_fingerprint,
            "anchor_fingerprint": self.anchor_fingerprint,
            "epoch": self.epoch,
            "epoch_seed": self.seed + self.epoch * 10_000_019,
            "shuffle_seed": self.seed,
            "target_permutation_sha256": permutation_hash(target_permutation),
            "anchor_permutation_sha256": permutation_hash(anchor_sequence),
            "anchor_sampling_strategy": self.anchor_sampling_strategy.value,
            "target_batch_size": self.target_batch_size,
            "anchor_batch_size": self.anchor_batch_size,
            "anchor_batches_per_target": float(self.anchor_ratio),
            "world_size": self.world_size,
            "per_rank_steps": self.per_rank_steps,
        }

    def validate_resume_state(self, state: dict[str, Any]) -> None:
        checkpoint_epoch = int(state.get("epoch", 0))
        self.set_epoch(checkpoint_epoch)
        expected = self.state_dict()
        immutable_keys = set(expected) - {"epoch"}
        mismatches = {
            key: (state.get(key), expected[key])
            for key in immutable_keys
            if state.get(key) != expected[key]
        }
        if mismatches:
            details = ", ".join(
                f"{key}: checkpoint={old!r}, current={new!r}"
                for key, (old, new) in sorted(mismatches.items())
            )
            raise ValueError(f"dual-stream resume state mismatch: {details}")


class EpochAwareSequentialSampler(Sampler[int]):
    """Sequential virtual indices with an epoch hook consumed by Accelerate."""

    def __init__(self, dataset: DualStreamDataset) -> None:
        self.dataset = dataset

    def __iter__(self):
        return iter(range(len(self.dataset)))

    def __len__(self) -> int:
        return len(self.dataset)

    def set_epoch(self, epoch: int) -> None:
        self.dataset.set_epoch(epoch)
