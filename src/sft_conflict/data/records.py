"""Validated in-memory token records with stable fingerprints."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from torch.utils.data import Dataset
from tqdm.auto import tqdm


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def make_sample_id(namespace: str, row_index: int, raw_sample: Mapping[str, Any]) -> str:
    payload = f"{namespace}\n{row_index}\n{_canonical_json(raw_sample)}".encode()
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class LogicalExample:
    sample_id: str
    dataset_name: str
    dataset_revision: str | None
    task_name: str
    split: str
    prompt: str
    response: str
    loss_mask: str
    source_metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LogicalDataset(Sequence[LogicalExample]):
    def __init__(
        self,
        examples: Sequence[LogicalExample],
        *,
        namespace: str,
        manifest_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if not examples:
            raise ValueError(f"logical dataset '{namespace}' is empty")
        self.namespace = namespace
        self._examples = tuple(examples)
        sample_ids = [example.sample_id for example in self._examples]
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError(f"logical dataset '{namespace}' has duplicate sample IDs")
        digest = hashlib.sha256()
        digest.update(namespace.encode())
        for example in self._examples:
            digest.update(_canonical_json(example.to_dict()).encode())
        self.fingerprint = digest.hexdigest()
        self.manifest = {
            "schema_version": 1,
            "namespace": namespace,
            "fingerprint": self.fingerprint,
            "sample_count": len(self._examples),
            "sample_ids": sample_ids,
            **dict(manifest_metadata or {}),
        }

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int) -> LogicalExample:
        return self._examples[index]


class TokenizedDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        namespace: str,
        preprocessing_stats: Mapping[str, Any] | None = None,
        manifest_metadata: Mapping[str, Any] | None = None,
        progress_description: str | None = None,
    ) -> None:
        if not records:
            raise ValueError(f"tokenized dataset '{namespace}' is empty")
        self.namespace = namespace
        self.preprocessing_stats = dict(preprocessing_stats or {})
        validation_records = (
            tqdm(
                records,
                total=len(records),
                desc=f"{progress_description}: validating",
                unit="sample",
                dynamic_ncols=True,
            )
            if progress_description is not None
            else records
        )
        self._records = tuple(
            self._validate_record(dict(record), index)
            for index, record in enumerate(validation_records)
        )
        sample_ids = [record["sample_id"] for record in self._records]
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError(f"tokenized dataset '{namespace}' has duplicate sample IDs")
        digest = hashlib.sha256()
        digest.update(namespace.encode())
        fingerprint_records = (
            tqdm(
                self._records,
                total=len(self._records),
                desc=f"{progress_description}: fingerprinting",
                unit="sample",
                dynamic_ncols=True,
            )
            if progress_description is not None
            else self._records
        )
        for record in fingerprint_records:
            digest.update(_canonical_json(record).encode())
        self.fingerprint = digest.hexdigest()
        self.manifest = {
            "schema_version": 1,
            "namespace": namespace,
            "fingerprint": self.fingerprint,
            "sample_count": len(self._records),
            "sample_ids": sample_ids,
            "preprocessing_stats": self.preprocessing_stats,
            **dict(manifest_metadata or {}),
        }

    def _validate_record(self, record: dict[str, Any], index: int) -> dict[str, Any]:
        for key in ("input_ids", "attention_mask", "labels"):
            if key not in record:
                raise ValueError(f"{self.namespace}[{index}] is missing '{key}'")
            if not isinstance(record[key], list):
                raise TypeError(f"{self.namespace}[{index}].{key} must be a list")
        lengths = {len(record[key]) for key in ("input_ids", "attention_mask", "labels")}
        if len(lengths) != 1 or not next(iter(lengths)):
            raise ValueError(
                f"{self.namespace}[{index}] token fields must have one non-zero length"
            )
        if any(value not in {0, 1} for value in record["attention_mask"]):
            raise ValueError(f"{self.namespace}[{index}].attention_mask must be binary")
        if "sample_id" not in record:
            record["sample_id"] = make_sample_id(self.namespace, index, record)
        if not isinstance(record["sample_id"], str) or not record["sample_id"]:
            raise ValueError(f"{self.namespace}[{index}].sample_id must be a non-empty string")
        record.setdefault("source_metadata", {})
        return record

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._records[index]
