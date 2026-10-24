"""Exact and approximate duplicate audits across experiment data roles."""

from __future__ import annotations

import hashlib
import json
import os
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

from .records import LogicalExample


@dataclass(frozen=True)
class LeakageAuditConfig:
    near_duplicate_threshold: float = 0.85
    shingle_size: int = 5
    signature_size: int = 16
    maximum_signature_bucket: int = 2_000
    maximum_reported_pairs: int = 1_000

    def validate(self) -> None:
        if not 0 < self.near_duplicate_threshold <= 1:
            raise ValueError("near_duplicate_threshold must be in (0, 1]")
        if self.shingle_size <= 0 or self.signature_size <= 0:
            raise ValueError("shingle_size and signature_size must be positive")
        if self.maximum_signature_bucket < 2 or self.maximum_reported_pairs <= 0:
            raise ValueError("leakage-audit limits must be positive")


def normalize_for_deduplication(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(normalized.split())


def _content(example: LogicalExample) -> str:
    return normalize_for_deduplication(f"{example.prompt}\n{example.response}")


def _shingles(text: str, size: int) -> frozenset[int]:
    if len(text) <= size:
        pieces = (text,)
    else:
        pieces = (text[index : index + size] for index in range(len(text) - size + 1))
    return frozenset(
        int.from_bytes(hashlib.blake2b(piece.encode(), digest_size=8).digest(), "big")
        for piece in pieces
    )


def _sample_record(role: str, example: LogicalExample, exact_hash: str) -> dict[str, Any]:
    return {
        "role": role,
        "sample_id": example.sample_id,
        "dataset_name": example.dataset_name,
        "split": example.split,
        "exact_normalized_sha256": exact_hash,
    }


def audit_dataset_leakage(
    datasets: Mapping[str, Sequence[LogicalExample]],
    config: LeakageAuditConfig | None = None,
) -> dict[str, Any]:
    config = config or LeakageAuditConfig()
    config.validate()
    if not datasets:
        raise ValueError("leakage audit requires at least one dataset role")

    indexed: list[tuple[str, LogicalExample, str, frozenset[int]]] = []
    exact_groups: defaultdict[str, list[int]] = defaultdict(list)
    sample_id_groups: defaultdict[str, list[int]] = defaultdict(list)
    role_counts: Counter[str] = Counter()
    for role in sorted(datasets):
        examples = datasets[role]
        if not examples:
            raise ValueError(f"leakage audit role '{role}' is empty")
        for example in examples:
            normalized = _content(example)
            exact_hash = hashlib.sha256(normalized.encode()).hexdigest()
            row_index = len(indexed)
            indexed.append((role, example, exact_hash, _shingles(normalized, config.shingle_size)))
            exact_groups[exact_hash].append(row_index)
            sample_id_groups[example.sample_id].append(row_index)
            role_counts[role] += 1

    exact_duplicate_groups: list[dict[str, Any]] = []
    exact_pairs: set[tuple[int, int]] = set()
    for exact_hash, indices in sorted(exact_groups.items()):
        if len(indices) < 2:
            continue
        exact_pairs.update(tuple(sorted(pair)) for pair in combinations(indices, 2))
        exact_duplicate_groups.append(
            {
                "exact_normalized_sha256": exact_hash,
                "samples": [
                    _sample_record(indexed[index][0], indexed[index][1], exact_hash)
                    for index in indices
                ],
            }
        )

    sample_id_collisions = [
        {
            "sample_id": sample_id,
            "samples": [
                _sample_record(indexed[index][0], indexed[index][1], indexed[index][2])
                for index in indices
            ],
        }
        for sample_id, indices in sorted(sample_id_groups.items())
        if len(indices) > 1
    ]

    signature_buckets: defaultdict[int, list[int]] = defaultdict(list)
    for index, (_, _, _, shingles) in enumerate(indexed):
        for signature in sorted(shingles)[: config.signature_size]:
            signature_buckets[signature].append(index)
    candidates: set[tuple[int, int]] = set()
    oversized_buckets = 0
    for indices in signature_buckets.values():
        if len(indices) > config.maximum_signature_bucket:
            oversized_buckets += 1
            continue
        candidates.update(tuple(sorted(pair)) for pair in combinations(indices, 2))

    near_duplicate_pairs: list[dict[str, Any]] = []
    near_pair_count = 0
    overlap_counts: Counter[str] = Counter()
    for left_index, right_index in sorted(candidates):
        if (left_index, right_index) in exact_pairs:
            left_role, right_role = indexed[left_index][0], indexed[right_index][0]
            overlap_counts["::".join(sorted((left_role, right_role)))] += 1
            continue
        left_role, left_example, left_hash, left_shingles = indexed[left_index]
        right_role, right_example, right_hash, right_shingles = indexed[right_index]
        union = len(left_shingles | right_shingles)
        similarity = len(left_shingles & right_shingles) / union if union else 1.0
        if similarity < config.near_duplicate_threshold:
            continue
        near_pair_count += 1
        overlap_counts["::".join(sorted((left_role, right_role)))] += 1
        if len(near_duplicate_pairs) < config.maximum_reported_pairs:
            near_duplicate_pairs.append(
                {
                    "similarity": similarity,
                    "left": _sample_record(left_role, left_example, left_hash),
                    "right": _sample_record(right_role, right_example, right_hash),
                }
            )

    exact_pair_count = len(exact_pairs)
    report_truncated = near_pair_count > len(near_duplicate_pairs)
    complete = oversized_buckets == 0
    return {
        "schema_version": 1,
        "normalization": "Unicode NFKC + casefold + whitespace collapse",
        "near_duplicate_method": {
            "shingle_type": "character",
            "shingle_size": config.shingle_size,
            "candidate_index": "bottom-k hashed-shingle LSH",
            "signature_size": config.signature_size,
            "jaccard_threshold": config.near_duplicate_threshold,
        },
        "role_counts": dict(sorted(role_counts.items())),
        "sample_count": len(indexed),
        "sample_id_collisions": sample_id_collisions,
        "exact_duplicate_group_count": len(exact_duplicate_groups),
        "exact_duplicate_pair_count": exact_pair_count,
        "exact_duplicate_groups": exact_duplicate_groups,
        "near_duplicate_pair_count": near_pair_count,
        "near_duplicate_pairs": near_duplicate_pairs,
        "overlap_pair_counts": dict(sorted(overlap_counts.items())),
        "report_truncated": report_truncated,
        "oversized_lsh_buckets_skipped": oversized_buckets,
        "complete": complete,
        "passed": (
            complete and not sample_id_collisions and exact_pair_count == 0 and near_pair_count == 0
        ),
    }


def write_leakage_report(report: dict[str, Any], output_path: str | Path) -> Path:
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
    return destination
