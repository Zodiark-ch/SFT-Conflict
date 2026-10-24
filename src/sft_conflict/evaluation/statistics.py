"""Paired statistics for task-native per-sample correctness results."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from typing import Any


def _correctness_by_id(samples: Sequence[dict[str, Any]]) -> dict[str, bool]:
    indexed: dict[str, bool] = {}
    for sample in samples:
        sample_id = str(sample["sample_id"])
        if sample_id in indexed:
            raise ValueError(f"duplicate evaluation sample_id: {sample_id}")
        indexed[sample_id] = bool(sample["correct"])
    if not indexed:
        raise ValueError("paired comparison requires non-empty sample results")
    return indexed


def _exact_mcnemar_p_value(baseline_wins: int, candidate_wins: int) -> float:
    discordant = baseline_wins + candidate_wins
    if discordant == 0:
        return 1.0
    tail = min(baseline_wins, candidate_wins)
    probability = sum(math.comb(discordant, value) for value in range(tail + 1)) / (2**discordant)
    return min(1.0, 2.0 * probability)


def paired_correctness_comparison(
    baseline_samples: Sequence[dict[str, Any]],
    candidate_samples: Sequence[dict[str, Any]],
    *,
    bootstrap_resamples: int = 10_000,
    seed: int = 42,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    if bootstrap_resamples <= 0:
        raise ValueError("bootstrap_resamples must be positive")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between zero and one")
    baseline = _correctness_by_id(baseline_samples)
    candidate = _correctness_by_id(candidate_samples)
    if baseline.keys() != candidate.keys():
        missing_from_candidate = sorted(baseline.keys() - candidate.keys())
        missing_from_baseline = sorted(candidate.keys() - baseline.keys())
        raise ValueError(
            "paired evaluation sample IDs differ: "
            f"missing_from_candidate={missing_from_candidate[:5]}, "
            f"missing_from_baseline={missing_from_baseline[:5]}"
        )

    sample_ids = sorted(baseline)
    differences = [int(candidate[key]) - int(baseline[key]) for key in sample_ids]
    baseline_wins = sum(value == -1 for value in differences)
    candidate_wins = sum(value == 1 for value in differences)
    ties = len(differences) - baseline_wins - candidate_wins
    observed = sum(differences) / len(differences)

    generator = random.Random(seed)
    bootstrapped = sorted(
        sum(differences[generator.randrange(len(differences))] for _ in differences)
        / len(differences)
        for _ in range(bootstrap_resamples)
    )
    tail = (1.0 - confidence_level) / 2.0
    lower_index = min(int(tail * bootstrap_resamples), bootstrap_resamples - 1)
    upper_index = min(int((1.0 - tail) * bootstrap_resamples), bootstrap_resamples - 1)
    return {
        "schema_version": 1,
        "sample_count": len(sample_ids),
        "candidate_minus_baseline": observed,
        "bootstrap": {
            "seed": seed,
            "resamples": bootstrap_resamples,
            "confidence_level": confidence_level,
            "lower": bootstrapped[lower_index],
            "upper": bootstrapped[upper_index],
        },
        "mcnemar": {
            "baseline_wins": baseline_wins,
            "candidate_wins": candidate_wins,
            "ties": ties,
            "exact_two_sided_p_value": _exact_mcnemar_p_value(
                baseline_wins,
                candidate_wins,
            ),
        },
    }
