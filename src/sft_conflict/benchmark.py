"""Reproducible M2 benchmark planning, resumption, and aggregation."""

from __future__ import annotations

import csv
import json
import math
import os
import statistics
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DEFAULT_GENERAL_TASKS = (
    "boolq",
    "rte",
    "hellaswag",
    "winogrande",
    "arc_challenge",
    "arc_easy",
    "openbookqa",
    "piqa",
    "truthfulqa_mc2",
)

DEFAULT_LLM_EVALUATION_TASKS = (
    "boolq",
    "rte",
    "hellaswag",
    "winogrande",
    "arc_easy",
    "arc_challenge",
    "openbookqa",
    "piqa",
    "truthfulqa_mc2",
    "commonsense_qa",
    "copa",
    "race",
    "mmlu",
    "gsm8k",
    "bbh",
    "ifeval",
    "lambada_openai",
)


@dataclass(frozen=True)
class BenchmarkCondition:
    name: str
    training_required: bool
    training_mode: str | None
    method: str
    requires_anchor: bool


BENCHMARK_CONDITIONS = (
    BenchmarkCondition("base", False, None, "target_only", False),
    BenchmarkCondition("full_ft_target_only", True, "full_ft", "target_only", False),
    BenchmarkCondition(
        "body_ft_frozen_head_target_only",
        True,
        "body_ft_frozen_head",
        "target_only",
        False,
    ),
    BenchmarkCondition("lora_target_only", True, "lora", "target_only", False),
    BenchmarkCondition(
        "target_plus_anchor_ce",
        True,
        "body_ft_frozen_head",
        "target_plus_anchor_ce",
        True,
    ),
    BenchmarkCondition(
        "target_plus_anchor_kl",
        True,
        "body_ft_frozen_head",
        "target_plus_anchor_kl",
        True,
    ),
)


@dataclass(frozen=True)
class BenchmarkRun:
    run_id: str
    condition: str
    seed: int
    model: str
    target_dataset: str
    anchor_dataset: str
    training_required: bool
    training_mode: str | None
    method: str
    general_tasks: tuple[str, ...]


def build_benchmark_plan(
    *,
    model: str,
    target_dataset: str,
    anchor_dataset: str,
    seeds: Sequence[int],
    formal: bool,
    general_tasks: Sequence[str] = DEFAULT_GENERAL_TASKS,
    name_prefix: str = "m2",
) -> dict[str, Any]:
    unique_seeds = tuple(dict.fromkeys(int(seed) for seed in seeds))
    if not unique_seeds:
        raise ValueError("benchmark plan requires at least one seed")
    if formal and len(unique_seeds) < 3:
        raise ValueError("formal benchmark plans require at least three unique seeds")
    if not general_tasks:
        raise ValueError("benchmark plan requires at least one general evaluation task")
    if anchor_dataset == "none":
        raise ValueError("the complete benchmark matrix requires an Anchor dataset")
    runs = [
        BenchmarkRun(
            run_id=f"{name_prefix}-{condition.name}-seed-{seed}",
            condition=condition.name,
            seed=seed,
            model=model,
            target_dataset=target_dataset,
            anchor_dataset=anchor_dataset if condition.requires_anchor else "none",
            training_required=condition.training_required,
            training_mode=condition.training_mode,
            method=condition.method,
            general_tasks=tuple(general_tasks),
        )
        for seed in unique_seeds
        for condition in BENCHMARK_CONDITIONS
    ]
    return {
        "schema_version": 1,
        "formal": formal,
        "seeds": list(unique_seeds),
        "conditions": [condition.name for condition in BENCHMARK_CONDITIONS],
        "general_tasks": list(general_tasks),
        "runs": [asdict(run) for run in runs],
    }


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_benchmark_plan(plan: dict[str, Any], output_path: str | Path) -> Path:
    destination = Path(output_path).expanduser().resolve()
    _atomic_json(destination, plan)
    return destination


class BenchmarkOrchestrator:
    """Execute plan entries once and atomically checkpoint each completed result."""

    def __init__(self, plan: dict[str, Any], state_path: str | Path) -> None:
        self.plan = plan
        self.state_path = Path(state_path).expanduser().resolve()

    def run(
        self,
        execute: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> list[dict[str, Any]]:
        state = {"schema_version": 1, "completed": {}}
        if self.state_path.is_file():
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
        completed = state.setdefault("completed", {})
        for run in self.plan["runs"]:
            run_id = run["run_id"]
            if run_id in completed:
                continue
            result = execute(run)
            completed[run_id] = {"run": run, "result": result}
            _atomic_json(self.state_path, state)
        return [completed[run["run_id"]] for run in self.plan["runs"]]


def aggregate_benchmark_results(
    observations: Sequence[dict[str, Any]],
    *,
    target_metric_name: str,
    general_metric_name: str,
    minimum_target_gain: float = 0.02,
) -> dict[str, Any]:
    if not observations:
        raise ValueError("benchmark aggregation requires observations")
    if minimum_target_gain <= 0:
        raise ValueError("minimum_target_gain must be positive")
    indexed: dict[tuple[str, int], dict[str, Any]] = {}
    for observation in observations:
        key = (str(observation["condition"]), int(observation["seed"]))
        if key in indexed:
            raise ValueError(f"duplicate benchmark observation: {key}")
        target_score = float(observation["target_score"])
        general_score = float(observation["general_score"])
        if not math.isfinite(target_score) or not math.isfinite(general_score):
            raise ValueError(f"benchmark observation has non-finite metrics: {key}")
        indexed[key] = dict(observation)

    enriched: list[dict[str, Any]] = []
    for (_condition, seed), observation in sorted(indexed.items()):
        base = indexed.get(("base", seed))
        if base is None:
            raise ValueError(f"benchmark seed {seed} has no paired base observation")
        enriched.append(
            {
                **observation,
                "target_gain": float(observation["target_score"]) - float(base["target_score"]),
                "general_capability_loss": float(base["general_score"])
                - float(observation["general_score"]),
            }
        )

    summaries: list[dict[str, Any]] = []
    for condition in sorted({row["condition"] for row in enriched}):
        rows = [row for row in enriched if row["condition"] == condition]
        summaries.append(
            {
                "condition": condition,
                "seed_count": len(rows),
                "target_score_mean": statistics.fmean(row["target_score"] for row in rows),
                "target_score_std": (
                    statistics.stdev(row["target_score"] for row in rows) if len(rows) > 1 else 0.0
                ),
                "general_score_mean": statistics.fmean(row["general_score"] for row in rows),
                "general_score_std": (
                    statistics.stdev(row["general_score"] for row in rows) if len(rows) > 1 else 0.0
                ),
                "target_gain_mean": statistics.fmean(row["target_gain"] for row in rows),
                "general_capability_loss_mean": statistics.fmean(
                    row["general_capability_loss"] for row in rows
                ),
            }
        )
    non_base = [row for row in enriched if row["condition"] != "base"]
    best = max(non_base, key=lambda row: row["target_gain"], default=None)
    return {
        "schema_version": 1,
        "target_metric_name": target_metric_name,
        "general_metric_name": general_metric_name,
        "observations": enriched,
        "summary": summaries,
        "acceptance": {
            "minimum_target_gain": minimum_target_gain,
            "best_condition": best["condition"] if best is not None else None,
            "best_target_gain": best["target_gain"] if best is not None else None,
            "target_learning_demonstrated": (
                best is not None and best["target_gain"] >= minimum_target_gain
            ),
        },
    }


def write_benchmark_results(result: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / "benchmark-results.json"
    csv_path = destination / "benchmark-table.csv"
    _atomic_json(json_path, result)
    fieldnames = (
        "condition",
        "seed_count",
        "target_score_mean",
        "target_score_std",
        "general_score_mean",
        "general_score_std",
        "target_gain_mean",
        "general_capability_loss_mean",
    )
    temporary = csv_path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result["summary"])
    os.replace(temporary, csv_path)
    return {"json": str(json_path), "csv": str(csv_path)}
