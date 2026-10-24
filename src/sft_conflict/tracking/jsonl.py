"""Append-only local metric logging independent of optional services."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from transformers import TrainerCallback


class JsonlMetricsCallback(TrainerCallback):
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.started_at = time.monotonic()

    def on_train_begin(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero or logs is None:
            return control
        record = {
            "step": state.global_step,
            "epoch": state.epoch,
            "wall_time_seconds": time.monotonic() - self.started_at,
            **logs,
        }
        if torch.cuda.is_available():
            record["resources/peak_gpu_memory_bytes"] = torch.cuda.max_memory_allocated()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, default=float) + "\n")
        return control


def tensor_bytes_by_dtype(values: Iterable[Any]) -> dict[str, int]:
    totals: defaultdict[str, int] = defaultdict(int)
    seen: set[int] = set()

    def visit(value: Any) -> None:
        if isinstance(value, torch.Tensor):
            if id(value) not in seen:
                seen.add(id(value))
                totals[str(value.dtype)] += value.numel() * value.element_size()
        elif isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, list | tuple):
            for child in value:
                visit(child)

    for value in values:
        visit(value)
    return dict(sorted(totals.items()))


class MemoryDiagnosticsCallback(TrainerCallback):
    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)

    def on_step_end(self, args, state, control, model=None, optimizer=None, **kwargs):
        if model is None or optimizer is None:
            return control
        parameters = list(model.parameters())
        underlying_optimizer = getattr(optimizer, "optimizer", optimizer)
        record = {
            "step": state.global_step,
            "rank": args.process_index,
            "optimizer_class": type(optimizer).__name__,
            "underlying_optimizer_class": type(underlying_optimizer).__name__,
            "optimizer_foreach": underlying_optimizer.defaults.get("foreach"),
            "optimizer_fused": underlying_optimizer.defaults.get("fused"),
            "parameter_bytes_by_dtype": tensor_bytes_by_dtype(parameters),
            "gradient_bytes_by_dtype": tensor_bytes_by_dtype(
                parameter.grad for parameter in parameters if parameter.grad is not None
            ),
            "optimizer_state_bytes_by_dtype": tensor_bytes_by_dtype(optimizer.state.values()),
        }
        if torch.cuda.is_available():
            record.update(
                {
                    "cuda_allocated_bytes": torch.cuda.memory_allocated(),
                    "cuda_reserved_bytes": torch.cuda.memory_reserved(),
                    "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                }
            )
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"memory.rank-{args.process_index}.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        return control
