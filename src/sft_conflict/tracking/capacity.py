"""Disk-capacity gate for external training artifacts."""

from __future__ import annotations

import math
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CapacityReport:
    checked_path: str
    total_bytes: int
    used_bytes: int
    free_bytes: int
    required_free_bytes: int
    estimated_run_bytes: int


def _existing_parent(path: Path) -> Path:
    candidate = path
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    if not candidate.exists():
        raise FileNotFoundError(f"no existing parent found for artifact path {path}")
    return candidate


def check_capacity(
    artifact_root: str | Path,
    *,
    estimated_run_bytes: int,
    minimum_free_bytes: int,
    free_space_multiplier: float,
) -> CapacityReport:
    root = Path(artifact_root).expanduser().resolve()
    checked_path = _existing_parent(root)
    usage = shutil.disk_usage(checked_path)
    required = max(
        int(minimum_free_bytes),
        math.ceil(estimated_run_bytes * free_space_multiplier),
    )
    report = CapacityReport(
        checked_path=str(checked_path),
        total_bytes=usage.total,
        used_bytes=usage.used,
        free_bytes=usage.free,
        required_free_bytes=required,
        estimated_run_bytes=estimated_run_bytes,
    )
    if usage.free < required:
        raise RuntimeError(
            f"artifact capacity gate failed at {checked_path}: "
            f"free={usage.free} bytes, required={required} bytes"
        )
    return report
