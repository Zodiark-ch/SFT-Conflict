"""Deterministic artifact layout and run naming."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sft_conflict.config import ExperimentConfig
from sft_conflict.models import ModelAudit

from .capacity import CapacityReport, check_capacity


@dataclass(frozen=True)
class RunPaths:
    run_id: str
    artifact_root: Path
    checkpoint_dir: Path
    result_dir: Path
    tensorboard_dir: Path
    capacity: CapacityReport


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-.").lower()
    return slug or "run"


def _generated_run_id(config: ExperimentConfig, audit: ModelAudit) -> str:
    state: dict[str, Any] = {
        "model": config.model.model_id,
        "revision": config.model.revision,
        "mode": config.model.training_mode.value,
        "method": config.method.name.value,
        "trainable_names": audit.trainable_parameter_names,
        "seed": config.training.seed,
    }
    digest = hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()[:8]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    model_name = _slug(config.model.model_id.rsplit("/", maxsplit=1)[-1])
    return _slug(
        f"{model_name}-{config.model.training_mode.value}-{config.method.name.value}"
        f"-s{config.training.seed}-{timestamp}-{digest}"
    )


def prepare_run(
    config: ExperimentConfig,
    audit: ModelAudit,
    *,
    allow_existing: bool = False,
) -> RunPaths:
    capacity = check_capacity(
        config.paths.artifact_root,
        estimated_run_bytes=config.paths.estimated_run_bytes,
        minimum_free_bytes=config.paths.minimum_free_bytes,
        free_space_multiplier=config.paths.free_space_multiplier,
    )
    artifact_root = Path(config.paths.artifact_root).expanduser().resolve()
    run_id = (
        _slug(config.paths.run_name) if config.paths.run_name else _generated_run_id(config, audit)
    )
    checkpoint_dir = (
        Path(config.paths.output_dir).expanduser().resolve()
        if config.paths.output_dir
        else artifact_root / "checkpoints" / run_id
    )
    result_dir = artifact_root / "results" / run_id
    tensorboard_dir = result_dir / "tensorboard"
    if not allow_existing and (
        (checkpoint_dir.exists() and any(checkpoint_dir.iterdir()))
        or (result_dir.exists() and any(result_dir.iterdir()))
    ):
        raise FileExistsError(
            f"run '{run_id}' already has artifacts; choose a new paths.run_name "
            "or resume explicitly"
        )
    for path in (checkpoint_dir, result_dir, tensorboard_dir):
        path.mkdir(parents=True, exist_ok=True)
    config.paths.run_name = run_id
    config.paths.output_dir = str(checkpoint_dir)
    return RunPaths(
        run_id=run_id,
        artifact_root=artifact_root,
        checkpoint_dir=checkpoint_dir,
        result_dir=result_dir,
        tensorboard_dir=tensorboard_dir,
        capacity=capacity,
    )
