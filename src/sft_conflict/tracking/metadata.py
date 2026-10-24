"""Machine-readable provenance for every run."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict
from enum import Enum
from pathlib import Path
from typing import Any

import torch

from sft_conflict.config import ExperimentConfig
from sft_conflict.models import ModelAudit

from .artifacts import RunPaths


def _git_metadata(repository: Path) -> dict[str, Any]:
    def run(*arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=False,
            capture_output=True,
            text=True,
        )

    revision = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {
        "commit": revision.stdout.strip() if revision.returncode == 0 else None,
        "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
    }


def _package_versions() -> dict[str, str | None]:
    names = ["torch", "transformers", "accelerate", "peft", "datasets", "huggingface-hub"]
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def collect_run_metadata(
    config: ExperimentConfig,
    audit: ModelAudit,
    paths: RunPaths,
    *,
    target_fingerprint: str,
    anchor_fingerprint: str | None,
) -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[3]
    return {
        "schema_version": 1,
        "run_id": paths.run_id,
        "git": _git_metadata(repository),
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": _package_versions(),
            "torch_cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_devices": [
                torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
            ],
            "distributed_strategy": config.training.distributed_strategy,
            "world_size": int(os.environ.get("WORLD_SIZE", "1")),
            "rank": int(os.environ.get("RANK", "0")),
            "local_rank": int(os.environ.get("LOCAL_RANK", "0")),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "offline_environment": {
            key: os.environ.get(key)
            for key in (
                "HF_HOME",
                "HF_HUB_CACHE",
                "HF_DATASETS_CACHE",
                "HF_HUB_OFFLINE",
                "HF_DATASETS_OFFLINE",
                "TRANSFORMERS_OFFLINE",
            )
        },
        "model_audit": asdict(audit),
        "data": {
            "target_fingerprint": target_fingerprint,
            "anchor_fingerprint": anchor_fingerprint,
        },
        "capacity": asdict(paths.capacity),
        "config": asdict(config),
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__} to JSON")


def write_json_atomic(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
