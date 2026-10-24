"""Inventory fixed-revision Hugging Face model and dataset caches."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from sft_conflict.quick_run import (
    ANCHOR_DATASET_PRESETS,
    EVALUATION_DATASET_PRESETS,
    MODEL_PRESETS,
    TARGET_DATASET_PRESETS,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path, root: Path) -> dict[str, Any]:
    resolved = path.resolve()
    blob_id = resolved.name if path.is_symlink() else None
    if blob_id is not None and len(blob_id) == 64:
        sha256 = blob_id
        hash_source = "huggingface_lfs_blob_id"
    else:
        sha256 = _sha256(resolved)
        hash_source = "computed_sha256"
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256,
        "hash_source": hash_source,
        "blob_id": blob_id,
    }


def _model_entry(name: str, preset, hub_cache: Path) -> dict[str, Any]:
    if preset.model_id is None:
        return {
            "preset": name,
            "repo_type": "local_generated_model",
            "repo_id": None,
            "revision": preset.revision,
            "cached": True,
            "files": [],
        }
    repository = hub_cache / ("models--" + preset.model_id.replace("/", "--"))
    snapshot = repository / "snapshots" / preset.revision
    files = (
        sorted(path for path in snapshot.iterdir() if path.is_file()) if snapshot.is_dir() else []
    )
    records = [_file_record(path, snapshot) for path in files]
    return {
        "preset": name,
        "repo_type": "model",
        "repo_id": preset.model_id,
        "revision": preset.revision,
        "cached": bool(records),
        "snapshot_path": str(snapshot),
        "file_count": len(records),
        "total_bytes": sum(record["size_bytes"] for record in records),
        "files": records,
    }


def _dataset_entry(name: str, preset, dataset_cache: Path, role: str) -> dict[str, Any]:
    if preset is None:
        return {
            "preset": name,
            "role": role,
            "repo_type": "none",
            "cached": True,
            "files": [],
        }
    if preset.path is None:
        return {
            "preset": name,
            "role": role,
            "repo_type": "local_generated_dataset",
            "repo_id": None,
            "revision": None,
            "cached": True,
            "files": [],
        }
    repository = dataset_cache / preset.path.replace("/", "___")
    revision_directories = (
        sorted(path for path in repository.rglob(preset.revision or "") if path.is_dir())
        if repository.is_dir() and preset.revision
        else []
    )
    files: list[Path] = []
    for revision_directory in revision_directories:
        files.extend(
            path
            for path in revision_directory.rglob("*")
            if path.is_file() and not path.name.endswith(".lock")
        )
    unique_files = sorted(set(files))
    records = [_file_record(path, dataset_cache) for path in unique_files]
    return {
        "preset": name,
        "role": role,
        "repo_type": "dataset",
        "repo_id": preset.path,
        "configuration": preset.name,
        "revision": preset.revision,
        "train_split": preset.train_split,
        "eval_split": preset.eval_split,
        "cached": bool(records),
        "cache_path": str(repository),
        "file_count": len(records),
        "total_bytes": sum(record["size_bytes"] for record in records),
        "files": records,
    }


def build_hf_cache_manifest(artifact_root: str | Path) -> dict[str, Any]:
    root = Path(artifact_root).expanduser().resolve() / "hf_cache"
    hub_cache = root / "hub"
    dataset_cache = root / "datasets"
    target_datasets = [
        _dataset_entry(name, preset, dataset_cache, "target")
        for name, preset in TARGET_DATASET_PRESETS.items()
    ]
    anchor_datasets = [
        _dataset_entry(name, preset, dataset_cache, "anchor")
        for name, preset in ANCHOR_DATASET_PRESETS.items()
    ]
    evaluation_datasets = [
        _dataset_entry(name, preset, dataset_cache, "evaluation")
        for name, preset in EVALUATION_DATASET_PRESETS.items()
    ]
    return {
        "schema_version": 1,
        "artifact_root": str(Path(artifact_root).expanduser().resolve()),
        "models": [_model_entry(name, preset, hub_cache) for name, preset in MODEL_PRESETS.items()],
        "datasets": target_datasets + anchor_datasets + evaluation_datasets,
    }


def write_hf_cache_manifest(artifact_root: str | Path, output_path: str | Path) -> None:
    manifest = build_hf_cache_manifest(artifact_root)
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
