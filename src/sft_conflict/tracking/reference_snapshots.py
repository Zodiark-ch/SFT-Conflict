"""Deterministic content inventories for local reference-code snapshots."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

IGNORED_DIRECTORY_NAMES = frozenset(
    {".git", ".mypy_cache", ".pytest_cache", ".ruff_cache", "__pycache__"}
)
IGNORED_FILE_SUFFIXES = frozenset({".pyc", ".pyo"})
LICENSE_PREFIXES = ("LICENSE", "NOTICE", "COPYING")
APACHE_MARKERS = (b"Apache License", b"SPDX-License-Identifier: Apache-2.0")


def _sha256_file(path: Path) -> tuple[str, int, bytes]:
    digest = hashlib.sha256()
    size = 0
    prefix = b""
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            if len(prefix) < 32 * 1024:
                prefix += chunk[: 32 * 1024 - len(prefix)]
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size, prefix


def inventory_snapshot(snapshot_root: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(snapshot_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"reference snapshot is not a directory: {root}")

    paths: list[Path] = []
    for current_root, directory_names, file_names in os.walk(root):
        directory_names[:] = sorted(
            name for name in directory_names if name not in IGNORED_DIRECTORY_NAMES
        )
        for file_name in sorted(file_names):
            path = Path(current_root) / file_name
            if path.suffix in IGNORED_FILE_SUFFIXES or not path.is_file():
                continue
            paths.append(path)
    paths.sort(key=lambda path: path.relative_to(root).as_posix())

    aggregate = hashlib.sha256()
    files: list[dict[str, Any]] = []
    license_files: list[dict[str, Any]] = []
    apache_header_files: list[str] = []
    total_bytes = 0
    for path in paths:
        relative_path = path.relative_to(root).as_posix()
        file_hash, size, prefix = _sha256_file(path)
        total_bytes += size
        aggregate.update(relative_path.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(bytes.fromhex(file_hash))
        aggregate.update(b"\0")
        record = {"path": relative_path, "size_bytes": size, "sha256": file_hash}
        files.append(record)
        if path.name.upper().startswith(LICENSE_PREFIXES):
            license_files.append(record)
        if any(marker in prefix for marker in APACHE_MARKERS):
            apache_header_files.append(relative_path)

    snapshot_hash = aggregate.hexdigest()
    summary = {
        "name": root.name,
        "revision_kind": "content_snapshot",
        "upstream_url": None,
        "upstream_revision": None,
        "snapshot_sha256": snapshot_hash,
        "file_count": len(files),
        "total_bytes": total_bytes,
        "license_files": license_files,
        "notice_files": [
            record
            for record in license_files
            if Path(record["path"]).name.upper().startswith("NOTICE")
        ],
        "apache_header_file_count": len(apache_header_files),
    }
    manifest = {
        "schema_version": 1,
        "snapshot": summary,
        "files": files,
        "apache_header_files": apache_header_files,
    }
    return summary, manifest


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_reference_manifests(
    reference_root: str | Path,
    summary_path: str | Path,
    manifest_directory: str | Path,
) -> dict[str, Any]:
    root = Path(reference_root).expanduser().resolve()
    snapshots = sorted(path for path in root.iterdir() if path.is_dir())
    if not snapshots:
        raise ValueError(f"no reference snapshots found under {root}")
    summaries: list[dict[str, Any]] = []
    manifest_root = Path(manifest_directory).expanduser().resolve()
    for snapshot in snapshots:
        summary, manifest = inventory_snapshot(snapshot)
        detailed_path = manifest_root / f"{snapshot.name}.manifest.json"
        _write_json(detailed_path, manifest)
        summary["detailed_manifest"] = str(detailed_path)
        summaries.append(summary)
    output = {
        "schema_version": 1,
        "hash_algorithm": "sha256(path + NUL + sha256(file) + NUL, sorted by path)",
        "ignored_directories": sorted(IGNORED_DIRECTORY_NAMES),
        "ignored_file_suffixes": sorted(IGNORED_FILE_SUFFIXES),
        "snapshots": summaries,
    }
    _write_json(Path(summary_path).expanduser().resolve(), output)
    return output
