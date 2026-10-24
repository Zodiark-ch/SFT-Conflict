"""Local run tracking and reproducibility metadata."""

from .artifacts import RunPaths, prepare_run
from .capacity import CapacityReport, check_capacity
from .hf_cache_manifest import build_hf_cache_manifest, write_hf_cache_manifest
from .jsonl import JsonlMetricsCallback, MemoryDiagnosticsCallback
from .metadata import collect_run_metadata, write_json_atomic
from .preflight import build_preflight_preview, render_preflight_preview
from .reference_snapshots import inventory_snapshot, write_reference_manifests
from .tensor_preview import (
    build_random_logical_tensor_preview,
    build_random_tensor_preview,
    render_tensor_preview,
)

__all__ = [
    "CapacityReport",
    "JsonlMetricsCallback",
    "MemoryDiagnosticsCallback",
    "RunPaths",
    "check_capacity",
    "build_preflight_preview",
    "build_hf_cache_manifest",
    "build_random_tensor_preview",
    "build_random_logical_tensor_preview",
    "collect_run_metadata",
    "inventory_snapshot",
    "prepare_run",
    "render_preflight_preview",
    "render_tensor_preview",
    "write_reference_manifests",
    "write_json_atomic",
    "write_hf_cache_manifest",
]
