"""CPU-testable building blocks for a health-gated, target-only pilot.

Callers own model loading, training, CLI policy, and rank-zero registry writes.
Compatibility keys must use the *formal source config*, before removing Anchor
data or switching the pilot's method to target_only. Health is a teacher-forced
test on a disjoint subset, not a guarantee of generation or downstream quality.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from enum import Enum
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from uuid import UUID, uuid4

import torch
import torch.distributed as distributed
import torch.nn.functional as functional
from torch.utils.data import Dataset

from sft_conflict.config import ExperimentConfig, LVDResponseMode
from sft_conflict.data.collator import CausalLMCollator, model_tensor_inputs
from sft_conflict.data.reference_cache import ReferenceCache
from sft_conflict.models.unembedding import UnembeddingInputCapture
from sft_conflict.pilot_stability import calibration_normalizer, calibration_stability_decision
from sft_conflict.tracking.metadata import write_json_atomic

PROTOCOL_VERSION = 1
PUBLICATION_VERSION = 2

_PUBLICATION_FIELDS = ("publication_version", "publication_id", "pilot_key", "report_sha256")

_CACHE_FIELDS = ("reference_token_ids", "hidden_states", "target_token_ids")
_COUNTS = (
    "sample_count",
    "supervised_token_count",
    "supervised_correct_count",
    "selected_position_count",
    "cached_position_count",
    "cached_reference_retained_count",
    "cached_label_token_count",
    "cached_label_correct_count",
    "cached_base_correct_count",
    "cached_base_correct_retained_count",
)
_SUMS = (
    "supervised_ce_sum",
    "hidden_norm_sum",
    "reference_hidden_norm_sum",
    "hidden_norm_ratio_sum",
    "relative_displacement_sum",
)
_RATIOS = {
    "supervised_accuracy": ("supervised_correct_count", "supervised_token_count"),
    "supervised_ce": ("supervised_ce_sum", "supervised_token_count"),
    "cached_reference_retention": ("cached_reference_retained_count", "cached_position_count"),
    "cached_label_accuracy": ("cached_label_correct_count", "cached_label_token_count"),
    "cached_base_correct_retention": (
        "cached_base_correct_retained_count",
        "cached_base_correct_count",
    ),
    "h0_cached_label_accuracy": ("cached_base_correct_count", "cached_label_token_count"),
    "hidden_norm": ("hidden_norm_sum", "cached_position_count"),
    "reference_hidden_norm": ("reference_hidden_norm_sum", "cached_position_count"),
    "hidden_norm_ratio": ("hidden_norm_ratio_sum", "cached_position_count"),
    "relative_displacement": ("relative_displacement_sum", "cached_position_count"),
}


def _json_default(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__} in pilot protocol")


def _sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False, default=_json_default
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _fingerprint(dataset: Dataset) -> str:
    value = getattr(dataset, "fingerprint", None)
    if not isinstance(value, str) or not value:
        raise ValueError("pilot Anchor dataset must expose a nonempty fingerprint")
    return value


class IndexedDatasetView(Dataset):
    """Ordered subset with separate cache-binding and stream identities.

    ``fingerprint`` remains the source cache identity. ``stream_fingerprint``
    binds source stream plus ordered indices, including for nested views. The
    caller must not subsequently mutate the source or this view's indices.
    """

    def __init__(self, source: Dataset, indices: Sequence[int]) -> None:
        self.source = source
        self.indices = list(indices)
        if any(type(index) is not int or not 0 <= index < len(source) for index in self.indices):
            raise ValueError("subset indices must be integers within the source dataset")
        self.fingerprint = _fingerprint(source)
        self.stream_fingerprint = _sha256(
            {
                "source_fingerprint": getattr(source, "stream_fingerprint", self.fingerprint),
                "indices": self.indices,
            }
        )
        if hasattr(source, "selector"):
            self.selector = getattr(source, "selector", None)
        self.manifest = {
            **dict(getattr(source, "manifest", {})),
            "subset_count": len(self.indices),
            "indices_sha256": _sha256(self.indices),
            "stream_fingerprint": self.stream_fingerprint,
        }
        self.preprocessing_stats = {
            **dict(getattr(source, "preprocessing_stats", {})),
            "subset_count": len(self.indices),
        }

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Any:
        return self.source[self.indices[index]]


def build_anchor_plan(
    dataset: Dataset,
    *,
    seed: int,
    calibration_samples: int,
    health_samples: int = 128,
    validation_samples: int = 128,
) -> dict[str, Any]:
    """Return disjoint health/validation/calibration/training splits without I/O.

    Each split is ``{"indices": [...], "ids": [...]}``. All requested counts
    must be positive integers, with at least one training row left over.
    Membership is SHA256-ordered by (seed, sample_id), independent of row order.
    ``plan_sha256`` hashes the canonical JSON of every other returned field.
    """
    if type(seed) is not int or seed < 0:
        raise ValueError("pilot plan seed must be a non-negative integer")
    counts = {
        "health": health_samples,
        "validation": validation_samples,
        "calibration": calibration_samples,
    }
    for name, count in counts.items():
        if type(count) is not int or count <= 0:
            raise ValueError(f"{name}_samples must be a positive integer")
    if sum(counts.values()) >= len(dataset):
        raise ValueError("Anchor plan needs enough samples to leave nonempty training data")
    fingerprint = _fingerprint(dataset)
    ids = [dataset[index]["sample_id"] for index in range(len(dataset))]
    if any(not isinstance(sample_id, str) or not sample_id for sample_id in ids):
        raise ValueError("Anchor sample_ids must be nonempty strings")
    if len(set(ids)) != len(ids):
        raise ValueError("Anchor plan rejects duplicate sample_ids")
    indices = sorted(range(len(ids)), key=lambda index: (_sha256([seed, ids[index]]), ids[index]))
    plan: dict[str, Any] = {
        "version": PROTOCOL_VERSION,
        "seed": seed,
        "dataset_fingerprint": fingerprint,
    }
    counts["training"] = len(dataset) - sum(counts.values())
    start = 0
    for name, count in counts.items():
        selected = indices[start : start + count]
        plan[name] = {"indices": selected, "ids": [ids[index] for index in selected]}
        start += count
    plan["plan_sha256"] = _sha256(plan)
    return plan


def _positioned_record(dataset: Dataset, index: int) -> dict[str, Any]:
    record = dict(dataset[index])
    if "selected_anchor_positions" not in record:
        selector = getattr(dataset, "selector", None)
        if selector is None:
            raise ValueError("Anchor health requires selected positions or a dataset.selector")
        selection = selector.select(record)
        record["selected_anchor_positions"] = list(selection.positions)
        record["selected_anchor_target_ids"] = list(selection.target_token_ids)
    return record


def _accumulate_health(
    stats: dict[str, float],
    batch: dict[str, Any],
    logits: torch.Tensor,
    hidden: torch.Tensor,
    cache: ReferenceCache,
    real_rows: int,
) -> None:
    if (
        logits.ndim != 3
        or hidden.ndim != 3
        or (logits.shape[:2] != batch["input_ids"].shape or hidden.shape[:2] != logits.shape[:2])
    ):
        raise ValueError("Anchor health requires full [batch, sequence, vocabulary/hidden] outputs")
    stats["sample_count"] += real_rows
    # Dummy-only rows contribute nothing, including to nonfinite diagnostics.
    stats["nonfinite_observations"] += int(not torch.isfinite(logits[:real_rows]).all())
    stats["nonfinite_observations"] += int(not torch.isfinite(hidden[:real_rows]).all())
    for row in range(real_rows):
        labels = batch["labels"][row, 1:].to(logits.device)
        supervised = labels.ne(-100)
        count = int(supervised.sum().item())
        stats["supervised_token_count"] += count
        if count:
            selected_logits = logits[row, :-1][supervised]
            selected_labels = labels[supervised]
            stats["supervised_correct_count"] += int(
                selected_logits.argmax(dim=-1).eq(selected_labels).sum().item()
            )
            stats["supervised_ce_sum"] += float(
                functional.cross_entropy(selected_logits.float(), selected_labels, reduction="none")
                .double()
                .sum()
                .item()
            )
        mask = batch["selected_anchor_position_mask"][row]
        positions = batch["selected_anchor_positions"][row][mask].tolist()
        targets = batch["selected_anchor_target_ids"][row][mask].tolist()
        if len(set(positions)) != len(positions):
            raise ValueError("duplicate selected Anchor positions")
        stats["selected_position_count"] += len(positions)
        for position, expected_target in zip(positions, targets, strict=True):
            if not 0 <= position < logits.shape[1] or not batch["attention_mask"][row, position]:
                raise ValueError("selected Anchor position is outside the active sequence")
            reference = cache.get_position_fields(batch["sample_ids"][row], position, _CACHE_FIELDS)
            if reference is None:
                continue
            reference_id = int(reference["reference_token_ids"].item())
            target_id = int(reference["target_token_ids"].item())
            if not 0 <= reference_id < logits.shape[-1] or (
                target_id != -100 and not 0 <= target_id < logits.shape[-1]
            ):
                raise ValueError("cached Anchor token ID is outside the model vocabulary")
            if target_id != expected_target:
                raise ValueError("cached Anchor target disagrees with the selected position")
            current = hidden[row, position].double()
            h0 = reference["hidden_states"].to(device=current.device, dtype=torch.float64)
            if h0.shape != current.shape:
                raise ValueError("cached h0 shape does not match the Unembedding input")
            stats["nonfinite_observations"] += int(not torch.isfinite(h0).all())
            current_norm = current.norm()
            reference_norm = h0.norm()
            denominator = reference_norm.clamp_min(1e-12)
            # Use the actual model output, including any post-head transforms.
            # Ordinary torch.argmax tie-breaking is valid; no geometry audit.
            prediction = int(logits[row, position].argmax(dim=-1).item())
            retained = prediction == reference_id
            labeled = target_id != -100
            base_correct = labeled and reference_id == target_id
            stats["cached_position_count"] += 1
            stats["cached_reference_retained_count"] += int(retained)
            stats["cached_label_token_count"] += int(labeled)
            stats["cached_label_correct_count"] += int(labeled and prediction == target_id)
            stats["cached_base_correct_count"] += int(base_correct)
            stats["cached_base_correct_retained_count"] += int(base_correct and retained)
            stats["hidden_norm_sum"] += float(current_norm.item())
            stats["reference_hidden_norm_sum"] += float(reference_norm.item())
            stats["hidden_norm_ratio_sum"] += float((current_norm / denominator).item())
            stats["relative_displacement_sum"] += float(
                ((current - h0).norm() / denominator).item()
            )


@torch.no_grad()
def evaluate_anchor_health(
    model: torch.nn.Module,
    dataset: Dataset,
    cache: ReferenceCache,
    *,
    pad_token_id: int,
    device: torch.device,
    batch_size: int = 1,
) -> dict[str, Any]:
    """Pool teacher-forced and cached-position evidence on ALL distributed ranks.

    Pass the same full subset and batch size on every rank. Rank-strided rows
    are padded with uncounted dummy examples so even empty ranks perform the
    same number of forwards (also for FSDP). Only a packed FP64 statistics vector
    is reduced; no LVD margins, head snapshot, or vocabulary geometry is loaded.
    The model must already be placed/wrapped; this function never moves it.

    Counts/sums are global. Ratios with empty denominators are None. h0 label
    accuracy is an estimate on cached labeled positions, NOT all response tokens.
    Norm ratios and relative displacement are means of per-position quantities
    with denominator max(||h0||, 1e-12). Nonfinite evidence or globally zero cached
    positions raises ValueError after matching forwards and collective reduction.
    """
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("health batch_size must be a positive integer")
    if not len(dataset):
        raise ValueError("Anchor health has no cached positions: the dataset is empty")
    fingerprint = _fingerprint(dataset)
    cache_manifest = dict(cache.manifest)
    if cache_manifest.get("dataset_fingerprint", fingerprint) != fingerprint:
        raise ValueError("Anchor health cache dataset fingerprint mismatch")
    distributed_run = distributed.is_available() and distributed.is_initialized()
    rank = distributed.get_rank() if distributed_run else 0
    world_size = distributed.get_world_size() if distributed_run else 1
    stats = dict.fromkeys((*_COUNTS, *_SUMS, "nonfinite_observations", "invalid_batches"), 0.0)
    collator = CausalLMCollator(pad_token_id, pad_to_multiple_of=None)
    training_states = [(module, module.training) for module in model.modules()]
    local_error = None
    try:
        model.eval()
        for start in range(0, len(dataset), world_size * batch_size):
            indices = range(
                start + rank, min(start + world_size * batch_size, len(dataset)), world_size
            )
            records = [_positioned_record(dataset, index) for index in indices]
            real_rows = len(records)
            if real_rows < batch_size:
                records += [_positioned_record(dataset, 0)] * (batch_size - real_rows)
            batch = collator(records)
            inputs = {
                name: tensor.to(device) for name, tensor in model_tensor_inputs(batch).items()
            }
            with UnembeddingInputCapture(model) as capture:
                outputs = model(**inputs)
            try:
                if capture.hidden_states is None:
                    raise ValueError("model forward did not capture its Unembedding input")
                _accumulate_health(
                    stats, batch, outputs.logits, capture.hidden_states, cache, real_rows
                )
            except (ValueError, TypeError, KeyError, IndexError) as error:
                # Defer data/metric failures until all ranks finish their forwards.
                stats["invalid_batches"] += 1
                local_error = str(error)
            del outputs, capture  # Do not retain the previous batch during the next forward.
    finally:
        model.train(training_states[0][1])
        for module, was_training in training_states:
            module.training = was_training
    reduction_device = device if distributed_run and distributed.get_backend() == "nccl" else "cpu"
    packed = torch.tensor(list(stats.values()), dtype=torch.float64, device=reduction_device)
    if distributed_run:
        distributed.all_reduce(packed, op=distributed.ReduceOp.SUM)
    stats = dict(zip(stats, packed.cpu().tolist(), strict=True))
    if stats["nonfinite_observations"] or not all(math.isfinite(value) for value in stats.values()):
        raise ValueError(
            "Anchor health encountered non-finite logits, hidden states, or statistics"
        )
    if stats["invalid_batches"]:
        raise ValueError(
            f"invalid Anchor health evidence: {local_error or 'failure on another rank'}"
        )
    if not stats["cached_position_count"]:
        raise ValueError("Anchor health found no cached positions globally")
    return {
        "version": PROTOCOL_VERSION,
        "status": "ok",
        "finite": True,
        "dataset_fingerprint": fingerprint,
        "stream_fingerprint": getattr(dataset, "stream_fingerprint", fingerprint),
        "cache_fingerprint": _sha256(cache_manifest),
        **{name: int(stats[name]) for name in _COUNTS},
        **{name: stats[name] for name in _SUMS},
        **{
            name: stats[numerator] / stats[denominator] if stats[denominator] else None
            for name, (numerator, denominator) in _RATIOS.items()
        },
    }


def _finite_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _health_report_errors(report: Mapping[str, Any], label: str) -> list[str]:
    errors = []
    if (
        type(report.get("version")) is not int
        or report["version"] != PROTOCOL_VERSION
        or report.get("status") != "ok"
        or report.get("finite") is not True
    ):
        errors.append(f"{label}: missing/invalid protocol, status, or finite flag")
    for name in ("dataset_fingerprint", "stream_fingerprint", "cache_fingerprint"):
        if not isinstance(report.get(name), str) or not report[name]:
            errors.append(f"{label}: missing {name}")
    for name in (*_COUNTS, *_SUMS):
        value = report.get(name)
        if not _finite_number(value) or value < 0 or (name in _COUNTS and type(value) is not int):
            errors.append(f"{label}: invalid/non-finite {name}")
    if errors:
        return errors
    for name in (
        "sample_count",
        "supervised_token_count",
        "cached_position_count",
        "cached_label_token_count",
        "cached_base_correct_count",
        "reference_hidden_norm_sum",
    ):
        if report[name] <= 0:
            errors.append(f"{label}: insufficient {name}")
    for numerator, denominator in (
        ("supervised_correct_count", "supervised_token_count"),
        ("cached_position_count", "selected_position_count"),
        ("cached_reference_retained_count", "cached_position_count"),
        ("cached_label_token_count", "cached_position_count"),
        ("cached_label_correct_count", "cached_label_token_count"),
        ("cached_base_correct_count", "cached_label_token_count"),
        ("cached_base_correct_retained_count", "cached_base_correct_count"),
        ("cached_base_correct_retained_count", "cached_reference_retained_count"),
        ("cached_base_correct_retained_count", "cached_label_correct_count"),
    ):
        if report[numerator] > report[denominator]:
            errors.append(f"{label}: inconsistent {numerator}/{denominator}")
    for name, (numerator, denominator) in _RATIOS.items():
        value = report.get(name)
        if not report[denominator]:
            if value is not None:
                errors.append(f"{label}: {name} must be None for an empty denominator")
        elif not _finite_number(value) or not math.isclose(
            value, report[numerator] / report[denominator], rel_tol=1e-9, abs_tol=1e-12
        ):
            errors.append(f"{label}: invalid/non-finite or inconsistent {name}")
    return errors


def pilot_health_decision(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    minimum_retention: float = 0.99,
    maximum_accuracy_drop: float = 0.01,
    minimum_relative_displacement: float = 1e-5,
) -> dict[str, Any]:
    """Fail closed on missing, mismatched, empty, nonfinite, or unhealthy evidence.

    Baseline must replay cached argmax AND base-correct labels at >= .999.
    Candidate retention gates only base-correct positions: correcting a wrong
    base prediction in an all_reference cache is not a retention failure.
    Displacement must exceed the baseline's replay noise by the requested floor.
    At least one cached base-correct and one supervised token are required;
    these thresholds are policy checks, not statistical/generalization guarantees.
    The returned decision (including baseline/candidate) is the registry report.
    """
    thresholds = {
        "minimum_retention": minimum_retention,
        "maximum_accuracy_drop": maximum_accuracy_drop,
        "minimum_relative_displacement": minimum_relative_displacement,
        "minimum_baseline_replay_retention": 0.999,
    }
    for name, value in thresholds.items():
        if (
            not _finite_number(value)
            or value < 0
            or (name != "minimum_relative_displacement" and value > 1)
        ):
            raise ValueError(f"invalid pilot health threshold: {name}")
    reasons = []
    for label, report in (("baseline", baseline), ("candidate", candidate)):
        if not isinstance(report, Mapping):
            reasons.append(f"{label}: health report must be a mapping")
        else:
            reasons.extend(_health_report_errors(report, label))
    accuracy_drop = displacement_increase = None
    if not reasons:
        for name in (
            "dataset_fingerprint",
            "stream_fingerprint",
            "cache_fingerprint",
            "sample_count",
            "supervised_token_count",
            "selected_position_count",
            "cached_position_count",
            "cached_label_token_count",
            "cached_base_correct_count",
        ):
            if baseline[name] != candidate[name]:
                reasons.append(f"baseline/candidate {name} mismatch")
        if not math.isclose(
            baseline["reference_hidden_norm_sum"],
            candidate["reference_hidden_norm_sum"],
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            reasons.append("baseline/candidate cached h0 norm estimate mismatch")
        for name in ("cached_reference_retention", "cached_base_correct_retention"):
            if baseline[name] < 0.999:
                reasons.append(f"baseline {name} is below the 0.999 replay requirement")
        if candidate["cached_base_correct_retention"] + 1e-12 < minimum_retention:
            reasons.append("candidate cached base-correct retention is below minimum_retention")
        accuracy_drop = baseline["supervised_accuracy"] - candidate["supervised_accuracy"]
        if accuracy_drop > maximum_accuracy_drop + 1e-12:
            reasons.append("supervised accuracy drop exceeds maximum_accuracy_drop")
        displacement_increase = (
            candidate["relative_displacement"] - baseline["relative_displacement"]
        )
        if displacement_increase + 1e-12 < minimum_relative_displacement:
            reasons.append(
                "relative displacement does not exceed baseline replay noise by the minimum"
            )
    return {
        "version": PROTOCOL_VERSION,
        "accepted": not reasons,
        "reasons": reasons,
        "thresholds": thresholds,
        "accuracy_drop": accuracy_drop,
        "relative_displacement_increase": displacement_increase,
        "baseline": dict(baseline) if isinstance(baseline, Mapping) else baseline,
        "candidate": dict(candidate) if isinstance(candidate, Mapping) else candidate,
    }


def _world_size() -> int:
    if distributed.is_available() and distributed.is_initialized():
        value = distributed.get_world_size()
    else:
        try:
            value = int(os.environ.get("WORLD_SIZE", "1"))
        except ValueError as error:
            raise ValueError("pilot WORLD_SIZE must be a positive integer") from error
    if type(value) is not int or value <= 0:
        raise ValueError("pilot world_size must be a positive integer")
    return value


def pilot_compatibility_key(config: ExperimentConfig) -> str:
    """Hash the formal source configuration, never the stripped pilot config.

    Includes model/LoRA, Target/Anchor streams, optimizer/scheduler/precision,
    method, reference cache paths, all LVD geometry/position/filter settings,
    calibration grouping, and world size (initialized torch.distributed first,
    otherwise WORLD_SIZE, default 1). These bind the exact reserved Anchor plan.
    Excludes paths/logging/saving/resume, lambda policy/value/warmup, pilot path,
    and target gradient ratio: a compatible pilot can be remeasured at a new ratio.
    The core transition fraction binds only three-segment geometry; omitting it
    for legacy modes preserves their pre-extension publication keys exactly.
    Median calibration omits the new statistic field even for three-segment
    geometry, preserving pre-RMS keys; RMS explicitly binds its different policy.
    No config mutation, filesystem access, or unrelated/secret environment input.
    """
    if config.data.anchor is None:
        raise ValueError("pilot compatibility requires the formal source config with data.anchor")
    training = asdict(config.training)
    for name in (
        "logging_steps",
        "save_strategy",
        "save_steps",
        "save_total_limit",
        "recovery_save_steps",
        "report_to",
        "resume_from_checkpoint",
    ):
        training.pop(name, None)
    data = asdict(config.data)
    for name in ("validation", "num_workers"):
        data.pop(name, None)
    lvd = asdict(config.lvd)
    for name in (
        "lambda_mode",
        "fixed_lambda",
        "resolved_lambda",
        "lambda_warmup_steps",
        "calibration_checkpoint",
        "target_gradient_ratio",
    ):
        lvd.pop(name, None)
    if config.lvd.calibration_statistic == "median":
        lvd.pop("calibration_statistic", None)
    if config.lvd.response_mode != LVDResponseMode.THREE_SEGMENT:
        lvd.pop("core_transition_fraction", None)
    method = asdict(config.method)
    method.pop("lvd_weight", None)

    def canonical_numbers(value):
        # YAML decoding normalizes typed floats (e.g. epochs=2 becomes 2.0).
        # Numerically identical configurations must resolve the same registry.
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, dict):
            return {name: canonical_numbers(child) for name, child in value.items()}
        if isinstance(value, list | tuple):
            return [canonical_numbers(child) for child in value]
        return value

    return _sha256(
        canonical_numbers(
            {
                "protocol_version": PROTOCOL_VERSION,
                "world_size": _world_size(),
                "model": asdict(config.model),
                "lora": asdict(config.lora),
                "data": data,
                "training": training,
                "method": method,
                "reference": asdict(config.reference),
                "lvd": lvd,
            }
        )
    )


def _registry_path(config: ExperimentConfig, key: str) -> Path:
    return (
        Path(config.paths.artifact_root).expanduser().resolve()
        / "reference_cache"
        / "pilots"
        / f"{key}.json"
    )


def _validate_export(checkpoint: str | Path) -> Path:
    """Structural HF/adapter check only; never deserialize large model weights."""
    source = Path(checkpoint).expanduser().resolve()
    if any((source / name).is_file() for name in ("model.safetensors", "pytorch_model.bin")):
        return source
    if (source / "adapter_config.json").is_file() and any(
        (source / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin")
    ):
        return source
    for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index = source / name
        if not index.is_file():
            continue
        try:
            weights = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
            if (
                isinstance(weights, dict)
                and weights
                and all(
                    isinstance(shard, str)
                    and Path(shard).name == shard
                    and (source / shard).is_file()
                    for shard in weights.values()
                )
            ):
                return source
        except (OSError, ValueError, KeyError, TypeError):
            continue
    raise ValueError(
        f"pilot checkpoint is not a loadable HF/adapter export: {source}; "
        "run generate_pilot and use its exported model, not FSDP resume shards"
    )


def _positive_integer(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _check_integer_fields(
    report: Mapping[str, Any], expected: Mapping[str, int], label: str
) -> None:
    if not isinstance(report, Mapping):
        raise ValueError(f"{label} must be a mapping")
    for name, value in expected.items():
        if type(report.get(name)) is not int or report[name] != value:
            raise ValueError(f"{label}.{name} must be {value}")


def _require_sha256(value: Any, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA256 digest")


def _require_close(value: Any, expected: float, label: str) -> None:
    if (
        not _finite_number(value)
        or not _finite_number(expected)
        or not math.isclose(value, expected, rel_tol=1e-9, abs_tol=0.0)
    ):
        raise ValueError(f"{label} is non-finite or inconsistent with its evidence")


def _validate_health_decision(report: Mapping[str, Any], label: str) -> None:
    if (
        not isinstance(report, Mapping)
        or type(report.get("version")) is not int
        or report["version"] != PROTOCOL_VERSION
        or report.get("accepted") is not True
        or report.get("reasons") != []
    ):
        raise ValueError(f"{label} requires an accepted pilot_health_decision report")
    # Recompute the fixed publication policy independently of caller-supplied
    # accepted flags AND thresholds. The standalone v1 health API stays tunable.
    floor = pilot_health_decision(report["baseline"], report["candidate"])
    if not floor["accepted"]:
        raise ValueError(f"{label} fails publication health thresholds: {floor['reasons']}")
    thresholds = report["thresholds"]
    if not isinstance(thresholds, Mapping):
        raise ValueError(f"{label}.thresholds must be a mapping")
    for name, required in floor["thresholds"].items():
        value = thresholds.get(name)
        if not _finite_number(value) or (
            value > required if name == "maximum_accuracy_drop" else value < required
        ):
            raise ValueError(f"{label}.{name} weakens the publication health thresholds")
    decision = pilot_health_decision(
        report["baseline"],
        report["candidate"],
        **{
            name: thresholds[name]
            for name in (
                "minimum_retention",
                "maximum_accuracy_drop",
                "minimum_relative_displacement",
            )
        },
    )
    if thresholds != decision["thresholds"] or not decision["accepted"]:
        raise ValueError(f"{label} does not pass its recorded health thresholds")
    for name in ("accuracy_drop", "relative_displacement_increase"):
        _require_close(report.get(name), decision[name], f"{label}.{name}")


def _validate_calibration_plan(
    calibration: Mapping[str, Any], config: ExperimentConfig, world_size: int
) -> None:
    groups = config.lvd.calibration_batches
    steps = config.lvd.calibration_accumulation_steps
    micros = groups * steps
    plan = calibration["calibration_plan"]
    _require_sha256(calibration["calibration_plan_sha256"], "calibration_plan_sha256")
    if _sha256(plan) != calibration["calibration_plan_sha256"]:
        raise ValueError("calibration_plan_sha256 does not match the recorded sample-ID plan")
    _check_integer_fields(
        plan,
        {
            "plan_version": 1,
            "world_size": world_size,
            "group_count": groups,
            "microbatches_per_group": steps,
        },
        "calibration_plan",
    )
    if plan.get("source") != "explicit":
        raise ValueError("calibration_plan.source must be explicit")
    ranks = plan.get("ranks")
    if not isinstance(ranks, list) or len(ranks) != world_size:
        raise ValueError("calibration_plan must contain exactly world_size ranks")
    sizes = {
        stream: _positive_integer(
            getattr(config.data, f"{stream}_batch_size"), f"{stream}_batch_size"
        )
        for stream in ("target", "anchor")
    }
    seen: dict[str, set[str]] = {stream: set() for stream in sizes}
    for rank, rank_plan in enumerate(ranks):
        _check_integer_fields(
            rank_plan,
            {"rank": rank, "microbatch_count": micros, "loader_restarts": 0},
            "calibration rank",
        )
        rows = rank_plan.get("microbatches")
        if not isinstance(rows, list) or len(rows) != micros:
            raise ValueError("calibration rank must contain exactly the configured microbatches")
        for micro, row in enumerate(rows):
            _check_integer_fields(
                row,
                {
                    "group_index": micro // steps,
                    "microbatch_index": micro,
                    "microbatch_in_group": micro % steps,
                    "loader_pass": 0,
                },
                "calibration microbatch",
            )
            for stream, size in sizes.items():
                ids = row.get(f"{stream}_sample_ids")
                if (
                    not isinstance(ids, list)
                    or len(ids) != size
                    or any(not isinstance(sample_id, str) or not sample_id for sample_id in ids)
                ):
                    raise ValueError(
                        f"calibration {stream}_sample_ids must match batch size {size}"
                    )
                if len(set(ids)) != size or seen[stream].intersection(ids):
                    raise ValueError(f"calibration {stream}_sample_ids repeat across batches/ranks")
                seen[stream].update(ids)


def _validate_three_segment_diagnostics(calibration: Mapping[str, Any]) -> None:
    """Check new reporting evidence, not geometry health or gradient-spread policy."""
    mode = LVDResponseMode.THREE_SEGMENT
    auxiliary = calibration["auxiliary_gradient_norms"]
    zero_groups = sum(value == 0 for value in auxiliary)
    _check_integer_fields(
        calibration,
        {
            "diagnostics_version": 2,
            "zero_auxiliary_groups": zero_groups,
            "positive_auxiliary_groups": len(auxiliary) - zero_groups,
        },
        "calibration",
    )
    schema = calibration.get("response_region_schema")
    _check_integer_fields(
        schema, {"version": 1, "count": mode.region_count}, "calibration.response_region_schema"
    )
    if schema != {"version": 1, "count": mode.region_count, "names": list(mode.region_names)}:
        raise ValueError("calibration.response_region_schema must match the exact response enum")
    count_names = tuple(f"lvd/response_region_{index}_count" for index in range(mode.region_count))
    rate_names = tuple(f"lvd/response_region_{index}_rate" for index in range(mode.region_count))
    total_name = "lvd/valid_boundary_count"
    batches, groups = calibration["batch_diagnostics"], calibration["group_diagnostics"]
    pooled = calibration["pooled_diagnostics"]
    for row in [*batches, *groups, pooled]:
        counts, metrics = row["counts"], row["metrics"]
        for values, expected in ((counts, count_names), (metrics, rate_names)):
            names = {
                name
                for name in values
                if isinstance(name, str) and name.startswith("lvd/response_region_")
            }
            if names != set(expected):
                raise ValueError("calibration diagnostics require only response regions 0, 1, 2")
        integers = {}
        for name in (*count_names, total_name):
            value = counts.get(name)
            if not _finite_number(value) or value < 0 or value != int(value):
                raise ValueError(
                    f"calibration diagnostic {name} must be a finite nonnegative integer count"
                )
            integers[name] = int(value)
        total = integers[total_name]
        if sum(integers[name] for name in count_names) != total:
            raise ValueError("calibration response region counts must sum to valid_boundary_count")
        for count_name, rate_name in zip(count_names, rate_names, strict=True):
            if total:
                _require_close(
                    metrics[rate_name], integers[count_name] / total, f"calibration {rate_name}"
                )
            elif metrics[rate_name] is not None:
                raise ValueError(
                    f"calibration {rate_name} must be null for zero valid_boundary_count"
                )
    for batch, group in zip(batches, groups, strict=True):
        if any(group[name] != batch[name] for name in ("counts", "metrics")):
            raise ValueError("calibration group diagnostics must mirror batch counts and metrics")
    # Counts are integer-valued FP64 measurements; compare integer sums exactly,
    # never with a relative tolerance that could hide missing boundary pairs.
    for name in (*count_names, total_name):
        if int(pooled["counts"][name]) != sum(int(group["counts"][name]) for group in groups):
            raise ValueError(f"calibration pooled {name} must equal the exact sum of group counts")


def _validate_calibration_report(
    calibration: Mapping[str, Any], config: ExperimentConfig, world_size: int
) -> None:
    three_segment = config.lvd.response_mode == LVDResponseMode.THREE_SEGMENT
    groups = _positive_integer(config.lvd.calibration_batches, "lvd.calibration_batches")
    steps = _positive_integer(
        config.lvd.calibration_accumulation_steps, "lvd.calibration_accumulation_steps"
    )
    training_steps = _positive_integer(
        config.training.gradient_accumulation_steps, "training.gradient_accumulation_steps"
    )
    _check_integer_fields(
        calibration,
        {
            "calibration_version": 3,
            "world_size": world_size,
            "calibration_batches": groups,
            "calibration_accumulation_steps": steps,
            "calibration_group_count": groups,
            "calibration_microbatch_count": groups * steps,
            "calibration_total_microbatch_count": groups * steps * world_size,
            "calibration_loader_restarts": 0,
            "training_gradient_accumulation_steps": training_steps,
        },
        "calibration",
    )
    statistic = config.lvd.calibration_statistic
    reported_statistic = calibration.get("calibration_statistic", "median")
    for label, value in (
        ("lvd.calibration_statistic", statistic),
        ("calibration.calibration_statistic", reported_statistic),
    ):
        if not isinstance(value, str) or value not in {"median", "rms"}:
            raise ValueError(f"{label} must be 'median' or 'rms'")
    if reported_statistic != statistic:
        raise ValueError(
            "calibration.calibration_statistic does not match lvd.calibration_statistic"
        )
    if statistic == "rms" and not three_segment:
        raise ValueError("calibration_statistic='rms' requires response_mode='three_segment'")
    for name, expected in {
        "mode": "gradient_ratio",
        "gradient_convention": "post_backward_distributed_average_compensated",
        "gradient_accumulation_convention": "mean_of_per_microbatch_losses",
        "calibration_batch_source": "explicit",
        "response_mode": config.lvd.response_mode,
    }.items():
        if calibration.get(name) != expected:
            raise ValueError(f"calibration.{name} must be {expected}")
    arrays = {}
    for name in (
        "target_gradient_norms",
        "auxiliary_gradient_norms",
        "auxiliary_to_target_ratios",
        "weighted_auxiliary_to_target_ratios",
    ):
        values = calibration.get(name)
        allow_zero = name == "weighted_auxiliary_to_target_ratios" or (
            three_segment and name != "target_gradient_norms"
        )
        if (
            not isinstance(values, list)
            or len(values) != groups
            or any(
                not _finite_number(value) or (value < 0 if allow_zero else value <= 0)
                for value in values
            )
        ):
            raise ValueError(f"calibration.{name} requires one finite valid value per group")
        arrays[name] = values
    ratio = calibration.get("target_gradient_ratio")
    resolved = calibration.get("resolved_lambda")
    if (
        not _finite_number(ratio)
        or ratio <= 0
        or not _finite_number(resolved)
        or resolved < 0
        or (statistic == "rms" and resolved == 0)
    ):
        raise ValueError(
            "calibration target_gradient_ratio/resolved_lambda must be finite and valid"
        )
    # Check recorded arithmetic, not a quality/spread threshold. In particular,
    # do NOT compare the measured ratio to config.lvd.target_gradient_ratio:
    # auto lookup deliberately permits remeasurement at a different ratio.
    ratios = arrays["auxiliary_to_target_ratios"]
    weighted = arrays["weighted_auxiliary_to_target_ratios"]
    for target, auxiliary, raw, scaled in zip(
        arrays["target_gradient_norms"],
        arrays["auxiliary_gradient_norms"],
        ratios,
        weighted,
        strict=True,
    ):
        if three_segment and (auxiliary == 0) != (raw == 0):
            raise ValueError(
                "calibration zero auxiliary_to_target_ratio requires a zero auxiliary norm"
            )
        _require_close(raw, auxiliary / target, "calibration auxiliary_to_target_ratio")
        _require_close(scaled, resolved * raw, "calibration weighted_auxiliary_to_target_ratio")
    normalizer = calibration_normalizer(ratios, statistic)
    if not _finite_number(normalizer) or normalizer <= 0:
        raise ValueError(
            f"calibration {statistic} auxiliary gradient ratio must be finite and positive"
        )
    # Under RMS, medians are descriptions, not normalization targets. Include
    # every legitimate zero and use the overflow-safe median for large ratios.
    median = calibration_normalizer(ratios) if statistic == "rms" else normalizer
    _require_close(
        calibration.get("median_auxiliary_to_target_ratio"), median, "calibration median"
    )
    _require_close(resolved, ratio / normalizer, "calibration resolved_lambda")
    _require_close(
        calibration.get("median_weighted_auxiliary_to_target_ratio"),
        calibration_normalizer(weighted),
        "calibration weighted median",
    )
    if statistic == "rms":
        weighted_rms = calibration_normalizer(weighted, statistic)
        if weighted_rms <= 0:
            raise ValueError("calibration weighted RMS gradient ratio must be positive")
        for name, expected in (
            ("normalization_value", normalizer),
            ("rms_auxiliary_to_target_ratio", normalizer),
            ("rms_weighted_auxiliary_to_target_ratio", weighted_rms),
        ):
            _require_close(calibration.get(name), expected, f"calibration {name}")
    _validate_calibration_plan(calibration, config, world_size)
    for name in ("batch_diagnostics", "group_diagnostics"):
        rows = calibration.get(name)
        if not isinstance(rows, list) or len(rows) != groups:
            raise ValueError(f"calibration.{name} must contain exactly {groups} groups")
        for index, row in enumerate(rows):
            expected = {"batch_index": index}
            if name == "group_diagnostics":
                expected.update(group_index=index, microbatch_count=steps)
            _check_integer_fields(row, expected, f"calibration.{name}")
    # Descriptive metrics may be missing (None) or poor. No retention, norm,
    # cap-rate, or ratio-spread threshold is imposed on calibration diagnostics.
    for row in [
        *calibration["batch_diagnostics"],
        *calibration["group_diagnostics"],
        calibration.get("pooled_diagnostics"),
    ]:
        if not isinstance(row, Mapping):
            raise ValueError("calibration diagnostics must be mappings")
        for name in ("metrics", "counts"):
            values = row.get(name)
            if not isinstance(values, Mapping) or any(
                value is not None and not _finite_number(value) for value in values.values()
            ):
                raise ValueError(f"calibration diagnostic {name} must be finite or null")
    if three_segment:
        _validate_three_segment_diagnostics(calibration)


def _validate_accepted_report(report: Mapping[str, Any], config: ExperimentConfig) -> str:
    """Validate publication v2 evidence without changing the v1 health contract."""
    try:
        _validate_health_decision(report, "health")
        validation = report["validation"]
        _validate_health_decision(validation, "validation")
        for name, decision, count in (
            ("health", report, config.lvd.pilot_health_samples),
            ("validation", validation, config.lvd.pilot_validation_samples),
        ):
            _positive_integer(count, f"lvd.pilot_{name}_samples")
            _check_integer_fields(decision["baseline"], {"sample_count": count}, name)
        for name in ("dataset_fingerprint", "cache_fingerprint"):
            if report["baseline"][name] != validation["baseline"][name]:
                raise ValueError(f"health/validation {name} mismatch")
        if report["baseline"]["stream_fingerprint"] == validation["baseline"]["stream_fingerprint"]:
            raise ValueError("health and validation must use independent Anchor subsets")
        world_size = _positive_integer(report.get("world_size"), "report.world_size")
        if world_size != _world_size():
            raise ValueError("report.world_size does not match the compatibility key world_size")
        pilot_steps = _positive_integer(report.get("pilot_steps"), "report.pilot_steps")
        scheduler_steps = _positive_integer(report.get("scheduler_steps"), "report.scheduler_steps")
        if pilot_steps > scheduler_steps:
            raise ValueError("pilot_steps must not exceed the full scheduler_steps")
        if "step" in report:
            _check_integer_fields(report, {"step": pilot_steps}, "health decision")
        # The workflow owns the full Anchor plan. Its digest binds that external
        # evidence; calibration's embedded sample-ID plan is rehashed below.
        _require_sha256(report.get("anchor_plan_sha256"), "anchor_plan_sha256")
        _validate_calibration_report(report["calibration"], config, world_size)
        stability = calibration_stability_decision(
            report["calibration"],
            minimum_groups=8
            if config.lvd.calibration_statistic == "rms"
            else max(2, min(8, config.lvd.calibration_batches)),
        )
        if not stability["accepted"]:
            raise ValueError(f"calibration stability checks failed: {stability['reasons']}")
        return _sha256(report)  # Also reject NaN/Inf in additional caller-supplied metadata.
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"invalid accepted pilot publication report: {error}") from error


def _verify_report_file(source: Path, report_sha256: str) -> None:
    try:
        stored = json.loads((source / "pilot-health.json").read_text(encoding="utf-8"))
        if _sha256(stored) != report_sha256:
            raise ValueError("report digest mismatch")
    except (OSError, TypeError, ValueError) as error:
        raise ValueError(f"pilot-health.json must match the published report: {error}") from error


def _write_publication_marker(source: Path, publication: Mapping[str, Any]) -> None:
    destination = source / "pilot-publication.json"
    # A complete temporary file + atomic link is create-only, unlike replace().
    # This also closes the race after the early already-published check.
    try:
        with TemporaryDirectory(prefix=".pilot-publication-", dir=source) as directory:
            temporary = Path(directory) / "publication.json"
            write_json_atomic(temporary, publication)
            os.link(temporary, destination)
    except FileExistsError as error:
        raise ValueError(
            f"pilot export is already published: {source}; use a new export"
        ) from error


def write_accepted_pilot_registry(
    config: ExperimentConfig,
    checkpoint: str | Path,
    report: Mapping[str, Any],
    *,
    registry_path: str | Path | None = None,
) -> Path:
    """Publish v2 evidence and bind it to a structurally loadable export, once.

    The caller (rank zero) first writes checkpoint/pilot-health.json containing
    report: a v1 health decision plus independent validation, full calibration v3,
    anchor_plan_sha256, world_size, and positive pilot_steps/scheduler_steps.
    The report hash is canonical JSON, not whitespace-sensitive file bytes.

    A fresh UUID marker is atomically created without overwriting an existing
    publication; then the registry is atomically replaced. If registry writing
    fails, the marker remains fail-closed: retry with a new export, not migration.
    This binds metadata, not tensor contents (large weights are never hashed).
    Returns the manifest Path without mutating report or adding fields to it.
    A custom registry_path is an explicit destination, not an auto-lookup path.
    """
    key = pilot_compatibility_key(config)
    report_sha256 = _validate_accepted_report(report, config)
    source = _validate_export(checkpoint)
    marker = source / "pilot-publication.json"
    if marker.exists() or marker.is_symlink():
        raise ValueError(f"pilot export is already published: {source}; use a new export")
    _verify_report_file(source, report_sha256)
    destination = Path(registry_path) if registry_path is not None else _registry_path(config, key)
    if destination.resolve() in {marker, source / "pilot-health.json"}:
        raise ValueError("registry_path must not overwrite the export's publication/report files")
    publication = {
        "publication_version": PUBLICATION_VERSION,
        "publication_id": str(uuid4()),
        "pilot_key": key,
        "report_sha256": report_sha256,
    }
    _write_publication_marker(source, publication)
    write_json_atomic(
        destination,
        {
            "version": PROTOCOL_VERSION,
            **publication,
            "status": "accepted",
            "checkpoint": str(source),
            "report": dict(report),
        },
    )
    return destination


def find_accepted_pilot(config: ExperimentConfig) -> dict[str, Any]:
    """Return the exact-key registry manifest or raise ValueError with next action.

    Read only the exact-key registry and its export's report/publication marker.
    Recheck both decisions, calibration, and UUID/key/report-digest binding.
    Legacy registries/exports are never scanned, rewritten, or promoted.
    """
    key = pilot_compatibility_key(config)
    path = _registry_path(config, key)
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or (
            type(manifest.get("version")) is not int
            or manifest["version"] != PROTOCOL_VERSION
            or type(manifest.get("publication_version")) is not int
            or manifest["publication_version"] != PUBLICATION_VERSION
            or manifest.get("pilot_key") != key
            or manifest.get("status") != "accepted"
        ):
            raise ValueError("registry publication version, pilot_key, or status mismatch")
        checkpoint = manifest.get("checkpoint")
        if not isinstance(checkpoint, str) or not Path(checkpoint).is_absolute():
            raise ValueError("registry checkpoint must be an absolute export path")
        report_sha256 = _validate_accepted_report(manifest["report"], config)
        publication = {name: manifest[name] for name in _PUBLICATION_FIELDS}
        identity = publication["publication_id"]
        if (
            not isinstance(identity, str)
            or UUID(identity).version != 4
            or str(UUID(identity)) != identity
        ):
            raise ValueError("publication_id must be a canonical UUID4")
        if publication["report_sha256"] != report_sha256:
            raise ValueError("registry report_sha256 does not match its report")
        source = _validate_export(checkpoint)
        if str(source) != checkpoint:
            raise ValueError("registry checkpoint must remain its canonical absolute export path")
        stored = json.loads((source / "pilot-publication.json").read_text(encoding="utf-8"))
        if _sha256(stored) != _sha256(publication):
            raise ValueError("export publication marker does not match the registry binding")
        _verify_report_file(source, report_sha256)
        return manifest
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise ValueError(
            f"no compatible accepted pilot at {path}: {error}; run generate_pilot "
            "with the matching formal source config"
        ) from error
