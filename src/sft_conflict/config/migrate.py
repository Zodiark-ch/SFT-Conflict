"""Conservative one-way migration from the Future Localization config shape."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from .io import parse_config

METHOD_MAP = {
    "TargetFT": "target_only",
    "TargetFT+PervasivenessFT": "target_plus_anchor_ce",
    "TargetFT+PervasivenessKL": "target_plus_anchor_kl",
}


def _leaf_paths(value: Any, prefix: str = "") -> list[str]:
    if isinstance(value, dict):
        paths: list[str] = []
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else key
            paths.extend(_leaf_paths(child, child_prefix))
        return paths
    return [prefix]


def _dataset_adapter(name: str) -> str:
    normalized = name.lower()
    if normalized in {"sst2", "sst-2"}:
        return "sst2"
    if normalized == "gsm8k":
        return "gsm8k"
    return "prompt_response"


def migrate_legacy_config(
    legacy: dict[str, Any],
    *,
    model_revision: str,
    dataset_revision: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    overall = dict(legacy.get("overall", {}))
    finetuning = dict(legacy.get("finetuning", {}))
    dataset = dict(legacy.get("dataset", {}))
    logger = dict(legacy.get("logger", {}))
    consumed: set[str] = set()
    mappings: list[dict[str, str]] = []

    def consume(path: str, target: str) -> None:
        consumed.add(path)
        mappings.append({"legacy": path, "new": target})

    model_id = overall.get("model_name")
    if not model_id:
        raise ValueError("legacy config is missing overall.model_name")
    consume("overall.model_name", "model.model_id")

    legacy_method = finetuning.get("finetuning_method", "TargetFT")
    if legacy_method not in METHOD_MAP:
        raise ValueError(f"unsupported legacy finetuning_method: {legacy_method}")
    method_name = METHOD_MAP[legacy_method]
    consume("finetuning.finetuning_method", "method.name")

    use_lora = bool(finetuning.get("use_lora", False))
    if "use_lora" in finetuning:
        consume("finetuning.use_lora", "model.training_mode")
    training_mode = "lora" if use_lora else "full_ft"
    target_name = dataset.get("target_dataset_name")
    if not target_name:
        raise ValueError("legacy config is missing dataset.target_dataset_name")
    consume("dataset.target_dataset_name", "data.target")

    batch_size = int(dataset.get("batch_size", 1))
    if "batch_size" in dataset:
        consume("dataset.batch_size", "data.target_batch_size,data.anchor_batch_size")
    data_config: dict[str, Any] = {
        "target": {
            "source": "hf",
            "path": str(target_name),
            "split": "train",
            "revision": dataset_revision,
            "adapter": _dataset_adapter(str(target_name)),
        },
        "target_batch_size": batch_size,
        "anchor_batch_size": batch_size,
    }
    if method_name != "target_only":
        anchor_name = dataset.get("pervasiveness_dataset_name")
        if not anchor_name or "," in str(anchor_name):
            raise ValueError(
                "anchor methods require one explicit legacy pervasiveness_dataset_name"
            )
        consume("dataset.pervasiveness_dataset_name", "data.anchor")
        data_config["anchor"] = {
            "source": "hf",
            "path": str(anchor_name),
            "split": "train",
            "revision": dataset_revision,
            "adapter": _dataset_adapter(str(anchor_name)),
        }
        data_config["anchor_batches_per_target"] = 1.0
    else:
        data_config["anchor_batches_per_target"] = 0.0

    if "dataset_seed" in dataset:
        data_config["seed"] = int(dataset["dataset_seed"])
        consume("dataset.dataset_seed", "data.seed")

    new_config: dict[str, Any] = {
        "model": {
            "model_id": str(model_id),
            "revision": model_revision,
            "training_mode": training_mode,
            "freeze_unembedding": use_lora,
        },
        "data": data_config,
        "method": {"name": method_name},
    }
    if method_name == "target_plus_anchor_ce" and "pervasiveness_weight" in finetuning:
        new_config["method"]["anchor_weight"] = float(finetuning["pervasiveness_weight"])
        consume("finetuning.pervasiveness_weight", "method.anchor_weight")
    if method_name == "target_plus_anchor_kl":
        new_config["reference"] = {"mode": "online_reference"}
        if "kl_weight" in finetuning:
            new_config["method"]["anchor_weight"] = float(finetuning["kl_weight"])
            consume("finetuning.kl_weight", "method.anchor_weight")

    training_map = {
        "num_epochs": "num_train_epochs",
        "lr": "learning_rate",
        "weight_decay": "weight_decay",
        "gradient_accumulation_steps": "gradient_accumulation_steps",
        "max_grad_norm": "max_grad_norm",
        "max_steps": "max_steps",
        "resume_path": "resume_from_checkpoint",
    }
    training_config: dict[str, Any] = {}
    for legacy_key, new_key in training_map.items():
        if legacy_key in finetuning and finetuning[legacy_key] is not None:
            training_config[new_key] = finetuning[legacy_key]
            consume(f"finetuning.{legacy_key}", f"training.{new_key}")
    if "seed" in overall:
        training_config["seed"] = int(overall["seed"])
        consume("overall.seed", "training.seed")
    if training_config:
        new_config["training"] = training_config

    lora_map = {
        "lora_default_rank": "rank",
        "lora_alpha": "alpha",
        "lora_dropout": "dropout",
    }
    lora_config: dict[str, Any] = {}
    if use_lora:
        for legacy_key, new_key in lora_map.items():
            if legacy_key in finetuning:
                lora_config[new_key] = finetuning[legacy_key]
                consume(f"finetuning.{legacy_key}", f"lora.{new_key}")
        target_modules = finetuning.get("lora_target_modules")
        if target_modules and target_modules != "auto":
            lora_config["target_modules"] = [
                value.strip() for value in str(target_modules).split(",") if value.strip()
            ]
            consume("finetuning.lora_target_modules", "lora.target_modules")
    if lora_config:
        new_config["lora"] = lora_config

    if logger.get("name"):
        new_config["paths"] = {"run_name": str(logger["name"])}
        consume("logger.name", "paths.run_name")
    json_root = logger.get("json", {}).get("root") if isinstance(logger.get("json"), dict) else None
    if json_root:
        new_config.setdefault("paths", {})["artifact_root"] = str(json_root)
        consume("logger.json.root", "paths.artifact_root")

    parse_config(new_config)
    all_paths = set(_leaf_paths(legacy))
    dropped = sorted(path for path in all_paths if path not in consumed)
    report = {
        "format": "sft-conflict-legacy-migration-v1",
        "mappings": mappings,
        "dropped_fields": [
            {"field": path, "reason": "no semantically equivalent M1 configuration field"}
            for path in dropped
        ],
        "warnings": [
            "Review dataset IDs and immutable revisions before training.",
            "Legacy evaluation fields are reported but not migrated into training behavior.",
        ],
    }
    return new_config, report


def migrate_legacy_config_file(
    input_path: str | Path,
    output_path: str | Path,
    report_path: str | Path,
    *,
    model_revision: str,
    dataset_revision: str,
) -> None:
    source_path = Path(input_path).expanduser().resolve()
    legacy = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    migrated, report = migrate_legacy_config(
        legacy,
        model_revision=model_revision,
        dataset_revision=dataset_revision,
    )
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(migrated, sort_keys=False), encoding="utf-8")
    report_destination = Path(report_path).expanduser().resolve()
    report_destination.parent.mkdir(parents=True, exist_ok=True)
    report_destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
