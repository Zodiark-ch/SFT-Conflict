"""Unified M1 command-line interface."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import socket
import subprocess
import sys
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, set_seed

from sft_conflict import __version__
from sft_conflict.adapters import load_and_merge_lora
from sft_conflict.benchmark import (
    DEFAULT_GENERAL_TASKS,
    DEFAULT_LLM_EVALUATION_TASKS,
    BenchmarkOrchestrator,
    aggregate_benchmark_results,
    build_benchmark_plan,
    write_benchmark_plan,
    write_benchmark_results,
)
from sft_conflict.cache_preparation import ensure_preset_cache, prepare_label_coverage_dataset
from sft_conflict.config import (
    ConfigError,
    ExperimentConfig,
    LVDLambdaMode,
    LVDPlusAnchorMode,
    MethodName,
    ReferenceMode,
    TrainingMode,
    dump_config,
    load_config,
)
from sft_conflict.config.migrate import migrate_legacy_config_file
from sft_conflict.data import (
    AnchorFilterScore,
    AnchorPositionConfig,
    AnchorPositionDataset,
    AnchorPositionSelector,
    AnchorPositionStrategy,
    CausalLMCollator,
    LeakageAuditConfig,
    ReferenceCache,
    ReferenceCacheBuilder,
    ReferenceCacheConfig,
    ReferenceFilterMode,
    ReferencePrecisionBudget,
    anchor_filter_score_dir,
    audit_dataset_leakage,
    estimate_reference_cache_bytes,
    filtered_anchor_dataset_dir,
    greedy_token_cover,
    load_logical_dataset,
    load_tokenized_dataset,
    read_filter_scores,
    reusable_filter_score_paths,
    score_anchor_shard,
    supervised_output_token_ids,
    write_filter_score_cache,
    write_filtered_anchor_artifacts,
    write_leakage_report,
)
from sft_conflict.data.anchor_creation import SUBMIXTURE_KEYS
from sft_conflict.data.collator import model_tensor_inputs
from sft_conflict.data.label_coverage import LABEL_COVERAGE_POLICY, LEGACY_FILTER_POLICY
from sft_conflict.diagnostics import (
    collect_cached_anchor_displacements,
    collect_supervised_next_token_predictions,
    crossing_statistics,
    token_direction_movement_statistics,
    write_interference_artifacts,
)
from sft_conflict.evaluation import (
    LmEvalRequest,
    build_native_evaluator,
    ensure_lm_eval_task_cache,
    run_lm_eval,
    write_evaluation_result,
)
from sft_conflict.evaluation.alpaca_eval_bridge import (
    ALPACA_EVAL_ANNOTATORS,
    run_official_alpaca_eval,
    write_alpaca_eval_scoring_instructions,
)
from sft_conflict.evaluation.native import ClassificationEvaluator
from sft_conflict.geometry import audit_lvd_partition
from sft_conflict.models import (
    ModelAudit,
    ModelBundle,
    UnembeddingInputCapture,
    audit_existing_training_model,
    audit_unembedding_forward,
    configure_model_for_training,
    get_unembedding,
    load_model_bundle,
    load_reference_model,
    torch_dtype_for_precision,
)
from sft_conflict.paths import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_LM_EVAL_HARNESS_PATH,
    DEFAULT_LM_EVAL_PYTHON,
)
from sft_conflict.quick_run import (
    DEFAULT_ANCHOR_GEOMETRY_CONFIG_PATH,
    DEFAULT_LVD_CONFIG_PATH,
    DEFAULT_LVD_PLUS_CONFIG_PATH,
    DEFAULT_S0_CONFIG_PATH,
    RUN_ANCHOR_DATASETS,
    RUN_MODELS,
    RUN_SUPERVISION_MODES,
    RUN_TARGET_DATASETS,
    QuickRunSettings,
    build_quick_run_config,
)
from sft_conflict.tracking import (
    JsonlMetricsCallback,
    MemoryDiagnosticsCallback,
    build_preflight_preview,
    build_random_logical_tensor_preview,
    check_capacity,
    collect_run_metadata,
    prepare_run,
    render_preflight_preview,
    render_tensor_preview,
    write_json_atomic,
)
from sft_conflict.trainers import SFTConflictTrainer
from sft_conflict.trainers.native_selection import NativeSelection


def _world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _is_world_process_zero() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def _enforce_offline_environment(config: ExperimentConfig) -> None:
    cache_root = Path(config.paths.artifact_root).expanduser().resolve() / "hf_cache"
    os.environ.update(
        {
            "HF_HOME": str(cache_root),
            "HF_HUB_CACHE": str(cache_root / "hub"),
            "HF_DATASETS_CACHE": str(cache_root / "datasets"),
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )


def _cache_paths(config: ExperimentConfig) -> tuple[Path, Path]:
    cache_root = Path(config.paths.artifact_root).expanduser().resolve() / "hf_cache"
    return cache_root / "hub", cache_root / "datasets"


def _validate_after_overrides(config: ExperimentConfig) -> None:
    try:
        config.validate()
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc


def _apply_overrides(config: ExperimentConfig, args: argparse.Namespace) -> None:
    if getattr(args, "run_name", None) is not None:
        config.paths.run_name = args.run_name
    if getattr(args, "mixed_precision", None) is not None:
        config.training.mixed_precision = args.mixed_precision
    if getattr(args, "training_mode", None) is not None:
        config.model.training_mode = TrainingMode(args.training_mode)
    if getattr(args, "freeze_unembedding", None) is not None:
        config.model.freeze_unembedding = args.freeze_unembedding
    if getattr(args, "resume_from_checkpoint", None) is not None:
        config.training.resume_from_checkpoint = args.resume_from_checkpoint
    if getattr(args, "eval_checkpoint", None) is not None:
        config.evaluation.eval_checkpoint = args.eval_checkpoint
    _validate_after_overrides(config)


def _preflight(config: ExperimentConfig) -> None:
    from sft_conflict.tracking.native_crash import enable_native_crash_diagnostics

    enable_native_crash_diagnostics()
    if _world_size() > 1 and not config.paths.run_name:
        raise ConfigError("multi-process runs require an explicit, shared paths.run_name")
    _enforce_offline_environment(config)
    check_capacity(
        config.paths.artifact_root,
        estimated_run_bytes=config.paths.estimated_run_bytes,
        minimum_free_bytes=config.paths.minimum_free_bytes,
        free_space_multiplier=config.paths.free_space_multiplier,
    )
    set_seed(config.training.seed)


def _distributed_strategy(config: ExperimentConfig, audit: ModelAudit) -> str:
    del audit
    world_size = _world_size()
    requested = config.training.distributed_strategy
    if world_size == 1:
        if requested in {"ddp", "fsdp"}:
            raise ConfigError(f"{requested.upper()} requires more than one process")
        return "single"
    if requested == "single":
        raise ConfigError("distributed_strategy=single cannot run with WORLD_SIZE > 1")
    return requested


def _fsdp_training_kwargs(
    config: ExperimentConfig,
    audit: ModelAudit,
) -> dict[str, Any]:
    if not audit.no_split_modules:
        raise ConfigError(
            f"{audit.model_type} does not declare transformer layers for FSDP wrapping"
        )
    fsdp_modes = ["full_shard"]
    cpu_offload = config.training.fsdp_cpu_offload
    if cpu_offload is None:
        cpu_offload = audit.total_parameters >= 6_000_000_000
    if cpu_offload:
        fsdp_modes.append("offload")
    fsdp_modes.append("auto_wrap")
    return {
        "fsdp": " ".join(fsdp_modes),
        "fsdp_config": {
            "transformer_layer_cls_to_wrap": list(audit.no_split_modules),
            "use_orig_params": (
                config.model.training_mode is not TrainingMode.FULL_FT
                or config.method.name.uses_dynamic_geometry
            ),
            "activation_checkpointing": config.model.gradient_checkpointing,
            "forward_prefetch": False,
            "limit_all_gathers": True,
        },
    }


def _training_arguments(
    config: ExperimentConfig,
    audit: ModelAudit,
    *,
    checkpoint_dir: Path,
    tensorboard_dir: Path,
    has_eval_dataset: bool,
    best_model_metric: str = "eval_loss",
) -> TrainingArguments:
    if best_model_metric not in {"eval_loss", "native_accuracy"}:
        raise ConfigError("best_model_metric must be eval_loss or native_accuracy")
    native_selection = has_eval_dataset and best_model_metric == "native_accuracy"
    strategy = _distributed_strategy(config, audit)
    kwargs: dict[str, Any] = {}
    if strategy == "fsdp":
        kwargs.update(_fsdp_training_kwargs(config, audit))
    trainer_mixed_precision = not (
        strategy == "fsdp" and config.training.fsdp_keep_low_precision_params
    )
    return TrainingArguments(
        output_dir=str(checkpoint_dir),
        overwrite_output_dir=False,
        do_train=True,
        do_eval=has_eval_dataset,
        eval_strategy="epoch" if has_eval_dataset else "no",
        prediction_loss_only=True,
        per_device_train_batch_size=config.data.target_batch_size,
        per_device_eval_batch_size=config.evaluation.batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        optim=config.training.optimizer,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        max_grad_norm=config.training.max_grad_norm,
        num_train_epochs=config.training.num_train_epochs,
        max_steps=config.training.max_steps,
        warmup_ratio=config.training.warmup_ratio,
        lr_scheduler_type=config.training.lr_scheduler_type,
        logging_strategy="steps",
        logging_steps=config.training.logging_steps,
        logging_first_step=True,
        logging_dir=str(tensorboard_dir),
        save_strategy="epoch" if has_eval_dataset else config.training.save_strategy,
        save_steps=config.training.save_steps,
        save_total_limit=config.training.save_total_limit,
        save_safetensors=True,
        load_best_model_at_end=has_eval_dataset,
        metric_for_best_model=(
            ("native_accuracy" if native_selection else "eval_loss") if has_eval_dataset else None
        ),
        greater_is_better=(native_selection if has_eval_dataset else None),
        seed=config.training.seed,
        data_seed=config.training.data_seed,
        bf16=trainer_mixed_precision and config.training.mixed_precision == "bf16",
        fp16=trainer_mixed_precision and config.training.mixed_precision == "fp16",
        report_to=list(config.training.report_to),
        run_name=config.paths.run_name,
        remove_unused_columns=False,
        include_num_input_tokens_seen=False,
        ddp_find_unused_parameters=False if strategy == "ddp" else None,
        gradient_checkpointing=(config.model.gradient_checkpointing and strategy != "fsdp"),
        **kwargs,
    )


def _load_datasets(
    config: ExperimentConfig,
    tokenizer: Any,
    *,
    include_eval: bool,
    include_validation: bool = False,
):
    _, dataset_cache = _cache_paths(config)
    target = load_tokenized_dataset(
        config.data.target,
        tokenizer,
        max_length=config.data.max_length,
        model_variant=config.model.model_variant,
        cache_dir=dataset_cache,
    )
    anchor = (
        load_tokenized_dataset(
            config.data.anchor,
            tokenizer,
            max_length=config.data.max_length,
            model_variant=config.model.model_variant,
            cache_dir=dataset_cache,
            progress_description=("Anchor dataset" if _is_world_process_zero() else None),
        )
        if config.data.anchor is not None
        else None
    )
    if anchor is not None and config.method.name.uses_lvd:
        selector = AnchorPositionSelector(
            AnchorPositionConfig(
                strategy=AnchorPositionStrategy(config.lvd.position_strategy),
                positions_per_sample=config.lvd.positions_per_sample,
                seed=config.lvd.position_seed,
            ),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        anchor = AnchorPositionDataset(anchor, selector)
    validation = (
        load_tokenized_dataset(
            config.data.validation,
            tokenizer,
            max_length=config.data.max_length,
            model_variant=config.model.model_variant,
            cache_dir=dataset_cache,
        )
        if include_validation and config.data.validation is not None
        else None
    )
    evaluation = None
    if include_eval:
        evaluation_config = config.evaluation.target or config.data.target
        evaluation = load_tokenized_dataset(
            evaluation_config,
            tokenizer,
            max_length=config.data.max_length,
            model_variant=config.model.model_variant,
            cache_dir=dataset_cache,
        )
    return target, anchor, validation, evaluation


def _native_selection(config: ExperimentConfig, tokenizer, validation_dataset):
    """Validation-split native accuracy for best-checkpoint selection, when requested."""
    if config.training.best_model_metric != "native_accuracy":
        return None
    if validation_dataset is None or config.data.validation is None:
        raise ConfigError(
            "best_model_metric=native_accuracy needs the training validation split "
            "(keep --training-validation enabled)"
        )
    task_name = config.data.validation.adapter
    try:
        evaluator = build_native_evaluator(task_name, max_input_length=config.data.max_length)
    except ValueError as exc:
        raise ConfigError(
            f"best_model_metric=native_accuracy has no native evaluator for {task_name!r}: {exc}"
        ) from exc
    if not isinstance(evaluator, ClassificationEvaluator) or not evaluator.supports_selection():
        raise ConfigError(
            "best_model_metric=native_accuracy requires a fixed-candidate classification task "
            f"(got {type(evaluator).__name__} for {task_name!r})"
        )
    dataset = load_logical_dataset(config.data.validation, cache_dir=_cache_paths(config)[1])
    return NativeSelection(
        evaluator=evaluator,
        dataset=dataset,
        tokenizer=tokenizer,
        batch_size=config.evaluation.batch_size,
        task_name=task_name,
    )


def _build_trainer(
    config: ExperimentConfig,
    bundle,
    target_dataset,
    anchor_dataset,
    eval_dataset,
    paths,
    *,
    with_reference: bool,
    validation_dataset=None,
) -> SFTConflictTrainer:
    reference_model = (
        load_reference_model(
            config.reference,
            config.model,
            config.training,
            cache_dir=_cache_paths(config)[0],
        )
        if (
            with_reference
            and not config.method.name.uses_lvd_plus
            and not config.method.name.uses_dynamic_geometry
        )
        else None
    )
    native_selection = (
        _native_selection(config, bundle.tokenizer, validation_dataset)
        if eval_dataset is not None
        else None
    )
    arguments = _training_arguments(
        config,
        bundle.audit,
        checkpoint_dir=paths.checkpoint_dir,
        tensorboard_dir=paths.tensorboard_dir,
        has_eval_dataset=eval_dataset is not None,
        best_model_metric="native_accuracy" if native_selection is not None else "eval_loss",
    )
    reference_cache = None
    resolved_lvd_lambda = None
    if config.method.name.uses_lvd:
        if anchor_dataset is None or not config.lvd.d0_cache_path:
            raise ConfigError("target_plus_anchor_lvd requires positioned Anchor data and d0 cache")
        if not (Path(config.lvd.d0_cache_path) / "manifest.json").is_file():
            raise ConfigError(
                "LVD d0 cache is missing. Run METHOD='record_anchor_d0' with the same "
                "LVD position settings first; an old four-position cache cannot cover "
                f"dynamic positions: {config.lvd.d0_cache_path}"
            )
        reference_cache = ReferenceCache(config.lvd.d0_cache_path)
        reference_cache.validate_integrity()
        reference_cache.validate_binding(
            model_id=config.model.model_id,
            model_revision=config.model.revision,
            vocabulary_size=int(get_unembedding(bundle.model).weight.shape[0]),
            dataset_fingerprint=anchor_dataset.fingerprint,
            position_config=anchor_dataset.selector.config.to_dict(),
            normal_epsilon=config.lvd.normal_epsilon,
            filter_mode=config.lvd.d0_filter_mode,
            storage_dtype=config.lvd.d0_storage_dtype,
            minimum_confidence=config.lvd.d0_minimum_confidence,
            minimum_margin=config.lvd.d0_minimum_margin,
        )
        if config.lvd.resolved_lambda is not None:
            resolved_lvd_lambda = config.lvd.resolved_lambda
        elif config.lvd.lambda_mode is LVDLambdaMode.FIXED:
            resolved_lvd_lambda = config.lvd.fixed_lambda
    plus_kwargs = {}
    if config.method.name.uses_lvd_plus:
        plus = config.lvd_plus
        plus_kwargs = {
            "lvd_plus_config": plus,
            "resolved_lvd_plus_lambda": (
                plus.resolved_lambda
                if plus.resolved_lambda is not None
                else plus.fixed_lambda
                if plus.lambda_mode is LVDLambdaMode.FIXED
                else None
            ),
        }
    trainer = SFTConflictTrainer(
        model=bundle.model,
        args=arguments,
        target_dataset=target_dataset,
        anchor_dataset=anchor_dataset,
        eval_dataset=eval_dataset,
        data_config=config.data,
        method_config=config.method,
        pad_token_id=bundle.tokenizer.pad_token_id,
        preview_tokenizer=bundle.tokenizer,
        reference_model=reference_model,
        resolved_config=config,
        anchor_composition_dir=paths.result_dir,
        fsdp_sync_each_batch=config.training.fsdp_sync_each_batch,
        lvd_config=config.lvd,
        reference_cache=reference_cache,
        resolved_lvd_lambda=resolved_lvd_lambda,
        native_selection=native_selection,
        **plus_kwargs,
    )
    if native_selection is not None and _is_world_process_zero():
        print(
            "Best-checkpoint selection: native validation accuracy "
            f"(task={native_selection.task_name}, samples={len(native_selection.dataset)}, "
            f"candidates={native_selection.evaluator.candidate_responses()})",
            flush=True,
        )
    trainer.add_callback(JsonlMetricsCallback(paths.result_dir / "metrics.jsonl"))
    trainer.add_callback(MemoryDiagnosticsCallback(paths.result_dir))
    return trainer


def _checkpoint_resolved_config(checkpoint: str | Path) -> Path:
    source = Path(checkpoint).expanduser().resolve()
    candidates = (source / "resolved-config.yaml", source.parent / "resolved-config.yaml")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise ConfigError(f"checkpoint is missing resolved-config.yaml: {source}")


def _validate_lvd_pilot(config: ExperimentConfig, checkpoint_config: ExperimentConfig) -> None:
    if checkpoint_config.method.name is not MethodName.TARGET_ONLY:
        raise ConfigError("LVD calibration checkpoint must come from method=target_only")
    if asdict(checkpoint_config.model) != asdict(config.model):
        raise ConfigError("LVD calibration checkpoint ModelConfig does not match")
    if asdict(checkpoint_config.lora) != asdict(config.lora):
        raise ConfigError("LVD calibration checkpoint LoraConfig does not match")
    if asdict(checkpoint_config.data.target) != asdict(config.data.target):
        raise ConfigError("LVD calibration checkpoint target dataset does not match")
    if checkpoint_config.data.max_length != config.data.max_length:
        raise ConfigError("LVD calibration checkpoint max_length does not match")


def _validate_lvd_pilot_export(checkpoint: str | Path) -> None:
    source = Path(checkpoint).expanduser().resolve()
    full_model_files = (
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
    )
    has_full_model = any((source / name).is_file() for name in full_model_files)
    has_adapter = (source / "adapter_config.json").is_file() and any(
        (source / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin")
    )
    if not has_full_model and not has_adapter:
        raise ConfigError(
            "LVD calibration checkpoint is not a loadable final model export: "
            f"{source}. Point LVD_CALIBRATION_CHECKPOINT to the pilot run's final/ "
            "directory; checkpoint-* directories contain FSDP resume shards."
        )


def _calibrate_lvd_from_pilot(config: ExperimentConfig) -> dict[str, Any]:
    from sft_conflict.pilot_workflow import (
        calibration_batches_for_config,
        resolve_pilot_checkpoint,
    )

    checkpoint = resolve_pilot_checkpoint(config)
    checkpoint_config = load_config(_checkpoint_resolved_config(checkpoint))
    _validate_lvd_pilot(config, checkpoint_config)
    _validate_lvd_pilot_export(checkpoint)
    pilot_model, tokenizer = _load_native_evaluation_model(
        config,
        checkpoint,
        trainable_adapter=True,
    )
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    if (checkpoint_path / "adapter_config.json").is_file():
        if config.model.training_mode is not TrainingMode.LORA:
            raise ConfigError("an Adapter pilot checkpoint requires training_mode=lora")
        if not any(parameter.requires_grad for parameter in pilot_model.parameters()):
            raise ConfigError("the calibration Adapter checkpoint has no trainable parameters")
    else:
        pilot_model, _ = configure_model_for_training(
            pilot_model,
            config.model,
            config.lora,
        )
    if any(parameter.requires_grad for parameter in get_unembedding(pilot_model).parameters()):
        raise ConfigError("LVD calibration pilot must keep the Unembedding frozen")
    bundle = ModelBundle(
        model=pilot_model,
        tokenizer=tokenizer,
        audit=audit_existing_training_model(pilot_model),
    )
    target, anchor, _validation, _evaluation = _load_datasets(
        config,
        tokenizer,
        include_eval=False,
        include_validation=False,
    )
    calibration_root = (
        Path(config.paths.artifact_root)
        / "tmp"
        / "lvd-lambda-calibration"
        / str(config.paths.run_name)
    )
    calibration_paths = SimpleNamespace(
        checkpoint_dir=calibration_root / "checkpoints",
        tensorboard_dir=calibration_root / "tensorboard",
        result_dir=calibration_root / "results",
    )
    trainer = _build_trainer(
        config,
        bundle,
        target,
        anchor,
        None,
        calibration_paths,
        with_reference=False,
    )
    try:
        batches = calibration_batches_for_config(config, target, anchor, tokenizer.pad_token_id)
        report = (
            trainer.calibrate_lvd_lambda()
            if batches is None
            else trainer.calibrate_lvd_lambda(batches=batches)
        )
        if report is None:
            raise RuntimeError("LVD calibration trainer did not register the LVD plugin")
        report["pilot_checkpoint"] = str(checkpoint_path)
        if config.lvd.calibration_seed is not None:
            from sft_conflict.pilot_stability import calibration_stability_decision

            stability = calibration_stability_decision(
                report,
                minimum_groups=max(2, min(8, config.lvd.calibration_batches)),
            )
            report["pilot_stability"] = stability
            if not stability["accepted"]:
                if _is_world_process_zero():
                    write_json_atomic(calibration_root / "rejected-calibration.json", report)
                raise RuntimeError(
                    "pilot recalibration failed stability checks; formal training was not started: "
                    f"{stability['reasons']}"
                )
        return report
    finally:
        del trainer, bundle, pilot_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _restore_lvd_lambda_for_resume(config: ExperimentConfig) -> None:
    if not config.method.name.uses_lvd or not config.training.resume_from_checkpoint:
        return
    checkpoint_config = load_config(
        _checkpoint_resolved_config(config.training.resume_from_checkpoint)
    )
    if config.method != checkpoint_config.method:
        raise ConfigError("LVD resume method/weights differ from the checkpoint; start a new run")
    if checkpoint_config.lvd.resolved_lambda is None:
        raise ConfigError("LVD resume checkpoint has no resolved calibrated lambda")
    current_lvd = asdict(config.lvd)
    checkpoint_lvd = asdict(checkpoint_config.lvd)
    current_lvd.pop("resolved_lambda", None)
    checkpoint_lvd.pop("resolved_lambda", None)
    if current_lvd != checkpoint_lvd:
        raise ConfigError("LVD resume configuration differs from the checkpoint")
    config.lvd.resolved_lambda = checkpoint_config.lvd.resolved_lambda


def _step_zero_audit(trainer: SFTConflictTrainer) -> dict[str, Any]:
    batch = next(iter(trainer.get_train_dataloader()))
    target_inputs = model_tensor_inputs(batch["target"])
    model_device = next(trainer.model.parameters()).device
    if model_device.type == "cpu" and trainer.args.device.type == "cuda":
        trainer.model.to(trainer.args.device)
        model_device = next(trainer.model.parameters()).device
    target_inputs = {key: value.to(model_device) for key, value in target_inputs.items()}
    audit = audit_unembedding_forward(trainer.model, target_inputs)
    return asdict(audit)


def _restore_lvd_plus_lambda_for_resume(config: ExperimentConfig) -> None:
    """Never borrow a legacy lambda or silently change label-site geometry on resume."""
    if not config.method.name.uses_lvd_plus or not config.training.resume_from_checkpoint:
        return
    saved = load_config(_checkpoint_resolved_config(config.training.resume_from_checkpoint))
    if not saved.method.name.uses_lvd_plus or saved.method != config.method:
        raise ConfigError("LVD+ resume method differs from the checkpoint; start a new run")
    for name in ("model", "lora", "data"):
        if asdict(getattr(config, name)) != asdict(getattr(saved, name)):
            raise ConfigError(f"LVD+ resume {name} configuration differs from the checkpoint")
    current, checkpoint = asdict(config.lvd_plus), asdict(saved.lvd_plus)
    current.pop("resolved_lambda", None)
    checkpoint.pop("resolved_lambda", None)
    if config.lvd_plus.uses_boundary_risk:
        for key in (
            "resolved_outer_scale",
            "resolved_risk_scale",
            "scale_cache_sha256",
            "lambda_cache_sha256",
        ):
            if key == "lambda_cache_sha256" and config.lvd_plus.lambda_mode is LVDLambdaMode.FIXED:
                continue
            if checkpoint.get(key) is None:
                raise ConfigError(f"hybrid resume checkpoint has no verified {key}")
            if current[key] is None:
                current[key] = checkpoint[key]
    if config.lvd_plus.anchor_mode is LVDPlusAnchorMode.RECORDED_S0:
        if asdict(config.s0) != asdict(saved.s0):
            raise ConfigError("LVD+ resume S0 solver settings differ from the checkpoint")
        if saved.lvd_plus.s0_cache_sha256 is None:
            raise ConfigError("LVD+ resume checkpoint has no verified S0 manifest SHA")
        if current["s0_cache_sha256"] is None:
            current["s0_cache_sha256"] = saved.lvd_plus.s0_cache_sha256
    if current != checkpoint:
        raise ConfigError("LVD+ resume configuration differs from the checkpoint")
    if saved.lvd_plus.resolved_lambda is None:
        raise ConfigError("LVD+ resume checkpoint has no resolved lambda")
    if config.lvd_plus.anchor_mode is LVDPlusAnchorMode.RECORDED_S0:
        config.lvd_plus.s0_cache_sha256 = saved.lvd_plus.s0_cache_sha256
    if config.lvd_plus.uses_boundary_risk:
        for key in (
            "resolved_outer_scale",
            "resolved_risk_scale",
            "scale_cache_sha256",
            "lambda_cache_sha256",
        ):
            setattr(config.lvd_plus, key, getattr(saved.lvd_plus, key))
    config.lvd_plus.resolved_lambda = saved.lvd_plus.resolved_lambda


def _write_run_provenance(
    config,
    bundle,
    target,
    anchor,
    validation,
    evaluation,
    paths,
    step_zero,
) -> None:
    if not _is_world_process_zero():
        return
    dump_config(config, paths.result_dir / "resolved-config.yaml")
    dump_config(config, paths.checkpoint_dir / "resolved-config.yaml")
    metadata = collect_run_metadata(
        config,
        bundle.audit,
        paths,
        target_fingerprint=target.fingerprint,
        anchor_fingerprint=anchor.fingerprint if anchor is not None else None,
    )
    metadata["step_zero_unembedding_audit"] = step_zero
    write_json_atomic(paths.result_dir / "run-metadata.json", metadata)
    write_json_atomic(paths.result_dir / "target-dataset-manifest.json", target.manifest)
    if anchor is not None:
        write_json_atomic(paths.result_dir / "anchor-dataset-manifest.json", anchor.manifest)
    if validation is not None:
        write_json_atomic(
            paths.result_dir / "validation-dataset-manifest.json",
            validation.manifest,
        )
    if evaluation is not None:
        write_json_atomic(
            paths.result_dir / "evaluation-dataset-manifest.json",
            evaluation.manifest,
        )


def _validate_lvd_training_cache_metadata(config: ExperimentConfig) -> None:
    """Reject missing/old-position caches before loading even the calibration model."""
    if not config.method.name.uses_lvd:
        return
    manifest_path = Path(config.lvd.d0_cache_path or "") / "manifest.json"
    if not manifest_path.is_file():
        raise ConfigError(
            "LVD d0 cache is missing. Run METHOD='record_anchor_d0' with the same "
            f"LVD position settings first: {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_positions = AnchorPositionConfig(
        strategy=AnchorPositionStrategy(config.lvd.position_strategy),
        positions_per_sample=config.lvd.positions_per_sample,
        seed=config.lvd.position_seed,
    ).to_dict()
    if manifest.get("position_config") != expected_positions:
        raise ConfigError(
            f"LVD d0 position binding mismatch at {manifest_path}: "
            f"cached={manifest.get('position_config')}, requested={expected_positions}. "
            "Select the cache recorded with these position settings. In run.py, set "
            "ANCHOR_D0_CACHE_PATH=None to use the automatically derived path; an explicit "
            "old path overrides it even after recording a new cache. Re-record only if "
            "no matching cache exists."
        )
    if manifest.get("status") != "complete":
        raise ConfigError("LVD d0 cache is incomplete; finish record_anchor_d0 before training")


def _train(config: ExperimentConfig) -> dict[str, Any]:
    if config.method.name.uses_lvd_plus or config.method.name.uses_dynamic_geometry:
        _validate_after_overrides(config)
    if config.method.name.uses_dynamic_geometry:
        path = Path(config.anchor_geometry.reference_cache_path) / "manifest.json"
        if not path.is_file():
            raise ConfigError(
                "dynamic geometry requires Base outputs; run "
                f"METHOD='prepare_anchor_geometry_reference' first: {path}"
            )
    _preflight(config)
    if config.method.name.uses_lvd_plus:
        if config.lvd_plus.uses_boundary_risk:
            from sft_conflict.trainers.lvd_plus_recording import resolve_record_paths

            scale_path, _ = resolve_record_paths(config, _world_size())
            if not scale_path.is_file():
                raise ConfigError(
                    "hybrid LVD+ requires recorded SO/SR; run "
                    f"METHOD='record_lvd_plus_scales' first: {scale_path}"
                )
        _restore_lvd_plus_lambda_for_resume(config)
        if config.lvd_plus.anchor_mode is LVDPlusAnchorMode.RECORDED_S0:
            from sft_conflict.s0_workflow import require_s0_cache

            require_s0_cache(config)
    elif not config.method.name.uses_dynamic_geometry:
        _restore_lvd_lambda_for_resume(config)
        _validate_lvd_training_cache_metadata(config)
    pilot_calibration = None
    if (
        config.method.name.uses_lvd
        and config.lvd.lambda_mode is LVDLambdaMode.GRADIENT_RATIO
        and config.lvd.resolved_lambda is None
    ):
        from sft_conflict.pilot_workflow import accepted_pilot_calibration

        pilot_calibration = accepted_pilot_calibration(config)
        if pilot_calibration is None:
            pilot_calibration = _calibrate_lvd_from_pilot(config)
    include_eval = config.evaluation.run_after_training
    bundle = load_model_bundle(
        config.model,
        config.lora,
        config.training,
        cache_dir=_cache_paths(config)[0],
    )
    target, anchor, validation, evaluation = _load_datasets(
        config,
        bundle.tokenizer,
        include_eval=include_eval,
        include_validation=True,
    )
    if pilot_calibration is not None and pilot_calibration.get("reused_calibration"):
        from sft_conflict.pilot_workflow import validate_pilot_calibration_datasets

        validate_pilot_calibration_datasets(pilot_calibration, target, anchor)
    if config.method.name.uses_lvd:
        from sft_conflict.pilot_workflow import training_anchor_view

        anchor = training_anchor_view(config, anchor)
    if _is_world_process_zero():
        preview = build_preflight_preview(
            config,
            bundle.audit,
            bundle.tokenizer,
            target,
            anchor,
            evaluation,
            validation_dataset=validation,
            sample_count=3,
        )
        preview_text = render_preflight_preview(preview)
        print(preview_text, flush=True)
    allow_existing = config.training.resume_from_checkpoint is not None or _world_size() > 1
    paths = prepare_run(config, bundle.audit, allow_existing=allow_existing)
    if _is_world_process_zero():
        write_json_atomic(paths.result_dir / "preflight-preview.json", preview)
        (paths.result_dir / "preflight-preview.txt").write_text(
            preview_text,
            encoding="utf-8",
        )
    trainer = _build_trainer(
        config,
        bundle,
        target,
        anchor,
        validation or evaluation,
        paths,
        with_reference=True,
        validation_dataset=validation,
    )
    lvd_plus_calibration = None
    if config.method.name.uses_lvd_plus:
        if config.lvd_plus.uses_boundary_risk:
            from sft_conflict.trainers.lvd_plus_recording import bind_lvd_plus_scales

            # Bind the actual initial model before distributed preparation;
            # missing/stale scale artifacts must fail before any training update.
            bind_lvd_plus_scales(trainer)
        # The ordinary audit reads the head outside forward. Run it while the
        # Base head is whole, before in-place calibration wraps it with FSDP.
        step_zero = _step_zero_audit(trainer)
        # This calibrates the loaded Base IN PLACE, before its first update.
        # It never consumes training cursors, reserves coverage rows, or loads a pilot.
        lvd_calibration = None
        lvd_plus_calibration = trainer.calibrate_lvd_plus_lambda(
            paths.result_dir / "lvd-plus-lambda-calibration.json"
        )
        if _is_world_process_zero():
            write_json_atomic(
                paths.result_dir / "lvd-plus-lambda-calibration.json", lvd_plus_calibration
            )
    elif config.method.name.uses_dynamic_geometry:
        lvd_calibration = None
    else:
        lvd_calibration = pilot_calibration or trainer.calibrate_lvd_lambda(
            paths.result_dir / "lvd-lambda-calibration.json"
        )
    if pilot_calibration is not None and _is_world_process_zero():
        write_json_atomic(
            paths.result_dir / "lvd-lambda-calibration.json",
            pilot_calibration,
        )
    if not config.method.name.uses_lvd_plus:
        step_zero = _step_zero_audit(trainer)
    _write_run_provenance(
        config,
        bundle,
        target,
        anchor,
        validation,
        evaluation,
        paths,
        step_zero,
    )
    result = trainer.train(resume_from_checkpoint=config.training.resume_from_checkpoint)
    final_dir = paths.checkpoint_dir / "final"
    trainer.save_model(str(final_dir))
    if trainer.is_world_process_zero():
        bundle.tokenizer.save_pretrained(final_dir)
        dump_config(config, final_dir / "resolved-config.yaml")
    metrics: dict[str, Any] = {"train": result.metrics, "global_step": result.global_step}
    if lvd_calibration is not None:
        metrics["lvd_lambda_calibration"] = lvd_calibration
    if lvd_plus_calibration is not None:
        metrics["lvd_plus_lambda_calibration"] = lvd_plus_calibration
    if config.method.name.uses_dynamic_geometry:
        from sft_conflict.trainers.anchor_geometry import save_geometry_state

        save_geometry_state(trainer, final_dir)
        report = trainer.anchor_geometry_calibration_report
        metrics["anchor_geometry_calibration"] = report
        if report is not None and trainer.is_world_process_zero():
            write_json_atomic(paths.result_dir / "anchor-geometry-calibration.json", report)
            dump_config(config, paths.result_dir / "resolved-config.yaml")
    if config.evaluation.run_after_training:
        if validation is not None:
            metrics["validation"] = trainer.evaluate(
                eval_dataset=validation,
                metric_key_prefix="validation",
            )
        if evaluation is not None:
            metric_prefix = "test" if validation is not None else "eval"
            metrics["evaluation"] = trainer.evaluate(
                eval_dataset=evaluation,
                metric_key_prefix=metric_prefix,
            )
    if trainer.is_world_process_zero():
        write_json_atomic(paths.result_dir / "summary.json", metrics)
    return {
        "run_id": paths.run_id,
        "result_dir": str(paths.result_dir),
        "final_checkpoint": str(final_dir),
        **metrics,
    }


def _prepare_anchor_geometry_reference(config: ExperimentConfig) -> dict[str, Any]:
    from sft_conflict.data.anchor_geometry_reference import build_anchor_geometry_reference

    if _world_size() != 1:
        raise ConfigError(
            "prepare_anchor_geometry_reference is a single-process Base inference job"
        )
    if not config.method.name.uses_dynamic_geometry:
        raise ConfigError("geometry output preparation requires the dynamic geometry method")
    _preflight(config)
    bundle = load_model_bundle(
        config.model, config.lora, config.training, cache_dir=_cache_paths(config)[0]
    )
    bundle.model.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    anchor = load_tokenized_dataset(
        config.data.anchor,
        bundle.tokenizer,
        max_length=config.data.max_length,
        model_variant=config.model.model_variant,
        cache_dir=_cache_paths(config)[1],
    )
    cache = build_anchor_geometry_reference(
        config.anchor_geometry.reference_cache_path,
        model=bundle.model,
        dataset=anchor,
        tokenizer=bundle.tokenizer,
        config=config,
    )
    return {
        "method": config.method.name.value,
        "reference_path": str(cache.path),
        "reference_sha256": cache.sha256,
        "sample_count": len(anchor),
        "optimizer_updates": 0,
    }


def _prepare_lvd_plus_record(config: ExperimentConfig, *, scales: bool) -> dict[str, Any]:
    """Standalone record actions: fresh Base only, no optimizer/train/final export."""
    if not config.method.name.uses_lvd_plus or not config.lvd_plus.uses_boundary_risk:
        raise ConfigError("LVD+ preparation actions require the pressure_plus_risk objective")
    if config.training.resume_from_checkpoint:
        raise ConfigError("LVD+ preparation actions do not accept checkpoint resume")
    if not scales and config.lvd_plus.lambda_mode is LVDLambdaMode.FIXED:
        raise ConfigError("calibrate_lvd_plus_lambda requires gradient_ratio, not fixed lambda")
    _validate_after_overrides(config)
    _preflight(config)
    from sft_conflict.s0_workflow import require_s0_cache

    require_s0_cache(config)
    bundle = load_model_bundle(
        config.model, config.lora, config.training, cache_dir=_cache_paths(config)[0]
    )
    target, anchor, _, _ = _load_datasets(
        config, bundle.tokenizer, include_eval=False, include_validation=False
    )
    paths = prepare_run(config, bundle.audit, allow_existing=_world_size() > 1)
    trainer = _build_trainer(config, bundle, target, anchor, None, paths, with_reference=False)
    if _is_world_process_zero():
        preview = build_preflight_preview(
            config, bundle.audit, bundle.tokenizer, target, anchor, None, sample_count=3
        )
        print(render_preflight_preview(preview), flush=True)
        write_json_atomic(paths.result_dir / "preflight-preview.json", preview)
    filename = "lvd-plus-scale-recording.json" if scales else "lvd-plus-lambda-calibration.json"
    result = (
        trainer.record_lvd_plus_scales(paths.result_dir / filename)
        if scales
        else trainer.calibrate_lvd_plus_lambda(paths.result_dir / filename)
    )
    if trainer.optimizer is not None:
        raise RuntimeError("preparation unexpectedly created an optimizer")
    if _is_world_process_zero():
        dump_config(config, paths.result_dir / "resolved-config.yaml")
        write_json_atomic(paths.result_dir / "summary.json", result)
    return {"run_id": paths.run_id, "result_dir": str(paths.result_dir), **result}


def _evaluate(config: ExperimentConfig) -> dict[str, Any]:
    if not config.evaluation.eval_checkpoint:
        raise ConfigError("evaluation requires evaluation.eval_checkpoint or --eval-checkpoint")
    _preflight(config)
    bundle = load_model_bundle(
        config.model,
        config.lora,
        config.training,
        cache_dir=_cache_paths(config)[0],
    )
    target, anchor, validation, evaluation = _load_datasets(
        config,
        bundle.tokenizer,
        include_eval=True,
    )
    paths = prepare_run(config, bundle.audit, allow_existing=_world_size() > 1)
    trainer = _build_trainer(
        config,
        bundle,
        target,
        anchor,
        evaluation,
        paths,
        with_reference=False,
    )
    trainer.load_checkpoint_for_evaluation(config.evaluation.eval_checkpoint)
    step_zero = _step_zero_audit(trainer)
    _write_run_provenance(
        config,
        bundle,
        target,
        anchor,
        validation,
        evaluation,
        paths,
        step_zero,
    )
    metrics = trainer.evaluate()
    if trainer.is_world_process_zero():
        write_json_atomic(paths.result_dir / "evaluation.json", metrics)
    return {"run_id": paths.run_id, "result_dir": str(paths.result_dir), "evaluation": metrics}


def _audit_model(
    config: ExperimentConfig,
    *,
    lvd_audit_max_positions: int,
    lvd_audit_vocabulary_chunk_size: int,
) -> dict[str, Any]:
    _preflight(config)
    bundle = load_model_bundle(
        config.model,
        config.lora,
        config.training,
        cache_dir=_cache_paths(config)[0],
    )
    target, _, _, _ = _load_datasets(config, bundle.tokenizer, include_eval=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle.model.to(device)
    collator = CausalLMCollator(bundle.tokenizer.pad_token_id, config.data.pad_to_multiple_of)
    batch = collator([target[0]])
    model_inputs = {key: value.to(device) for key, value in model_tensor_inputs(batch).items()}
    forward_audit = audit_unembedding_forward(bundle.model, model_inputs)
    valid_positions = model_inputs["attention_mask"][0].nonzero(as_tuple=False).flatten()
    if not len(valid_positions):
        raise ValueError("LVD geometry audit found no valid token positions")
    position_count = min(lvd_audit_max_positions, len(valid_positions))
    selected_offsets = (
        torch.linspace(
            0,
            len(valid_positions) - 1,
            steps=position_count,
            device=device,
        )
        .round()
        .long()
    )
    positions = valid_positions[selected_offsets].unsqueeze(0)
    was_training = bundle.model.training
    bundle.model.eval()
    try:
        with torch.no_grad(), UnembeddingInputCapture(bundle.model, positions=positions) as capture:
            outputs = bundle.model(**model_inputs)
    finally:
        bundle.model.train(was_training)
    if capture.hidden_states is None:
        raise RuntimeError("LVD geometry audit did not capture the Unembedding input")
    output_embeddings = get_unembedding(bundle.model)
    bias = getattr(output_embeddings, "bias", None)
    tolerance = 1e-5 if config.training.mixed_precision == "fp32" else 2e-3
    partition_audit = audit_lvd_partition(
        capture.hidden_states.squeeze(0),
        output_embeddings.weight,
        bias,
        vocabulary_chunk_size=lvd_audit_vocabulary_chunk_size,
        normal_epsilon=config.lvd.normal_epsilon,
        absolute_tolerance=tolerance,
        relative_tolerance=tolerance,
    )
    selected_logits = outputs.logits[0, positions.squeeze(0)]
    model_output_ids = tuple(int(value) for value in selected_logits.argmax(dim=-1).tolist())
    model_output_matches = model_output_ids == partition_audit.unembedding_token_ids
    report = {
        **asdict(partition_audit),
        "positions": tuple(int(value) for value in positions.squeeze(0).tolist()),
        "model_output_token_ids": model_output_ids,
        "model_output_token_ids_match": model_output_matches,
        "model_output_tokens": tuple(
            str(value) for value in bundle.tokenizer.convert_ids_to_tokens(list(model_output_ids))
        ),
        "unembedding_tokens": tuple(
            str(value)
            for value in bundle.tokenizer.convert_ids_to_tokens(
                list(partition_audit.unembedding_token_ids)
            )
        ),
        "lvd_cell_tokens": tuple(
            str(value)
            for value in bundle.tokenizer.convert_ids_to_tokens(
                list(partition_audit.lvd_cell_token_ids)
            )
        ),
    }
    report["passed"] = bool(
        partition_audit.token_ids_match
        and partition_audit.all_points_inside_lvd_cells
        and partition_audit.boundary_tie_count == 0
        and partition_audit.power_identity_matches
        and model_output_matches
    )
    audit_name = config.paths.run_name or datetime.now(timezone.utc).strftime(
        f"lvd-geometry-audit-{Path(config.model.model_id).name}-%Y%m%dT%H%M%S-%fZ"
    )
    report_path = (
        Path(config.paths.artifact_root) / "results" / audit_name / "lvd-geometry-audit.json"
    )
    result = {
        "model": asdict(bundle.audit),
        "unembedding_forward": asdict(forward_audit),
        "lvd_partition": report,
        "report_path": str(report_path),
    }
    write_json_atomic(report_path, result)
    if not report["passed"]:
        raise RuntimeError(
            f"Unembedding/LVD convex-cell audit failed; report saved to {report_path}: {report}"
        )
    return result


def _merge_lora(config: ExperimentConfig, adapter_path: str, output_path: str) -> dict[str, Any]:
    if config.model.training_mode is not TrainingMode.LORA:
        raise ConfigError("merge-lora requires model.training_mode=lora")
    if _world_size() != 1:
        raise ConfigError("merge-lora is a single-process export command")
    _preflight(config)
    hub_cache, dataset_cache = _cache_paths(config)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.model_id,
        revision=config.model.tokenizer_revision,
        cache_dir=str(hub_cache),
        trust_remote_code=config.model.trust_remote_code,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer has neither a pad token nor an EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        config.model.model_id,
        revision=config.model.revision,
        cache_dir=str(hub_cache),
        trust_remote_code=config.model.trust_remote_code,
        local_files_only=True,
        torch_dtype=torch_dtype_for_precision(config.training.mixed_precision),
    )
    sample_dataset = load_tokenized_dataset(
        config.data.target,
        tokenizer,
        max_length=config.data.max_length,
        model_variant=config.model.model_variant,
        cache_dir=dataset_cache,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model.to(device)
    sample_batch = CausalLMCollator(tokenizer.pad_token_id, None)([sample_dataset[0]])
    model_inputs = {
        key: value.to(device) for key, value in model_tensor_inputs(sample_batch).items()
    }
    tolerance = 1e-5 if config.training.mixed_precision == "fp32" else 2e-3
    merged_model, audit = load_and_merge_lora(
        base_model,
        adapter_path,
        model_inputs,
        absolute_tolerance=tolerance,
        relative_tolerance=tolerance,
    )
    destination = Path(output_path).expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"merged export directory is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(destination, safe_serialization=True)
    tokenizer.save_pretrained(destination)
    write_json_atomic(destination / "merge-audit.json", asdict(audit))
    return {"output": str(destination), "merge_audit": asdict(audit)}


def preset_settings_from_args(args: argparse.Namespace) -> QuickRunSettings:
    return QuickRunSettings(
        action=args.action,
        model=args.model,
        target_dataset=args.target_dataset,
        anchor_dataset=args.anchor_dataset,
        method=args.method,
        supervision=args.supervision,
        training_mode=args.training_mode,
        distributed_strategy=args.distributed_strategy,
        mixed_precision=args.mixed_precision,
        optimizer=args.optimizer,
        max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_length=args.max_length,
        target_max_samples=args.target_max_samples,
        target_validation_max_samples=args.target_validation_max_samples,
        anchor_max_samples=args.anchor_max_samples,
        anchor_metadata_path=args.anchor_metadata_path,
        evaluation_max_samples=args.evaluation_max_samples,
        evaluation_batch_size=args.evaluation_batch_size,
        target_batch_size=args.target_batch_size,
        anchor_batch_size=args.anchor_batch_size,
        anchor_sampling_strategy=args.anchor_sampling_strategy,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        recovery_save_steps=args.recovery_save_steps,
        fsdp_sync_each_batch=args.fsdp_sync_each_batch,
        fsdp_cpu_offload=args.fsdp_cpu_offload,
        fsdp_keep_low_precision_params=args.fsdp_keep_low_precision_params,
        gradient_checkpointing=args.gradient_checkpointing,
        run_after_training=args.run_after_training,
        training_validation=args.training_validation,
        best_model_metric=args.best_model_metric,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        anchor_weight=args.anchor_weight,
        kl_temperature=args.kl_temperature,
        seed=args.seed,
        artifact_root=args.artifact_root,
        run_name=args.run_name,
        lvd_config_path=args.lvd_config,
        lvd_direction=args.lvd_direction,
        lvd_position_strategy=args.lvd_position_strategy,
        lvd_positions_per_sample=args.lvd_positions_per_sample,
        anchor_d0_cache_path=args.anchor_d0_cache_path,
        lvd_calibration_checkpoint=args.lvd_calibration_checkpoint,
        lvd_use_capacity_weights=args.lvd_use_capacity_weights,
        lvd_plus_config_path=getattr(args, "lvd_plus_config", str(DEFAULT_LVD_PLUS_CONFIG_PATH)),
        lvd_plus_direction=getattr(args, "lvd_plus_direction", None),
        filtering_policy=getattr(args, "filtering_policy", LABEL_COVERAGE_POLICY),
        s0_config_path=getattr(args, "s0_config", str(DEFAULT_S0_CONFIG_PATH)),
        s0_cache_path=getattr(args, "s0_cache_path", None),
        lvd_plus_scale_cache_path=getattr(args, "lvd_plus_scale_cache_path", None),
        lvd_plus_lambda_cache_path=getattr(args, "lvd_plus_lambda_cache_path", None),
        anchor_geometry_config_path=getattr(
            args,
            "anchor_geometry_config",
            str(DEFAULT_ANCHOR_GEOMETRY_CONFIG_PATH),
        ),
        anchor_geometry_reference_path=getattr(args, "anchor_geometry_reference_path", None),
    )


def _normalize_tokenizer_padding(tokenizer):
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer has neither a pad token nor an EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def _load_plain_tokenizer(config: ExperimentConfig):
    hub_cache, _ = _cache_paths(config)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.model_id,
        revision=config.model.tokenizer_revision,
        cache_dir=str(hub_cache),
        trust_remote_code=config.model.trust_remote_code,
        local_files_only=True,
    )
    return _normalize_tokenizer_padding(tokenizer)


def _diagnostic_checkpoint_config(
    config: ExperimentConfig,
    checkpoint: str | None,
) -> tuple[Path, ExperimentConfig]:
    if not checkpoint:
        raise ConfigError(f"method={config.method.name.value} requires --eval-checkpoint")
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    checkpoint_config = load_config(_checkpoint_resolved_config(checkpoint_path))
    identity_fields = (
        "model_id",
        "revision",
        "tokenizer_revision",
        "model_variant",
        "trust_remote_code",
    )
    mismatches = {
        name: (getattr(checkpoint_config.model, name), getattr(config.model, name))
        for name in identity_fields
        if getattr(checkpoint_config.model, name) != getattr(config.model, name)
    }
    if mismatches:
        raise ConfigError(f"diagnostic checkpoint base-model identity mismatch: {mismatches}")
    if checkpoint_config.data.max_length != config.data.max_length:
        raise ConfigError(
            "diagnostic checkpoint max_length differs from the current Anchor tokenization"
        )
    return checkpoint_path, checkpoint_config


def _load_diagnostic_anchor(config: ExperimentConfig, tokenizer):
    if config.data.anchor is None:
        raise ConfigError(f"method={config.method.name.value} requires an Anchor dataset")
    return load_tokenized_dataset(
        config.data.anchor,
        tokenizer,
        max_length=config.data.max_length,
        model_variant=config.model.model_variant,
        cache_dir=_cache_paths(config)[1],
        progress_description=("Anchor dataset" if _is_world_process_zero() else None),
    )


def _load_base_diagnostic_model(config: ExperimentConfig):
    model_kwargs: dict[str, Any] = {
        "revision": config.model.revision,
        "cache_dir": str(_cache_paths(config)[0]),
        "local_files_only": True,
        "trust_remote_code": config.model.trust_remote_code,
        "torch_dtype": torch_dtype_for_precision(config.training.mixed_precision),
    }
    if config.model.use_flash_attention_2:
        model_kwargs["attn_implementation"] = "flash_attention_2"
    return AutoModelForCausalLM.from_pretrained(config.model.model_id, **model_kwargs)


def _clear_diagnostic_model_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _test_interference(config: ExperimentConfig, args: argparse.Namespace) -> dict[str, Any]:
    if _world_size() != 1:
        raise ConfigError("test_interference is a single-process workflow")
    if not config.lvd.d0_cache_path:
        raise ConfigError("test_interference requires ANCHOR_D0_CACHE_PATH/--anchor-d0-cache-path")
    _preflight(config)
    checkpoint_path, checkpoint_config = _diagnostic_checkpoint_config(
        config,
        args.eval_checkpoint,
    )
    tuned_model, tokenizer = _load_native_evaluation_model(
        checkpoint_config,
        str(checkpoint_path),
    )
    anchor = _load_diagnostic_anchor(config, tokenizer)
    reference_cache = ReferenceCache(config.lvd.d0_cache_path)
    reference_cache.validate_integrity()
    if reference_cache.manifest["filter_mode"] != ReferenceFilterMode.ALL_REFERENCE.value:
        raise ConfigError(
            "test_interference requires a d0 cache recorded with d0_filter_mode=all_reference "
            "so every selected Anchor position is represented"
        )
    position_config = reference_cache.manifest["position_config"]
    selector = AnchorPositionSelector(
        AnchorPositionConfig(
            strategy=AnchorPositionStrategy(position_config["strategy"]),
            positions_per_sample=int(position_config["positions_per_sample"]),
            seed=int(position_config["seed"]),
            exclude_eos=bool(position_config["exclude_eos"]),
            exclude_pad=bool(position_config["exclude_pad"]),
        ),
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    positioned_anchor = AnchorPositionDataset(anchor, selector)
    reference_cache.validate_binding(
        model_id=config.model.model_id,
        model_revision=config.model.revision,
        vocabulary_size=int(reference_cache.manifest["vocabulary_size"]),
        dataset_fingerprint=positioned_anchor.fingerprint,
        position_config=selector.config.to_dict(),
        normal_epsilon=float(reference_cache.manifest["normal_epsilon"]),
        filter_mode=str(reference_cache.manifest["filter_mode"]),
        storage_dtype=str(reference_cache.manifest["storage_dtype"]),
        minimum_confidence=reference_cache.manifest["minimum_confidence"],
        minimum_margin=reference_cache.manifest["minimum_margin"],
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tuned_model.to(device)
    collection = collect_cached_anchor_displacements(
        tuned_model,
        positioned_anchor,
        reference_cache,
        pad_token_id=tokenizer.pad_token_id,
        batch_size=config.data.anchor_batch_size,
        device=device,
    )
    if collection.position_count != int(reference_cache.manifest["position_count"]):
        raise RuntimeError(
            "interference scan position count differs from the all-reference d0 cache"
        )
    del tuned_model
    _clear_diagnostic_model_memory()

    base_model = _load_base_diagnostic_model(config)
    base_unembedding = get_unembedding(base_model).weight.detach().float().cpu().contiguous()
    del base_model
    _clear_diagnostic_model_memory()
    if base_unembedding.shape[0] != int(reference_cache.manifest["vocabulary_size"]):
        raise RuntimeError("base Unembedding vocabulary differs from the d0 cache")
    statistics = token_direction_movement_statistics(
        collection.displacements.to(device),
        base_unembedding,
        token_chunk_size=args.interference_token_chunk_size,
        position_chunk_size=args.interference_position_chunk_size,
        normal_epsilon=float(reference_cache.manifest["normal_epsilon"]),
    )
    token_labels = tuple(
        str(value)
        for value in tokenizer.convert_ids_to_tokens(list(range(base_unembedding.shape[0])))
    )
    output_dir = Path(
        args.interference_output_dir
        or Path(config.paths.artifact_root)
        / "results"
        / str(config.paths.run_name)
        / "test-interference"
    )
    metadata = {
        "action": "test_interference",
        "checkpoint": str(checkpoint_path),
        "trained_method": checkpoint_config.method.name.value,
        "model_id": config.model.model_id,
        "model_revision": config.model.revision,
        "anchor_dataset_fingerprint": positioned_anchor.fingerprint,
        "d0_cache_path": str(reference_cache.root),
        "d0_schema_version": reference_cache.manifest["schema_version"],
        "d0_storage_dtype": reference_cache.manifest["storage_dtype"],
        "direction_basis": "unit rows of the base-model Unembedding",
    }
    paths = write_interference_artifacts(
        statistics,
        token_labels=token_labels,
        output_dir=output_dir,
        metadata=metadata,
    )
    ranking = torch.argsort(statistics.mean_positive_movement, descending=True)
    top_tokens = [
        {
            "rank": rank,
            "token_id": int(token_id),
            "token": token_labels[int(token_id)],
            "mean_positive_movement": float(
                statistics.mean_positive_movement[int(token_id)].item()
            ),
        }
        for rank, token_id in enumerate(ranking[:20].tolist(), start=1)
    ]
    return {
        **metadata,
        "sample_count": collection.sample_count,
        "position_count": statistics.position_count,
        "vocabulary_size": len(token_labels),
        "total_positive_movement": statistics.total_positive_movement,
        "total_absolute_movement": statistics.total_absolute_movement,
        "mean_hidden_l2_displacement": statistics.mean_hidden_l2_displacement,
        "top_tokens": top_tokens,
        "artifacts": paths,
    }


def _test_crossing(config: ExperimentConfig, args: argparse.Namespace) -> dict[str, Any]:
    if _world_size() != 1:
        raise ConfigError("test_crossing is a single-process workflow")
    _preflight(config)
    checkpoint_path, checkpoint_config = _diagnostic_checkpoint_config(
        config,
        args.eval_checkpoint,
    )
    tokenizer = _load_plain_tokenizer(config)
    anchor = _load_diagnostic_anchor(config, tokenizer)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_model = _load_base_diagnostic_model(config)
    base_model.to(device)
    base = collect_supervised_next_token_predictions(
        base_model,
        anchor,
        pad_token_id=tokenizer.pad_token_id,
        batch_size=config.data.anchor_batch_size,
        device=device,
    )
    del base_model
    _clear_diagnostic_model_memory()

    tuned_model, _ = _load_native_evaluation_model(checkpoint_config, str(checkpoint_path))
    tuned_model.to(device)
    tuned = collect_supervised_next_token_predictions(
        tuned_model,
        anchor,
        pad_token_id=tokenizer.pad_token_id,
        batch_size=config.data.anchor_batch_size,
        device=device,
    )
    del tuned_model
    _clear_diagnostic_model_memory()
    if not torch.equal(base.target_token_ids, tuned.target_token_ids):
        raise RuntimeError("base and tuned crossing scans produced different target-token streams")
    statistics = crossing_statistics(
        base.predictions,
        tuned.predictions,
        base.target_token_ids,
    )
    report = asdict(statistics)
    report.update(
        {
            "action": "test_crossing",
            "checkpoint": str(checkpoint_path),
            "trained_method": checkpoint_config.method.name.value,
            "model_id": config.model.model_id,
            "model_revision": config.model.revision,
            "anchor_sample_count": len(anchor),
            "crossing_definition": "model argmax token differs from the supervised label token",
            "new_crossing_rate_among_base_correct": (
                statistics.new_crossing_count / statistics.base_correct_count
                if statistics.base_correct_count
                else 0.0
            ),
            "prediction_flip_rate": statistics.prediction_flip_count / statistics.token_count,
        }
    )
    return report


def _build_reference_cache(config: ExperimentConfig, args: argparse.Namespace) -> dict[str, Any]:
    if config.data.anchor is None:
        raise ConfigError("Anchor d0 recording requires an Anchor dataset")
    if _world_size() != 1:
        raise ConfigError("Anchor d0 recording is a single-process action")
    _preflight(config)
    tokenizer = _load_plain_tokenizer(config)
    _, dataset_cache = _cache_paths(config)
    anchor = load_tokenized_dataset(
        config.data.anchor,
        tokenizer,
        max_length=config.data.max_length,
        model_variant=config.model.model_variant,
        cache_dir=dataset_cache,
        progress_description=("Anchor dataset" if _is_world_process_zero() else None),
    )
    reference = load_reference_model(
        type(config.reference)(
            mode=ReferenceMode.ONLINE,
            model_id=config.model.model_id,
            revision=config.model.revision,
        ),
        config.model,
        config.training,
        cache_dir=_cache_paths(config)[0],
    )
    if reference is None:
        raise RuntimeError("reference model loader returned no model")
    record_d0 = config.method.name is MethodName.RECORD_ANCHOR_D0
    position_strategy = config.lvd.position_strategy if record_d0 else args.anchor_position_strategy
    positions_per_sample = (
        config.lvd.positions_per_sample if record_d0 else args.anchor_positions_per_sample
    )
    position_seed = config.lvd.position_seed if record_d0 else config.training.seed
    storage_dtype = config.lvd.d0_storage_dtype if record_d0 else args.reference_storage_dtype
    samples_per_shard = (
        config.lvd.d0_samples_per_shard if record_d0 else args.reference_samples_per_shard
    )
    filter_mode = config.lvd.d0_filter_mode if record_d0 else args.reference_filter_mode
    cache_name = (
        f"all-token-d0-v5-{Path(config.model.model_id).name}-{config.data.anchor.adapter}-"
        f"{position_strategy}-seed-{position_seed}"
    )
    output_dir = (config.lvd.d0_cache_path if record_d0 else args.reference_cache_dir) or str(
        Path(config.paths.artifact_root) / "reference_cache" / cache_name
    )
    precision_budget = None
    if storage_dtype != "float32":
        precision_budget = ReferencePrecisionBudget(
            hidden_max_absolute_error=args.hidden_max_absolute_error,
            hidden_mean_absolute_error=args.hidden_mean_absolute_error,
            minimum_argmax_agreement=args.minimum_argmax_agreement,
            margin_max_absolute_error=args.margin_max_absolute_error,
            margin_max_relative_error=args.margin_max_relative_error,
            maximum_margin_sign_flip_rate=args.maximum_margin_sign_flip_rate,
        )
    selector = AnchorPositionSelector(
        AnchorPositionConfig(
            strategy=AnchorPositionStrategy(position_strategy),
            positions_per_sample=positions_per_sample,
            seed=position_seed,
        ),
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    anchor = AnchorPositionDataset(anchor, selector)
    cache_config = ReferenceCacheConfig(
        output_dir=output_dir,
        samples_per_shard=samples_per_shard,
        storage_dtype=storage_dtype,
        filter_mode=ReferenceFilterMode(filter_mode),
        minimum_confidence=(
            config.lvd.d0_minimum_confidence if record_d0 else args.minimum_reference_confidence
        ),
        minimum_margin=(
            config.lvd.d0_minimum_margin if record_d0 else args.minimum_reference_margin
        ),
        precision_budget=precision_budget,
        normal_epsilon=(config.lvd.normal_epsilon if record_d0 else 1e-12),
    )
    hidden_size = int(reference.config.hidden_size)
    estimated_bytes = estimate_reference_cache_bytes(
        sample_count=len(anchor),
        positions_per_sample=positions_per_sample,
        hidden_size=hidden_size,
        vocabulary_size=int(get_unembedding(reference).weight.shape[0]),
        storage_dtype=storage_dtype,
    )
    manifest = ReferenceCacheBuilder(
        model=reference,
        tokenizer=tokenizer,
        dataset=anchor,
        selector=selector,
        config=cache_config,
        model_id=config.model.model_id,
        model_revision=config.model.revision,
        dataset_config=anchor.manifest,
    ).build()
    return {
        "action": "record_anchor_d0" if record_d0 else "build-reference-cache",
        "output_dir": output_dir,
        "estimated_unfiltered_bytes": estimated_bytes,
        "manifest": manifest,
    }


def _load_native_evaluation_model(
    config: ExperimentConfig,
    checkpoint: str | None,
    *,
    trainable_adapter: bool = False,
):
    if checkpoint is None:
        bundle = load_model_bundle(
            config.model,
            config.lora,
            config.training,
            cache_dir=_cache_paths(config)[0],
        )
        return bundle.model, bundle.tokenizer

    checkpoint_path = Path(checkpoint).expanduser().resolve()
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(f"evaluation checkpoint is missing: {checkpoint_path}")
    tokenizer_source = (
        checkpoint_path if (checkpoint_path / "tokenizer_config.json").is_file() else None
    )
    tokenizer = (
        AutoTokenizer.from_pretrained(tokenizer_source, local_files_only=True)
        if tokenizer_source is not None
        else _load_plain_tokenizer(config)
    )
    tokenizer = _normalize_tokenizer_padding(tokenizer)
    dtype = torch_dtype_for_precision(config.training.mixed_precision)
    if (checkpoint_path / "adapter_config.json").is_file():
        from peft import PeftModel

        model_kwargs: dict[str, Any] = {
            "revision": config.model.revision,
            "cache_dir": str(_cache_paths(config)[0]),
            "local_files_only": True,
            "trust_remote_code": config.model.trust_remote_code,
            "torch_dtype": dtype,
        }
        if config.model.use_flash_attention_2:
            model_kwargs["attn_implementation"] = "flash_attention_2"
        base_model = AutoModelForCausalLM.from_pretrained(
            config.model.model_id,
            **model_kwargs,
        )
        if config.model.use_flash_attention_2 and (
            getattr(base_model.config, "_attn_implementation", None) != "flash_attention_2"
        ):
            raise RuntimeError(
                "Flash Attention 2 was requested but the pilot base model did not activate it"
            )
        model = PeftModel.from_pretrained(
            base_model,
            checkpoint_path,
            local_files_only=True,
            is_trainable=trainable_adapter,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            local_files_only=True,
            torch_dtype=dtype,
        )
    model.to(dtype=dtype)
    return model, tokenizer


def _native_evaluate(config: ExperimentConfig, args: argparse.Namespace) -> dict[str, Any]:
    if _world_size() != 1:
        raise ConfigError("native-evaluate is a single-process action")
    _preflight(config)
    evaluation_config = config.evaluation.target or config.data.target
    dataset = load_logical_dataset(evaluation_config, cache_dir=_cache_paths(config)[1])
    model, tokenizer = _load_native_evaluation_model(config, args.eval_checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        print(
            f"Native evaluation uses one GPU (cuda:{torch.cuda.current_device()}); "
            "additional visible GPUs are not used. The training FSDP/DDP setting "
            "does not shard native evaluation.",
            flush=True,
        )
    evaluator = build_native_evaluator(
        args.target_dataset,
        max_input_length=config.data.max_length,
        max_new_tokens=args.native_max_new_tokens,
        stop_tokens=tuple(args.native_stop_token),
        allow_code_execution=args.allow_code_execution,
        humaneval_samples_per_problem=args.humaneval_samples_per_problem,
        humaneval_temperature=args.humaneval_temperature,
        humaneval_top_p=args.humaneval_top_p,
        humaneval_timeout_seconds=args.humaneval_timeout_seconds,
        humaneval_execution_workers=args.humaneval_execution_workers,
        generator_name=str(config.paths.run_name),
    )
    tensor_preview = build_random_logical_tensor_preview(
        dataset,
        tokenizer,
        stage="native evaluation",
        max_length=evaluator.protocol.max_input_length,
        model_variant=config.model.model_variant,
        evaluator_type=evaluator.protocol.evaluator_type,
        pad_to_multiple_of=config.data.pad_to_multiple_of,
    )
    print(render_tensor_preview(tensor_preview), flush=True)
    batch_count = math.ceil(len(dataset) / config.evaluation.batch_size)
    print(
        "Native evaluation started: "
        f"task={evaluator.protocol.task_name}, samples={len(dataset)}, "
        f"batch_size={config.evaluation.batch_size}, batches={batch_count}, "
        f"max_input_length={evaluator.protocol.max_input_length}, "
        f"max_new_tokens={evaluator.protocol.max_new_tokens}, "
        f"checkpoint={args.eval_checkpoint or 'base'}, device={device}",
        flush=True,
    )
    result = evaluator.evaluate(
        model,
        tokenizer,
        dataset,
        batch_size=config.evaluation.batch_size,
    )
    output_dir = args.native_evaluation_dir or str(
        Path(config.paths.artifact_root)
        / "results"
        / str(config.paths.run_name)
        / "native-evaluation"
    )
    paths = write_evaluation_result(result, output_dir)
    official_scoring = None
    if evaluator.protocol.task_name == "alpaca_eval":
        generator_name = str(config.paths.run_name)
        model_outputs = paths["model_outputs"]
        paths["scoring_instructions"] = write_alpaca_eval_scoring_instructions(
            model_outputs,
            artifact_root=config.paths.artifact_root,
            generator_name=generator_name,
            output_dir=output_dir,
        )
        if args.alpaca_eval_run_judge:
            official_scoring = run_official_alpaca_eval(
                model_outputs,
                artifact_root=config.paths.artifact_root,
                annotators_config=args.alpaca_eval_annotators_config,
                generator_name=generator_name,
                output_dir=Path(output_dir) / args.alpaca_eval_annotators_config,
                auto_prepare=args.auto_prepare_cache,
            )
    elif args.alpaca_eval_run_judge:
        raise ConfigError("--alpaca-eval-run-judge requires target_dataset=alpaca_eval")
    response = {
        "action": "native-evaluate",
        "checkpoint": args.eval_checkpoint or "base",
        "output_dir": output_dir,
        "aggregate": result.aggregate,
        "artifacts": paths,
    }
    if official_scoring is not None:
        response["official_scoring"] = official_scoring
    return response


def _audit_data(config: ExperimentConfig, args: argparse.Namespace) -> dict[str, Any]:
    _enforce_offline_environment(config)
    _, dataset_cache = _cache_paths(config)
    datasets = {
        "target_train": load_logical_dataset(config.data.target, cache_dir=dataset_cache),
        "target_evaluation": load_logical_dataset(
            config.evaluation.target or config.data.target,
            cache_dir=dataset_cache,
        ),
    }
    if config.data.anchor is not None:
        datasets["anchor_train"] = load_logical_dataset(
            config.data.anchor,
            cache_dir=dataset_cache,
        )
    report = audit_dataset_leakage(
        datasets,
        LeakageAuditConfig(near_duplicate_threshold=args.near_duplicate_threshold),
    )
    output_path = args.data_audit_output or str(
        Path(config.paths.artifact_root)
        / "results"
        / str(config.paths.run_name)
        / "data-leakage-audit.json"
    )
    write_leakage_report(report, output_path)
    return {"action": "audit-data", "output": output_path, "report": report}


def _filter_anchor(config: ExperimentConfig, args: argparse.Namespace) -> dict[str, Any]:
    # Namespaces from the old internal API remain explicitly legacy-compatible;
    # the CLI and click launcher always supply the new default policy.
    policy = getattr(args, "filtering_policy", LEGACY_FILTER_POLICY)
    if policy == LABEL_COVERAGE_POLICY:
        return prepare_label_coverage_dataset(
            preset_settings_from_args(args), auto_prepare=args.auto_prepare_cache
        )
    if policy != LEGACY_FILTER_POLICY:
        raise ConfigError(f"unsupported filtering policy: {policy}")
    if not 0.0 <= args.filter_threshold <= 1.0:
        raise ConfigError("filter_threshold must be between 0 and 1")
    if config.data.anchor is None:
        raise ConfigError("filtering_anchor requires an Anchor dataset")

    _enforce_offline_environment(config)
    world_size = _world_size()
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not torch.cuda.is_available():
        raise RuntimeError("distributed filtering_anchor requires CUDA")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    if world_size > 1 and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")

    destination = filtered_anchor_dataset_dir(config.paths.artifact_root, args.model)
    score_cache_dir = anchor_filter_score_dir(
        config.paths.artifact_root,
        args.model,
        config.data.max_length,
    )
    cached_score_paths = reusable_filter_score_paths(
        score_dir=score_cache_dir,
        model_name=args.model,
        model_id=config.model.model_id,
        model_revision=config.model.revision,
        max_length=config.data.max_length,
        source_dataset_id=config.data.anchor.path,
        source_dataset_revision=config.data.anchor.revision,
    )
    existing_manifest_path = destination / "manifest.json"
    score_manifest_path = score_cache_dir / "manifest.json"
    if (
        cached_score_paths is not None
        and existing_manifest_path.is_file()
        and score_manifest_path.is_file()
    ):
        existing_manifest = json.loads(existing_manifest_path.read_text(encoding="utf-8"))
        score_manifest = json.loads(score_manifest_path.read_text(encoding="utf-8"))
        existing_files = existing_manifest.get("files", {})
        if (
            existing_manifest.get("model") == score_manifest.get("model")
            and existing_manifest.get("tokenization") == score_manifest.get("tokenization")
            and existing_manifest.get("source_fingerprint")
            == score_manifest.get("source", {}).get("fingerprint")
            and math.isclose(
                float(existing_manifest.get("filter", {}).get("threshold", -1.0)),
                args.filter_threshold,
                abs_tol=1e-12,
            )
            and (destination / str(existing_files.get("dataset", ""))).is_file()
            and (destination / str(existing_files.get("output_token_dict", ""))).is_file()
        ):
            return {
                "action": "filtering-anchor",
                "filter_threshold": args.filter_threshold,
                "output_dataset": existing_manifest["dataset_name"],
                "output_dir": str(destination),
                "manifest": str(existing_manifest_path),
                "source_samples": existing_manifest["filter"]["source_samples"],
                "eligible_samples": existing_manifest["filter"]["eligible_samples"],
                "selected_samples": existing_manifest["cover"]["selected_samples"],
                "covered_token_types": existing_manifest["cover"]["selected_token_types"],
                "reused_existing_dataset": True,
            }
    tokenizer: Any | None = None
    anchor = None
    scores: tuple[AnchorFilterScore, ...] | None = None
    if cached_score_paths is None:
        filter_model_config = replace(
            config.model,
            training_mode=TrainingMode.FULL_FT,
            freeze_unembedding=False,
            untie_tied_embeddings=False,
            gradient_checkpointing=False,
        )
        bundle = load_model_bundle(
            filter_model_config,
            config.lora,
            config.training,
            cache_dir=_cache_paths(config)[0],
        )
        model = bundle.model
        tokenizer = bundle.tokenizer
        anchor_config = replace(config.data.anchor, max_samples=None)
        anchor = load_tokenized_dataset(
            anchor_config,
            tokenizer,
            max_length=config.data.max_length,
            model_variant=config.model.model_variant,
            cache_dir=_cache_paths(config)[1],
            progress_description="Anchor filtering source" if rank == 0 else None,
        )
        model.requires_grad_(False)
        model.to(device)

        score_dir = (
            Path(config.paths.artifact_root)
            / "results"
            / str(config.paths.run_name)
            / "anchor-filtering"
        )
        score_path = score_dir / f"scores.rank-{rank:05d}-of-{world_size:05d}.jsonl"
        score_anchor_shard(
            model=model,
            dataset=anchor,
            collator=CausalLMCollator(
                tokenizer.pad_token_id,
                config.data.pad_to_multiple_of,
            ),
            device=device,
            batch_size=args.filtering_batch_size,
            rank=rank,
            world_size=world_size,
            output_path=score_path,
        )
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        score_paths = tuple(
            score_dir / f"scores.rank-{score_rank:05d}-of-{world_size:05d}.jsonl"
            for score_rank in range(world_size)
        )
        if rank == 0:
            scores = read_filter_scores(score_paths, expected_samples=len(anchor))
            write_filter_score_cache(
                destination=score_cache_dir,
                score_paths=score_paths,
                model_name=args.model,
                model_id=config.model.model_id,
                model_revision=config.model.revision,
                max_length=config.data.max_length,
                source_dataset_id=config.data.anchor.path,
                source_dataset_revision=config.data.anchor.revision,
                source_fingerprint=anchor.fingerprint,
                source_samples=len(anchor),
            )
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        del model, bundle
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    elif rank == 0:
        print(f"Reusing threshold-independent Anchor scores: {score_cache_dir}", flush=True)
        tokenizer = _load_plain_tokenizer(config)
        anchor = load_tokenized_dataset(
            replace(config.data.anchor, max_samples=None),
            tokenizer,
            max_length=config.data.max_length,
            model_variant=config.model.model_variant,
            cache_dir=_cache_paths(config)[1],
            progress_description="Anchor filtering source",
        )
        score_manifest = json.loads((score_cache_dir / "manifest.json").read_text(encoding="utf-8"))
        if (
            score_manifest["source"]["samples"] != len(anchor)
            or score_manifest["source"]["fingerprint"] != anchor.fingerprint
        ):
            raise ValueError("cached Anchor correctness scores do not match the source dataset")
        scores = read_filter_scores(cached_score_paths, expected_samples=len(anchor))

    result: dict[str, Any] = {
        "action": "filtering-anchor",
        "filter_threshold": args.filter_threshold,
        "rank": rank,
    }
    if rank == 0:
        if anchor is None or tokenizer is None or scores is None:
            raise RuntimeError("rank 0 is missing Anchor filtering state")
        eligible_scores: tuple[AnchorFilterScore, ...] = tuple(
            score for score in scores if score.accuracy + 1e-12 >= args.filter_threshold
        )
        eligible_tokens = {
            score.source_index: supervised_output_token_ids(anchor[score.source_index])
            for score in eligible_scores
        }
        selection = greedy_token_cover(eligible_tokens)
        manifest = write_filtered_anchor_artifacts(
            destination=destination,
            dataset=anchor,
            scores=scores,
            threshold=args.filter_threshold,
            selection=selection,
            tokenizer=tokenizer,
            model_name=args.model,
            model_id=config.model.model_id,
            model_revision=config.model.revision,
            max_length=config.data.max_length,
            source_fingerprint=anchor.fingerprint,
        )
        result = {
            "action": "filtering-anchor",
            "filter_threshold": args.filter_threshold,
            "output_dataset": manifest["dataset_name"],
            "output_dir": str(destination),
            "manifest": str(destination / "manifest.json"),
            "source_samples": manifest["filter"]["source_samples"],
            "eligible_samples": manifest["filter"]["eligible_samples"],
            "selected_samples": manifest["cover"]["selected_samples"],
            "covered_token_types": manifest["cover"]["selected_token_types"],
        }
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    return result


def _lm_eval_metric(results: dict[str, Any]) -> tuple[float, dict[str, float]]:
    per_task: dict[str, float] = {}
    preferred = (
        "acc_norm,none",
        "acc,none",
        "exact_match,strict-match",
        "exact_match,none",
        "mc2,none",
        "prompt_level_strict_acc,none",
        "inst_level_strict_acc,none",
    )
    for task_name, metrics in results.get("results", {}).items():
        selected = next(
            (
                float(metrics[name])
                for name in preferred
                if name in metrics and isinstance(metrics[name], int | float)
            ),
            None,
        )
        if selected is None or not math.isfinite(selected):
            raise RuntimeError(f"lm-eval task '{task_name}' has no finite primary metric")
        per_task[task_name] = selected
    if not per_task:
        raise RuntimeError("lm-eval returned no task metrics")
    return sum(per_task.values()) / len(per_task), per_task


def _general_lm_evaluate(
    config: ExperimentConfig,
    *,
    checkpoint: str | None,
    output_dir: Path,
    tasks: tuple[str, ...],
    seed: int,
    python_executable: str,
    harness_path: str,
    batch_size: str,
    num_fewshot: int | None,
    limit: float | None,
    check_integrity: bool,
    auto_prepare: bool,
    prepared_task_cache: dict[str, Any] | None = None,
) -> dict[str, Any]:
    evaluation_dtype = {
        "fp32": "float32",
        "fp16": "float16",
        "bf16": "bfloat16",
    }[config.training.mixed_precision]
    if checkpoint is None:
        model_args = (
            f"pretrained={config.model.model_id},revision={config.model.revision},"
            f"tokenizer={config.model.model_id},dtype={evaluation_dtype}"
        )
    else:
        checkpoint_path = Path(checkpoint).expanduser().resolve()
        if (checkpoint_path / "adapter_config.json").is_file():
            model_args = (
                f"pretrained={config.model.model_id},revision={config.model.revision},"
                f"tokenizer={config.model.model_id},peft={checkpoint_path},"
                f"dtype={evaluation_dtype}"
            )
        else:
            model_args = (
                f"pretrained={checkpoint_path},tokenizer={checkpoint_path},dtype={evaluation_dtype}"
            )
    result = run_lm_eval(
        LmEvalRequest(
            python_executable=python_executable,
            model="hf",
            model_args=model_args,
            tasks=tasks,
            output_path=str(output_dir),
            harness_path=harness_path,
            batch_size=batch_size,
            num_fewshot=num_fewshot,
            limit=limit,
            seed=seed,
            log_samples=True,
            check_integrity=check_integrity,
            auto_prepare=auto_prepare,
        ),
        prepared_task_cache=prepared_task_cache,
    )
    score, per_task = _lm_eval_metric(result["results"])
    return {
        "score": score,
        "per_task": per_task,
        "tasks": list(tasks),
        "result_path": result["result_path"],
        "metadata_path": result["metadata_path"],
        "harness_revision": result["harness_revision"],
    }


def _release_native_evaluation_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _native_evaluate_with_llm_evaluation(
    config: ExperimentConfig,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if not args.llm_evaluation:
        return _native_evaluate(config, args)
    if args.llm_evaluation_num_fewshot is not None and args.llm_evaluation_num_fewshot < 0:
        raise ConfigError("--llm-evaluation-num-fewshot must be non-negative")
    if args.llm_evaluation_limit is not None and args.llm_evaluation_limit <= 0:
        raise ConfigError("--llm-evaluation-limit must be positive")

    tasks = tuple(dict.fromkeys(args.llm_evaluation_task or DEFAULT_LLM_EVALUATION_TASKS))
    _enforce_offline_environment(config)
    task_cache = ensure_lm_eval_task_cache(
        LmEvalRequest(
            python_executable=args.lm_eval_python,
            model="hf",
            model_args="pretrained=unused",
            tasks=tasks,
            output_path="unused",
            harness_path=args.lm_eval_harness_path,
            auto_prepare=args.auto_prepare_cache,
        )
    )
    native_result = _native_evaluate(config, args)
    _release_native_evaluation_memory()
    output_dir = Path(str(native_result["output_dir"])).expanduser().resolve().parent
    output_dir /= "llm-evaluation"
    print(
        "LLM evaluation started after native evaluation: "
        f"tasks={','.join(tasks)}, checkpoint={args.eval_checkpoint or 'base'}, "
        f"output={output_dir}",
        flush=True,
    )
    native_result["llm_evaluation"] = _general_lm_evaluate(
        config,
        checkpoint=args.eval_checkpoint,
        output_dir=output_dir,
        tasks=tasks,
        seed=config.training.seed,
        python_executable=args.lm_eval_python,
        harness_path=args.lm_eval_harness_path,
        batch_size=args.llm_evaluation_batch_size,
        num_fewshot=args.llm_evaluation_num_fewshot,
        limit=args.llm_evaluation_limit,
        check_integrity=args.llm_evaluation_check_integrity,
        auto_prepare=args.auto_prepare_cache,
        prepared_task_cache=task_cache,
    )
    return native_result


def _free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(("127.0.0.1", 0))
        return int(server.getsockname()[1])


def _distributed_benchmark_command(
    config: ExperimentConfig,
    config_path: Path,
    *,
    process_count: int,
) -> list[str]:
    if process_count < 2:
        raise ConfigError(
            f"benchmark {config.training.distributed_strategy} execution requires at least "
            "two visible GPUs"
        )
    accelerate = Path(sys.executable).resolve().parent / "accelerate"
    if not accelerate.is_file():
        raise FileNotFoundError(f"Accelerate executable not found: {accelerate}")
    command = [
        str(accelerate),
        "launch",
        "--num_processes",
        str(process_count),
        "--num_machines",
        "1",
        "--mixed_precision",
        (
            "no"
            if config.training.mixed_precision == "fp32"
            or config.training.fsdp_keep_low_precision_params
            else config.training.mixed_precision
        ),
        "--dynamo_backend",
        "no",
        "--main_process_port",
        str(_free_local_port()),
    ]
    if config.training.distributed_strategy == "ddp":
        command.append("--multi_gpu")
    elif config.training.distributed_strategy == "fsdp":
        command.extend(
            [
                "--use_fsdp",
                "--fsdp_sharding_strategy",
                "1",
                "--fsdp_auto_wrap_policy",
                "transformer_based_wrap",
                "--fsdp_use_orig_params",
                str(config.model.training_mode is not TrainingMode.FULL_FT).lower(),
                "--fsdp_state_dict_type",
                "FULL_STATE_DICT",
            ]
        )
        if config.model.gradient_checkpointing:
            command.extend(["--fsdp_activation_checkpointing", "true"])
    else:
        raise ConfigError("distributed benchmark command requires ddp or fsdp")
    command.extend(
        [
            "--module",
            "sft_conflict.cli.main",
            "train",
            "--config",
            str(config_path),
        ]
    )
    return command


def _run_distributed_benchmark_training(
    config: ExperimentConfig,
    *,
    output_dir: Path,
) -> dict[str, Any]:
    run_id = str(config.paths.run_name)
    config_path = output_dir / "configs" / f"{run_id}.yaml"
    dump_config(config, config_path)
    process_count = torch.cuda.device_count()
    command = _distributed_benchmark_command(
        config,
        config_path,
        process_count=process_count,
    )
    subprocess.run(command, check=True, env=os.environ.copy())
    result_dir = Path(config.paths.artifact_root) / "results" / run_id
    final_checkpoint = Path(config.paths.artifact_root) / "checkpoints" / run_id / "final"
    summary_path = result_dir / "summary.json"
    if not summary_path.is_file() or not final_checkpoint.is_dir():
        raise RuntimeError(f"distributed benchmark run did not finish: {run_id}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return {
        "run_id": run_id,
        "result_dir": str(result_dir),
        "final_checkpoint": str(final_checkpoint),
        **summary,
    }


def _benchmark(args: argparse.Namespace, settings: QuickRunSettings) -> dict[str, Any]:
    benchmark_name = settings.run_name or datetime.now(timezone.utc).strftime(
        "m2-benchmark-%Y%m%dT%H%M%S-%fZ"
    )
    settings = replace(settings, run_name=benchmark_name)
    seeds = tuple(args.benchmark_seed or [settings.seed])
    tasks = tuple(args.benchmark_general_task or DEFAULT_GENERAL_TASKS)
    if args.benchmark_formal and args.benchmark_general_limit is not None:
        raise ConfigError("formal benchmark runs cannot use --benchmark-general-limit")
    plan = build_benchmark_plan(
        model=settings.model,
        target_dataset=settings.target_dataset,
        anchor_dataset=settings.anchor_dataset,
        seeds=seeds,
        formal=args.benchmark_formal,
        general_tasks=tasks,
        name_prefix=benchmark_name,
    )
    output_dir = (
        Path(
            args.benchmark_output_dir
            or Path(settings.artifact_root) / "results" / benchmark_name / "benchmark"
        )
        .expanduser()
        .resolve()
    )
    plan_path = write_benchmark_plan(plan, output_dir / "benchmark-plan.json")
    if not args.benchmark_execute:
        return {
            "action": "benchmark",
            "executed": False,
            "plan": str(plan_path),
            "run_count": len(plan["runs"]),
        }
    ensure_preset_cache(settings, auto_prepare=args.auto_prepare_cache)

    def execute(run: dict[str, Any]) -> dict[str, Any]:
        condition_settings = replace(
            settings,
            action="train" if run["training_required"] else "native-evaluate",
            seed=run["seed"],
            run_name=run["run_id"],
            training_mode=run["training_mode"] or settings.training_mode,
            method=run["method"],
            anchor_dataset=run["anchor_dataset"],
            run_after_training=False,
        )
        ensure_preset_cache(condition_settings, auto_prepare=args.auto_prepare_cache)
        condition_config = build_quick_run_config(condition_settings)
        checkpoint = None
        training_result = None
        if run["training_required"]:
            if condition_config.training.distributed_strategy == "single":
                training_result = _train(condition_config)
            else:
                training_result = _run_distributed_benchmark_training(
                    condition_config,
                    output_dir=output_dir,
                )
            checkpoint = training_result["final_checkpoint"]
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        evaluation_args = argparse.Namespace(**vars(args))
        evaluation_args.target_dataset = run["target_dataset"]
        evaluation_args.eval_checkpoint = checkpoint
        evaluation_args.native_evaluation_dir = str(output_dir / "runs" / run["run_id"] / "target")
        target_result = _native_evaluate(condition_config, evaluation_args)
        _release_native_evaluation_memory()
        general_result = _general_lm_evaluate(
            condition_config,
            checkpoint=checkpoint,
            output_dir=output_dir / "runs" / run["run_id"] / "general",
            tasks=tuple(run["general_tasks"]),
            seed=int(run["seed"]),
            python_executable=args.lm_eval_python,
            harness_path=args.lm_eval_harness_path,
            batch_size=args.lm_eval_batch_size,
            num_fewshot=args.benchmark_num_fewshot,
            limit=args.benchmark_general_limit,
            check_integrity=args.benchmark_check_integrity,
            auto_prepare=args.auto_prepare_cache,
        )
        target_metric_name = str(target_result["aggregate"]["metric_name"])
        observation = {
            "condition": run["condition"],
            "seed": run["seed"],
            "target_score": float(target_result["aggregate"][target_metric_name]),
            "general_score": float(general_result["score"]),
            "target_metric_name": target_metric_name,
            "general_metric_name": "mean_primary_metric",
            "checkpoint": checkpoint or "base",
        }
        return {
            "observation": observation,
            "training": training_result,
            "target_evaluation": target_result,
            "general_evaluation": general_result,
        }

    completed = BenchmarkOrchestrator(
        plan,
        output_dir / "benchmark-state.json",
    ).run(execute)
    observations = [entry["result"]["observation"] for entry in completed]
    target_metric_names = {row["target_metric_name"] for row in observations}
    if len(target_metric_names) != 1:
        raise RuntimeError(f"benchmark target metrics differ: {sorted(target_metric_names)}")
    aggregated = aggregate_benchmark_results(
        observations,
        target_metric_name=target_metric_names.pop(),
        general_metric_name="mean_primary_metric",
    )
    artifacts = write_benchmark_results(aggregated, output_dir)
    if args.benchmark_formal and not aggregated["acceptance"]["target_learning_demonstrated"]:
        raise RuntimeError(
            "formal benchmark did not meet the preregistered target-learning threshold; "
            f"results were preserved at {artifacts['json']}"
        )
    return {
        "action": "benchmark",
        "executed": True,
        "plan": str(plan_path),
        "state": str(output_dir / "benchmark-state.json"),
        "artifacts": artifacts,
        "summary": aggregated["summary"],
    }


def _preset_run(args: argparse.Namespace) -> dict[str, Any]:
    settings = preset_settings_from_args(args)
    if args.action == "generate-pilot" and args.resume_from_checkpoint:
        raise ConfigError("generate-pilot does not support --resume-from-checkpoint")
    if args.action == "benchmark":
        return _benchmark(args, settings)
    if args.action == "filtering-anchor" and settings.filtering_policy == LABEL_COVERAGE_POLICY:
        # No experiment model/config construction, reference YAML, or CUDA setup.
        return prepare_label_coverage_dataset(settings, auto_prepare=args.auto_prepare_cache)
    if args.action == "create-anchor-dataset":
        # CPU selection over the fixed SMART submixtures plus an optional single-GPU
        # accuracy report; no experiment config, reference YAML or launcher is used.
        from sft_conflict.anchor_creation_workflow import (
            AnchorCreationOptions,
            create_anchor_dataset,
        )

        options = AnchorCreationOptions(
            seed=args.anchor_creation_seed,
            workers=args.anchor_creation_workers,
            scoring_batch_size=args.anchor_creation_scoring_batch_size,
            compute_accuracy=args.anchor_creation_compute_accuracy,
            submixtures=tuple(args.anchor_creation_submixture or SUBMIXTURE_KEYS),
            max_samples_per_task=args.anchor_creation_max_samples_per_task,
            guarantee_task_coverage=args.anchor_creation_guarantee_task_coverage,
        )
        return create_anchor_dataset(settings, options, auto_prepare=args.auto_prepare_cache)
    ensure_preset_cache(settings, auto_prepare=args.auto_prepare_cache)
    config = build_quick_run_config(settings)
    if args.action == "prepare-anchor-geometry-reference":
        if args.resume_from_checkpoint:
            raise ConfigError("Base output preparation does not support checkpoint resume")
        return _prepare_anchor_geometry_reference(config)
    if args.action in {"record-lvd-plus-scales", "calibrate-lvd-plus-lambda"}:
        if args.resume_from_checkpoint:
            raise ConfigError("LVD+ preparation actions do not accept --resume-from-checkpoint")
        return _prepare_lvd_plus_record(config, scales=args.action == "record-lvd-plus-scales")
    if args.action == "record-s0":
        if _world_size() != 1:
            raise ConfigError("record_s0 is a single-process head-only job; do not use torchrun")
        if args.resume_from_checkpoint:
            raise ConfigError("record_s0 does not support training checkpoint resume")
        from sft_conflict.s0_workflow import record_s0

        _preflight(config)
        return record_s0(config, device="cuda:0" if torch.cuda.is_available() else "cpu")
    if args.action == "generate-pilot":
        from sft_conflict.pilot_workflow import generate_pilot

        return generate_pilot(
            config,
            steps=args.pilot_steps,
            health_interval=args.pilot_health_interval,
        )
    if config.method.name is MethodName.FILTERING_ANCHOR:
        if args.action != "filtering-anchor":
            raise ConfigError("method=filtering_anchor requires action=filtering-anchor")
        return _filter_anchor(config, args)
    if config.method.name is MethodName.TEST_INTERFERENCE:
        if args.action != "test-interference":
            raise ConfigError("method=test_interference requires action=test-interference")
        return _test_interference(config, args)
    if config.method.name is MethodName.TEST_CROSSING:
        if args.action != "test-crossing":
            raise ConfigError("method=test_crossing requires action=test-crossing")
        return _test_crossing(config, args)
    if config.method.name is MethodName.RECORD_ANCHOR_D0:
        if args.action not in {"train", "record-anchor-d0"}:
            raise ConfigError("method=record_anchor_d0 cannot be combined with this action")
        return _build_reference_cache(config, args)
    if args.action == "train":
        config.training.resume_from_checkpoint = args.resume_from_checkpoint
        return _train(config)
    if args.action == "audit-model":
        return _audit_model(
            config,
            lvd_audit_max_positions=args.lvd_audit_max_positions,
            lvd_audit_vocabulary_chunk_size=args.lvd_audit_vocabulary_chunk_size,
        )
    if args.action == "evaluate":
        if not args.eval_checkpoint:
            raise ConfigError("action=evaluate requires --eval-checkpoint")
        config.evaluation.eval_checkpoint = args.eval_checkpoint
        return _evaluate(config)
    if args.action == "merge-lora":
        if not args.adapter or not args.output:
            raise ConfigError("action=merge-lora requires --adapter and --output")
        return _merge_lora(config, args.adapter, args.output)
    if args.action in {"build-reference-cache", "record-anchor-d0"}:
        return _build_reference_cache(config, args)
    if args.action == "native-evaluate":
        return _native_evaluate_with_llm_evaluation(config, args)
    if args.action == "audit-data":
        return _audit_data(config, args)
    raise AssertionError(args.action)


def _add_config_overrides(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, help="Path to one complete experiment YAML")
    parser.add_argument("--run-name", help="Override the artifact run ID")
    parser.add_argument("--mixed-precision", choices=("bf16", "fp16", "fp32"))
    parser.add_argument("--training-mode", choices=tuple(mode.value for mode in TrainingMode))
    parser.add_argument(
        "--freeze-unembedding",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Explicitly freeze or train the output Unembedding",
    )


def _optional_positive_int(value: str) -> int | None:
    if value.casefold() == "none":
        return None
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("sample limits must be positive integers or None")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative integer")
    return parsed


def _add_lvd_audit_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--lvd-audit-max-positions", type=_positive_int, default=8)
    parser.add_argument(
        "--lvd-audit-vocabulary-chunk-size",
        type=_positive_int,
        default=256,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sft-conflict")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    run_parser = commands.add_parser(
        "run",
        help="Run a preset Tiny or cached real experiment with defaults",
    )
    run_parser.add_argument(
        "--action",
        choices=(
            "train",
            "generate-pilot",
            "audit-model",
            "evaluate",
            "merge-lora",
            "build-reference-cache",
            "record-anchor-d0",
            "record-s0",
            "record-lvd-plus-scales",
            "calibrate-lvd-plus-lambda",
            "prepare-anchor-geometry-reference",
            "native-evaluate",
            "audit-data",
            "benchmark",
            "test-interference",
            "test-crossing",
            "filtering-anchor",
            "create-anchor-dataset",
        ),
        default="train",
    )
    run_parser.add_argument("--model", choices=RUN_MODELS, default="tiny_qwen2")
    run_parser.add_argument(
        "--target-dataset",
        choices=RUN_TARGET_DATASETS,
        default="tiny_gsm8k",
    )
    run_parser.add_argument(
        "--anchor-dataset",
        choices=RUN_ANCHOR_DATASETS,
        default="none",
    )
    run_parser.add_argument(
        "--method",
        choices=tuple(method.value for method in MethodName),
        default=MethodName.TARGET_ONLY.value,
    )
    run_parser.add_argument(
        "--supervision",
        choices=RUN_SUPERVISION_MODES,
        default="response_only",
    )
    run_parser.add_argument(
        "--training-mode",
        choices=tuple(mode.value for mode in TrainingMode),
        default=TrainingMode.BODY_FT_FROZEN_HEAD.value,
    )
    run_parser.add_argument(
        "--distributed-strategy",
        choices=("single", "ddp", "fsdp"),
        default="single",
    )
    run_parser.add_argument(
        "--mixed-precision",
        choices=("bf16", "fp16", "fp32"),
        default="fp32",
    )
    run_parser.add_argument(
        "--optimizer",
        choices=("adamw_torch", "adamw_bnb_8bit", "adafactor"),
        default="adamw_torch",
    )
    run_parser.add_argument("--max-steps", type=int, default=2)
    run_parser.add_argument("--pilot-steps", type=_positive_int, default=10)
    run_parser.add_argument("--pilot-health-interval", type=_positive_int, default=5)
    run_parser.add_argument("--num-train-epochs", type=float, default=1.0)
    run_parser.add_argument("--learning-rate", type=float, default=1e-3)
    run_parser.add_argument("--weight-decay", type=float, default=0.0)
    run_parser.add_argument("--warmup-ratio", type=float, default=0.0)
    run_parser.add_argument("--max-length", type=int, default=64)
    _add_lvd_audit_arguments(run_parser)
    run_parser.add_argument("--interference-token-chunk-size", type=_positive_int, default=256)
    run_parser.add_argument(
        "--interference-position-chunk-size",
        type=_positive_int,
        default=1024,
    )
    run_parser.add_argument("--interference-output-dir")
    run_parser.add_argument("--target-max-samples", type=_optional_positive_int, default=256)
    run_parser.add_argument("--target-validation-max-samples", type=_optional_positive_int)
    run_parser.add_argument("--anchor-max-samples", type=_optional_positive_int, default=256)
    run_parser.add_argument("--anchor-metadata-path")
    run_parser.add_argument("--filter-threshold", type=float, default=0.8)
    run_parser.add_argument("--filtering-batch-size", type=_positive_int, default=4)
    run_parser.add_argument(
        "--filtering-policy",
        choices=(LABEL_COVERAGE_POLICY, LEGACY_FILTER_POLICY),
        default=LABEL_COVERAGE_POLICY,
        help="Label-only source-order cover (CPU); legacy policy retains correctness filtering.",
    )
    run_parser.add_argument("--anchor-creation-seed", type=int, default=42)
    run_parser.add_argument("--anchor-creation-workers", type=_non_negative_int, default=8)
    run_parser.add_argument("--anchor-creation-scoring-batch-size", type=_positive_int, default=4)
    run_parser.add_argument(
        "--anchor-creation-compute-accuracy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Report base-model teacher-forced accuracy on the created Anchor dataset.",
    )
    run_parser.add_argument(
        "--anchor-creation-submixture",
        action="append",
        choices=SUBMIXTURE_KEYS,
        help="Restrict create-anchor-dataset to these SMART submixtures (default: all five).",
    )
    run_parser.add_argument(
        "--anchor-creation-max-samples-per-task",
        type=_optional_positive_int,
        default=None,
        help="Smoke-test cap on samples per task; None traverses every sample.",
    )
    run_parser.add_argument(
        "--anchor-creation-guarantee-task-coverage",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep each task's first drawn supervised sample unconditionally on its first turn.",
    )
    run_parser.add_argument(
        "--evaluation-max-samples",
        type=_optional_positive_int,
        default=256,
    )
    run_parser.add_argument("--evaluation-batch-size", type=int, default=1)
    run_parser.add_argument("--target-batch-size", type=int, default=2)
    run_parser.add_argument("--anchor-batch-size", type=int, default=2)
    run_parser.add_argument(
        "--anchor-sampling-strategy",
        choices=("task_balanced", "mixture_proportional", "random"),
        default="task_balanced",
    )
    run_parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    run_parser.add_argument(
        "--recovery-save-steps",
        type=_optional_positive_int,
        default=None,
        help="Extra optimizer-step checkpoints; keeps epoch evaluation and best-model selection.",
    )
    run_parser.add_argument(
        "--fsdp-sync-each-batch",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    run_parser.add_argument(
        "--fsdp-cpu-offload",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Offload FSDP parameters/gradients to CPU; default: auto for models >=6B.",
    )
    run_parser.add_argument(
        "--fsdp-keep-low-precision-params",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    run_parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    run_parser.add_argument(
        "--run-after-training",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    run_parser.add_argument(
        "--training-validation",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    run_parser.add_argument(
        "--best-model-metric",
        choices=("eval_loss", "native_accuracy"),
        default="eval_loss",
        help=(
            "validation metric that picks the checkpoint loaded at the end of training: "
            "the evaluation loss or the task-native classification accuracy on the "
            "same training validation split"
        ),
    )
    run_parser.add_argument("--lora-rank", type=int, default=8)
    run_parser.add_argument("--lora-alpha", type=int, default=32)
    run_parser.add_argument("--lora-dropout", type=float, default=0.05)
    run_parser.add_argument("--anchor-weight", type=float, default=0.2)
    run_parser.add_argument("--kl-temperature", type=float, default=1.0)
    run_parser.add_argument("--seed", type=int, default=42)
    run_parser.add_argument(
        "--artifact-root",
        default=DEFAULT_ARTIFACT_ROOT,
    )
    run_parser.add_argument("--run-name")
    run_parser.add_argument("--lvd-config", default=str(DEFAULT_LVD_CONFIG_PATH))
    run_parser.add_argument("--lvd-plus-config", default=str(DEFAULT_LVD_PLUS_CONFIG_PATH))
    run_parser.add_argument(
        "--anchor-geometry-config", default=str(DEFAULT_ANCHOR_GEOMETRY_CONFIG_PATH)
    )
    run_parser.add_argument("--anchor-geometry-reference-path")
    run_parser.add_argument("--lvd-plus-direction", type=_positive_int)
    run_parser.add_argument("--lvd-plus-scale-cache-path")
    run_parser.add_argument("--lvd-plus-lambda-cache-path")
    run_parser.add_argument("--s0-config", default=str(DEFAULT_S0_CONFIG_PATH))
    run_parser.add_argument("--s0-cache-path")
    run_parser.add_argument("--lvd-direction", type=int, help="Override LVD competitor count")
    run_parser.add_argument(
        "--lvd-use-capacity-weights",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override YAML capacity weights; disable for normalized uniform alpha=1/K",
    )
    run_parser.add_argument(
        "--lvd-position-strategy",
        choices=tuple(strategy.value for strategy in AnchorPositionStrategy),
    )
    run_parser.add_argument("--lvd-positions-per-sample", type=int)
    run_parser.add_argument("--anchor-d0-cache-path")
    run_parser.add_argument("--lvd-calibration-checkpoint")
    run_parser.add_argument(
        "--auto-prepare-cache",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    run_parser.add_argument("--resume-from-checkpoint")
    run_parser.add_argument("--eval-checkpoint")
    run_parser.add_argument("--adapter")
    run_parser.add_argument("--output")
    run_parser.add_argument("--reference-cache-dir")
    run_parser.add_argument(
        "--anchor-position-strategy",
        choices=tuple(strategy.value for strategy in AnchorPositionStrategy),
        default=AnchorPositionStrategy.SAMPLE_K_POSITIONS.value,
    )
    run_parser.add_argument("--anchor-positions-per-sample", type=int, default=4)
    run_parser.add_argument("--reference-samples-per-shard", type=int, default=4)
    run_parser.add_argument(
        "--reference-storage-dtype",
        choices=("float32", "bfloat16", "float16"),
        default="float32",
    )
    run_parser.add_argument(
        "--reference-filter-mode",
        choices=tuple(mode.value for mode in ReferenceFilterMode),
        default=ReferenceFilterMode.BASE_CORRECT.value,
    )
    run_parser.add_argument("--minimum-reference-confidence", type=float)
    run_parser.add_argument("--minimum-reference-margin", type=float)
    run_parser.add_argument("--hidden-max-absolute-error", type=float, default=0.02)
    run_parser.add_argument("--hidden-mean-absolute-error", type=float, default=0.002)
    run_parser.add_argument("--minimum-argmax-agreement", type=float, default=0.999)
    run_parser.add_argument("--margin-max-absolute-error", type=float, default=0.05)
    run_parser.add_argument("--margin-max-relative-error", type=float, default=0.10)
    run_parser.add_argument("--maximum-margin-sign-flip-rate", type=float, default=0.001)
    run_parser.add_argument("--native-max-new-tokens", type=int, default=256)
    run_parser.add_argument(
        "--allow-code-execution",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    run_parser.add_argument("--humaneval-samples-per-problem", type=int, default=20)
    run_parser.add_argument("--humaneval-temperature", type=float, default=0.2)
    run_parser.add_argument("--humaneval-top-p", type=float, default=0.95)
    run_parser.add_argument("--humaneval-timeout-seconds", type=float, default=3.0)
    run_parser.add_argument("--humaneval-execution-workers", type=int, default=16)
    run_parser.add_argument("--native-stop-token", action="append", default=[])
    run_parser.add_argument("--native-evaluation-dir")
    run_parser.add_argument(
        "--llm-evaluation",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    run_parser.add_argument("--llm-evaluation-task", action="append")
    run_parser.add_argument("--llm-evaluation-batch-size", default="auto")
    run_parser.add_argument("--llm-evaluation-num-fewshot", type=int)
    run_parser.add_argument("--llm-evaluation-limit", type=float)
    run_parser.add_argument(
        "--llm-evaluation-check-integrity",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    run_parser.add_argument(
        "--alpaca-eval-run-judge",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    run_parser.add_argument(
        "--alpaca-eval-annotators-config",
        choices=ALPACA_EVAL_ANNOTATORS,
        default="chatgpt_fn",
    )
    run_parser.add_argument("--near-duplicate-threshold", type=float, default=0.85)
    run_parser.add_argument("--data-audit-output")
    run_parser.add_argument("--benchmark-seed", type=int, action="append")
    run_parser.add_argument(
        "--benchmark-formal",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    run_parser.add_argument(
        "--benchmark-execute",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    run_parser.add_argument("--benchmark-general-task", action="append")
    run_parser.add_argument("--benchmark-general-limit", type=float)
    run_parser.add_argument("--benchmark-num-fewshot", type=int, default=0)
    run_parser.add_argument(
        "--benchmark-check-integrity",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    run_parser.add_argument("--benchmark-output-dir")
    run_parser.add_argument(
        "--lm-eval-python",
        default=DEFAULT_LM_EVAL_PYTHON,
    )
    run_parser.add_argument(
        "--lm-eval-harness-path",
        default=DEFAULT_LM_EVAL_HARNESS_PATH,
    )
    run_parser.add_argument("--lm-eval-batch-size", default="auto")

    train_parser = commands.add_parser("train", help="Run standard Trainer training")
    _add_config_overrides(train_parser)
    train_parser.add_argument("--resume-from-checkpoint")

    for action in ("record-lvd-plus-scales", "calibrate-lvd-plus-lambda"):
        preparation_parser = commands.add_parser(action, help="Prepare a hybrid LVD+ Base record")
        _add_config_overrides(preparation_parser)

    evaluate_parser = commands.add_parser("evaluate", help="Evaluate one Trainer checkpoint")
    _add_config_overrides(evaluate_parser)
    evaluate_parser.add_argument("--eval-checkpoint")

    audit_parser = commands.add_parser(
        "audit-model",
        help="Audit Unembedding logits and planar LVD convex-cell ownership",
    )
    _add_config_overrides(audit_parser)
    _add_lvd_audit_arguments(audit_parser)

    merge_parser = commands.add_parser("merge-lora", help="Export one Adapter as a merged model")
    _add_config_overrides(merge_parser)
    merge_parser.add_argument("--adapter", required=True)
    merge_parser.add_argument("--output", required=True)

    migrate_parser = commands.add_parser("migrate-config", help="Migrate one legacy config")
    migrate_parser.add_argument("--input", required=True)
    migrate_parser.add_argument("--output", required=True)
    migrate_parser.add_argument("--report", required=True)
    migrate_parser.add_argument("--model-revision", required=True)
    migrate_parser.add_argument("--dataset-revision", required=True)

    lm_eval_parser = commands.add_parser("lm-eval", help="Run isolated lm-eval 0.4.9")
    lm_eval_parser.add_argument(
        "--python",
        default=DEFAULT_LM_EVAL_PYTHON,
    )
    lm_eval_parser.add_argument("--model", default="hf")
    lm_eval_parser.add_argument(
        "--harness-path",
        default=DEFAULT_LM_EVAL_HARNESS_PATH,
    )
    lm_eval_parser.add_argument("--model-args", required=True)
    lm_eval_parser.add_argument("--tasks", required=True)
    lm_eval_parser.add_argument("--output", required=True)
    lm_eval_parser.add_argument("--batch-size", default="auto")
    lm_eval_parser.add_argument("--num-fewshot", type=int)
    lm_eval_parser.add_argument("--limit", type=float)
    lm_eval_parser.add_argument("--seed", type=int, default=42)
    lm_eval_parser.add_argument(
        "--log-samples", action=argparse.BooleanOptionalAction, default=True
    )
    lm_eval_parser.add_argument("--check-integrity", action="store_true")
    lm_eval_parser.add_argument("--write-out", action="store_true")
    lm_eval_parser.add_argument(
        "--use-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    lm_eval_parser.add_argument("--request-cache-root")
    lm_eval_parser.add_argument(
        "--apply-chat-template",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    lm_eval_parser.add_argument(
        "--fewshot-as-multiturn",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    lm_eval_parser.add_argument("--generation-kwargs")
    lm_eval_parser.add_argument(
        "--auto-prepare-cache",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    raw_arguments = list(sys.argv[1:] if argv is None else argv)
    if not raw_arguments:
        raw_arguments = ["run"]
    args = _parser().parse_args(raw_arguments)
    if args.command == "migrate-config":
        migrate_legacy_config_file(
            args.input,
            args.output,
            args.report,
            model_revision=args.model_revision,
            dataset_revision=args.dataset_revision,
        )
        return 0
    if args.command == "lm-eval":
        request = LmEvalRequest(
            python_executable=args.python,
            model=args.model,
            model_args=args.model_args,
            tasks=tuple(task.strip() for task in args.tasks.split(",") if task.strip()),
            output_path=args.output,
            harness_path=args.harness_path,
            batch_size=args.batch_size,
            num_fewshot=args.num_fewshot,
            limit=args.limit,
            seed=args.seed,
            log_samples=args.log_samples,
            check_integrity=args.check_integrity,
            write_out=args.write_out,
            use_cache=args.use_cache,
            request_cache_root=args.request_cache_root,
            apply_chat_template=args.apply_chat_template,
            fewshot_as_multiturn=args.fewshot_as_multiturn,
            generation_kwargs=args.generation_kwargs,
            auto_prepare=args.auto_prepare_cache,
        )
        result = run_lm_eval(request)
    elif args.command == "run":
        result = _preset_run(args)
    else:
        config = load_config(args.config)
        _apply_overrides(config, args)
        if args.command == "train":
            result = _train(config)
        elif args.command in {"record-lvd-plus-scales", "calibrate-lvd-plus-lambda"}:
            result = _prepare_lvd_plus_record(
                config, scales=args.command == "record-lvd-plus-scales"
            )
        elif args.command == "evaluate":
            result = _evaluate(config)
        elif args.command == "audit-model":
            result = _audit_model(
                config,
                lvd_audit_max_positions=args.lvd_audit_max_positions,
                lvd_audit_vocabulary_chunk_size=args.lvd_audit_vocabulary_chunk_size,
            )
        elif args.command == "merge-lora":
            result = _merge_lora(config, args.adapter, args.output)
        else:
            raise AssertionError(args.command)
    if _is_world_process_zero():
        json.dump(result, sys.stdout, indent=2, sort_keys=True, default=str)
        sys.stdout.write("\n")
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
