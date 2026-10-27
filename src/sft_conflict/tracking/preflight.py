"""Human-readable experiment preview printed before every fine-tuning run."""

from __future__ import annotations

import os
from typing import Any

from sft_conflict.config import ExperimentConfig, LVDPlusAnchorMode, MethodName, TrainingMode
from sft_conflict.models import ModelAudit

from .tensor_preview import build_random_tensor_preview


def _optimization_summary(config: ExperimentConfig, audit: ModelAudit) -> dict[str, Any]:
    mode = config.model.training_mode
    frozen_parameters = audit.total_parameters - audit.trainable_parameters
    if mode is TrainingMode.FULL_FT:
        trainable_scope = "all model parameters, including the Unembedding"
        frozen_scope = "none"
    elif mode is TrainingMode.BODY_FT_FROZEN_HEAD:
        trainable_scope = "Transformer body and input embeddings"
        frozen_scope = f"Unembedding module {audit.unembedding_module}"
    else:
        lora = audit.lora or {}
        requested_targets = lora.get("requested_targets", ())
        matched_modules = lora.get("matched_modules", ())
        trainable_scope = (
            f"LoRA adapters for {', '.join(requested_targets)} "
            f"across {len(matched_modules)} matched modules"
        )
        frozen_scope = "base model parameters, input embeddings, and Unembedding"
    return {
        "mode": mode.value,
        "trainable_scope": trainable_scope,
        "frozen_scope": frozen_scope,
        "total_parameters": audit.total_parameters,
        "trainable_parameters": audit.trainable_parameters,
        "frozen_parameters": frozen_parameters,
        "unembedding_module": audit.unembedding_module,
        "unembedding_frozen": audit.unembedding_frozen,
        "tied_before": audit.tied_before,
        "tied_after": audit.tied_after,
        "mixed_precision": config.training.mixed_precision,
        "optimizer": config.training.optimizer,
        "recovery_save_steps": config.training.recovery_save_steps,
        "fsdp_keep_low_precision_params": config.training.fsdp_keep_low_precision_params,
        "fsdp_sync_each_batch": config.training.fsdp_sync_each_batch,
        "fsdp_cpu_offload_requested": config.training.fsdp_cpu_offload,
        "fsdp_cpu_offload": config.training.distributed_strategy == "fsdp"
        and (
            config.training.fsdp_cpu_offload
            if config.training.fsdp_cpu_offload is not None
            else audit.total_parameters >= 6_000_000_000
        ),
        "gradient_checkpointing": config.model.gradient_checkpointing,
        "distributed_strategy": config.training.distributed_strategy,
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        "rank": int(os.environ.get("RANK", "0")),
        "local_rank": int(os.environ.get("LOCAL_RANK", "0")),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
    }


def _loss_plan(config: ExperimentConfig) -> list[dict[str, Any]]:
    target_is_regression = config.data.target.adapter == "stsb"
    losses: list[dict[str, Any]] = [
        {
            "dataset_role": "target",
            "loss": "STSBTokenExpectationMSE" if target_is_regression else "TargetCrossEntropy",
            "weight": 1.0,
            "supervision": (
                "expected integer/fraction numeric token score"
                if target_is_regression
                else "response_only shifted Causal LM labels"
            ),
            "constraint": "primary target-task optimization",
        }
    ]
    method = config.method.name
    if method in {
        MethodName.TARGET_PLUS_ANCHOR_CE,
        MethodName.TARGET_PLUS_ANCHOR_CE_LVD,
        MethodName.TARGET_PLUS_ANCHOR_CE_GEOMETRY,
    }:
        losses.append(
            {
                "dataset_role": "anchor",
                "loss": "AnchorCrossEntropy",
                "weight": config.method.anchor_weight,
                "supervision": "response_only shifted Causal LM labels",
                "constraint": "supervised general-capability rehearsal",
            }
        )
    elif method is MethodName.TARGET_PLUS_ANCHOR_KL:
        losses.append(
            {
                "dataset_role": "anchor",
                "loss": "AnchorKLDistillation",
                "weight": config.method.anchor_weight,
                "temperature": config.method.kl_temperature,
                "supervision": "response-token prediction positions",
                "constraint": "KL to a frozen online reference model",
            }
        )
    elif method is MethodName.TARGET_PLUS_NULL_LVD:
        losses.append(
            {
                "dataset_role": "anchor",
                "loss": "NullLVDLoss",
                "weight": 0.0,
                "supervision": "selected Unembedding-input hidden states",
                "constraint": "M1 interface check only; no optimization effect",
            }
        )
    if method.uses_dynamic_geometry:
        geometry = config.anchor_geometry
        losses.append(
            {
                "dataset_role": "anchor",
                "loss": {
                    "base_band_v2": "BaseConditionedBandHuber",
                    "relative_band_v3": "RelativeErosionHuber",
                }.get(geometry.objective, "CenteredDynamicMarginHuber"),
                "objective": geometry.objective,
                "weight": geometry.resolved_lambda
                if geometry.resolved_lambda is not None
                else f"post_warmup_anchor_gradient_ratio:{geometry.geometry_gradient_ratio}",
                "supervision": "Base-correct supervised shifted labels; CE still uses all labels",
                "constraint": (
                    "live trainable head + hidden, differentiable normalized margin; "
                    f"current/Base/coverage={geometry.current_directions}/"
                    f"{geometry.base_directions}/{geometry.coverage_directions}, "
                    f"masses={geometry.current_mass}/{geometry.base_mass}/"
                    f"{geometry.coverage_mass}, "
                    f"delta={geometry.geometry_delta}, CE warmup={geometry.geometry_start_step}, "
                    f"ramp={geometry.geometry_ramp_steps}, "
                    f"reference={geometry.reference_cache_path}"
                ),
                "reference_sha256": geometry.reference_sha256,
                "calibrated_at_step": geometry.calibrated_at_step,
                "calibration_sha256": geometry.calibration_sha256,
                **(
                    {
                        "band_rule": {
                            "scale": "Base top-k normalized-margin median * scale_ratio",
                            "width": (
                                "scale * exp((base_logit - non_gold_peak)/temperature) "
                                "* frequency_factor"
                            ),
                            "scale_ratio": geometry.band_scale_ratio,
                            "temperature": geometry.band_temperature,
                            "frequency_strength": geometry.band_frequency_strength,
                            "scale_top_k": geometry.band_scale_top_k,
                            "center": "nonnegative convex minimizer",
                            "boundary": "current margin >= 0 inside the band; soft outside penalty",
                            "solver_iterations": geometry.band_solver_iterations,
                        }
                    }
                    if geometry.objective == "base_band_v2"
                    else {
                        "relative_rule": {
                            "erosion": "(base_margin - current_margin) / max(base_margin, floor)",
                            "tolerance": geometry.relative_tolerance,
                            "huber_delta": geometry.relative_delta,
                            "floor": geometry.margin_floor,
                            "one_sided": "margin increases are never penalised",
                            "calibration": geometry.calibration_mode,
                            "head_gradient_scale": geometry.head_gradient_scale,
                            "hidden_gradient_scale": geometry.hidden_gradient_scale,
                            "pull_cap_ratio": geometry.pull_cap_ratio,
                        }
                    }
                    if geometry.objective == "relative_band_v3"
                    else {}
                ),
            }
        )
    if method.uses_lvd:
        if config.lvd.resolved_lambda is not None:
            lambda_value: float | str = config.lvd.resolved_lambda
        elif config.lvd.lambda_mode.value == "fixed":
            lambda_value = config.lvd.fixed_lambda
        else:
            lambda_value = f"gradient_ratio:{config.lvd.target_gradient_ratio}"
        losses.append(
            {
                "dataset_role": "anchor",
                "loss": "WeightedPressureLVDSecondMoment",
                "weight": lambda_value,
                "supervision": "cached base d0 at selected Unembedding-input positions",
                "constraint": (
                    f"{config.lvd.neighbor_strategy.value}, top_k={config.lvd.top_k}, "
                    f"positions={config.lvd.position_strategy}, "
                    f"position_chunk_size={config.lvd.position_chunk_size}, "
                    f"moment={config.lvd.moment_mode.value}, "
                    f"d0_cache={config.lvd.d0_cache_path}"
                ),
            }
        )
    if method.uses_lvd_plus:
        plus = config.lvd_plus
        anchor_description = (
            f"recorded s0_j, cache={plus.s0_cache_path}"
            if plus.anchor_mode is LVDPlusAnchorMode.RECORDED_S0
            else "c_j=W_j/2"
        )
        weight = (
            plus.resolved_lambda
            if plus.resolved_lambda is not None
            else plus.fixed_lambda
            if plus.lambda_mode.value == "fixed"
            else f"fresh_base_rms_gradient_ratio:{plus.target_gradient_ratio}"
        )
        losses.append(
            {
                "dataset_role": "anchor",
                "loss": "LabelSiteLVDPlusSecondMoment",
                "weight": weight,
                "post_crossing_slope_gain": plus.post_crossing_slope_gain,
                "supervision": (
                    "all shifted gold response labels, including EOS and wrong predictions"
                ),
                "constraint": (
                    f"{anchor_description}, current_topk={plus.top_k}, uniform weights, "
                    f"position_chunk_size={plus.position_chunk_size}, "
                    f"post_crossing_slope_gain={plus.post_crossing_slope_gain}, "
                    f"moment={plus.moment_mode.value}; no d0, no reference model, no pilot"
                ),
            }
        )
        if plus.uses_boundary_risk:
            hybrid = losses[-1]
            hybrid.update(
                {
                    "loss": "LabelSiteLVDPlusPressurePlusRisk",
                    "objective_mode": plus.objective_mode.value,
                    "hybrid_formula": "L = L_target + lambda * (O / S_O + eta * R / S_R)",
                    "outer_formula": "O = ||P^T P / N||_F^2 (original gain-1 pressure)",
                    "risk_formula": (
                        "R = mean_i[T * softplus(d_i / T)], "
                        "d_i = max_{k != j_i}(z_ik - z_ij_i) (full vocabulary, all signs)"
                    ),
                    "risk_temperature": plus.risk_temperature,
                    "risk_weight": plus.risk_weight,
                    "resolved_outer_scale": plus.resolved_outer_scale,
                    "resolved_risk_scale": plus.resolved_risk_scale,
                    "s0_cache_path": plus.s0_cache_path,
                    "s0_cache_sha256": plus.s0_cache_sha256,
                    "scale_cache_path": plus.scale_cache_path,
                    "scale_cache_sha256": plus.scale_cache_sha256,
                    "lambda_cache_path": plus.lambda_cache_path,
                    "lambda_cache_sha256": plus.lambda_cache_sha256,
                }
            )
            if plus.resolved_lambda is None and plus.lambda_mode.value != "fixed":
                hybrid["weight"] = (
                    f"prepared_hybrid_base_rms_gradient_ratio:{plus.target_gradient_ratio}"
                )
    return losses


def _dataset_summary(
    role: str,
    config,
    dataset,
    tokenizer,
    *,
    sample_count: int,
    optimization_role: str,
    pad_token_id: int,
    pad_to_multiple_of: int | None,
) -> dict[str, Any]:
    tensor_preview = (
        build_random_tensor_preview(
            dataset,
            tokenizer,
            stage="training",
            pad_token_id=pad_token_id,
            pad_to_multiple_of=pad_to_multiple_of,
            count=sample_count,
        )
        if sample_count
        else None
    )
    return {
        "role": role,
        "optimization_role": optimization_role,
        "source": config.source,
        "path": config.path,
        "name": config.name,
        "revision": config.revision,
        "split": config.split,
        "adapter": config.adapter,
        "loaded_samples": len(dataset),
        "fingerprint": dataset.fingerprint,
        "preprocessing_stats": dataset.preprocessing_stats,
        "tensor_preview": tensor_preview,
        "samples": tensor_preview["samples"] if tensor_preview is not None else [],
    }


def build_preflight_preview(
    config: ExperimentConfig,
    audit: ModelAudit,
    tokenizer,
    target_dataset,
    anchor_dataset,
    evaluation_dataset,
    *,
    validation_dataset=None,
    sample_count: int = 3,
) -> dict[str, Any]:
    target_loss = _loss_plan(config)[0]
    datasets = [
        _dataset_summary(
            "target",
            config.data.target,
            target_dataset,
            tokenizer,
            sample_count=sample_count,
            optimization_role=f"optimized with {target_loss['loss']}",
            pad_token_id=tokenizer.pad_token_id,
            pad_to_multiple_of=config.data.pad_to_multiple_of,
        )
    ]
    if anchor_dataset is not None and config.data.anchor is not None:
        anchor_loss = next(item for item in _loss_plan(config) if item["dataset_role"] == "anchor")
        datasets.append(
            _dataset_summary(
                "anchor",
                config.data.anchor,
                anchor_dataset,
                tokenizer,
                sample_count=sample_count,
                optimization_role=(
                    f"optimized with {anchor_loss['loss']} at weight {anchor_loss['weight']}"
                ),
                pad_token_id=tokenizer.pad_token_id,
                pad_to_multiple_of=config.data.pad_to_multiple_of,
            )
        )
    if validation_dataset is not None and config.data.validation is not None:
        datasets.append(
            _dataset_summary(
                "validation",
                config.data.validation,
                validation_dataset,
                tokenizer,
                sample_count=0,
                optimization_role="internal model selection only; never used for updates",
                pad_token_id=tokenizer.pad_token_id,
                pad_to_multiple_of=config.data.pad_to_multiple_of,
            )
        )
    if evaluation_dataset is not None:
        evaluation_config = config.evaluation.target or config.data.target
        datasets.append(
            _dataset_summary(
                "evaluation",
                evaluation_config,
                evaluation_dataset,
                tokenizer,
                sample_count=0,
                optimization_role="evaluation only; never used for optimizer updates",
                pad_token_id=tokenizer.pad_token_id,
                pad_to_multiple_of=config.data.pad_to_multiple_of,
            )
        )
    return {
        "model": {
            "model_id": config.model.model_id,
            "revision": config.model.revision,
            "model_type": audit.model_type,
        },
        "optimization": _optimization_summary(config, audit),
        "method": config.method.name.value,
        "loss_plan": _loss_plan(config),
        "datasets": datasets,
    }


def render_preflight_preview(preview: dict[str, Any]) -> str:
    model = preview["model"]
    optimization = preview["optimization"]
    lines = [
        "",
        "=" * 88,
        "SFT-CONFLICT PRE-FINETUNING PREVIEW",
        "=" * 88,
        f"Model: {model['model_id']} @ {model['revision']}",
        f"Model type: {model['model_type']}",
        f"Fine-tuning mode: {optimization['mode']}",
        f"Trainable scope: {optimization['trainable_scope']}",
        f"Frozen scope: {optimization['frozen_scope']}",
        (
            "Parameters: "
            f"trainable={optimization['trainable_parameters']:,}, "
            f"frozen={optimization['frozen_parameters']:,}, "
            f"total={optimization['total_parameters']:,}"
        ),
        (
            "Unembedding: "
            f"module={optimization['unembedding_module']}, "
            f"frozen={optimization['unembedding_frozen']}, "
            f"tied_before={optimization['tied_before']}, "
            f"tied_after={optimization['tied_after']}"
        ),
        (
            "Numerics: "
            f"precision={optimization['mixed_precision']}, "
            f"optimizer={optimization['optimizer']}, "
            f"gradient_checkpointing={optimization['gradient_checkpointing']}"
        ),
        (
            "FSDP memory: "
            f"keep_low_precision_params={optimization['fsdp_keep_low_precision_params']}, "
            f"sync_each_batch={optimization['fsdp_sync_each_batch']}, "
            f"cpu_offload={optimization['fsdp_cpu_offload']} "
            f"(requested={optimization['fsdp_cpu_offload_requested']})"
        ),
        (
            "CPU threading: "
            f"OMP_NUM_THREADS={optimization['omp_num_threads']}, "
            f"MKL_NUM_THREADS={optimization['mkl_num_threads']}"
        ),
        (
            "Distributed: "
            f"strategy={optimization['distributed_strategy']}, "
            f"world_size={optimization['world_size']}, "
            f"rank={optimization['rank']}, local_rank={optimization['local_rank']}, "
            f"CUDA_VISIBLE_DEVICES={optimization['cuda_visible_devices']}"
        ),
        f"Method: {preview['method']}",
        (
            "Extra recovery checkpoint interval: "
            f"{optimization['recovery_save_steps']} optimizer steps"
        ),
        "Loss and constraint plan:",
    ]
    for loss in preview["loss_plan"]:
        detail = (
            f"  - {loss['dataset_role']}: {loss['loss']} "
            f"(weight={loss['weight']}) | {loss['constraint']} | {loss['supervision']}"
        )
        if "temperature" in loss:
            detail += f" | temperature={loss['temperature']}"
        lines.append(detail)
        if loss.get("objective_mode") == "pressure_plus_risk":
            lines.extend(
                [
                    "  Hybrid LVD+ objective (unchanged formal method):",
                    f"    {loss['hybrid_formula']}",
                    f"    {loss['outer_formula']}",
                    f"    {loss['risk_formula']}",
                    "  Hybrid parameters and immutable cache identities:",
                    "  | Setting | Value |",
                    "  | --- | --- |",
                ]
            )
            rows = (
                ("objective_mode", loss["objective_mode"]),
                ("post_crossing_slope_gain", loss["post_crossing_slope_gain"]),
                ("T (risk_temperature)", loss["risk_temperature"]),
                ("eta (risk_weight)", loss["risk_weight"]),
                ("S_O (resolved_outer_scale)", loss["resolved_outer_scale"]),
                ("S_R (resolved_risk_scale)", loss["resolved_risk_scale"]),
                ("lambda", loss["weight"]),
                ("S0 cache path", loss["s0_cache_path"]),
                ("S0 cache SHA-256", loss["s0_cache_sha256"]),
                ("Scale cache path", loss["scale_cache_path"]),
                ("Scale cache SHA-256", loss["scale_cache_sha256"]),
                ("Lambda cache path", loss["lambda_cache_path"]),
                ("Lambda cache SHA-256", loss["lambda_cache_sha256"]),
            )
            for name, value in rows:
                rendered = (
                    "unresolved (CLI runtime verification required)"
                    if value is None
                    else str(value)
                )
                lines.append(f"  | {name} | {rendered} |")

    for dataset in preview["datasets"]:
        identifier = dataset["path"]
        if dataset["name"]:
            identifier += f"/{dataset['name']}"
        lines.extend(
            [
                "-" * 88,
                (
                    f"Dataset [{dataset['role']}]: {identifier} | split={dataset['split']} | "
                    f"revision={dataset['revision'] or 'local'}"
                ),
                (
                    f"Loaded samples={dataset['loaded_samples']:,} | "
                    f"adapter={dataset['adapter']} | "
                    f"role={dataset['optimization_role']}"
                ),
                f"Preprocessing: {dataset['preprocessing_stats']}",
            ]
        )
        for number, sample in enumerate(dataset["samples"], start=1):
            if number == 1:
                tensor_preview = dataset["tensor_preview"]
                lines.append(f"Tensor sample source: {tensor_preview['source']}")
                lines.append(
                    "Tensor shapes: "
                    f"input_ids={tensor_preview['input_tensor_shape']}, "
                    f"labels={tensor_preview['label_tensor_shape']}"
                )
            lines.extend(
                [
                    f"  Sample {number}: index={sample['dataset_index']} id={sample['sample_id']}",
                    "    FULL INPUT TEXT (teacher-forced input_ids tensor):",
                    "      " + sample["input_text"].replace("\n", "\n      "),
                    "    FIRST-LABEL PREDICTION PREFIX (label token excluded):",
                    "      " + sample["prediction_prefix_text"].replace("\n", "\n      "),
                    "    LABEL TEXT (decoded from labels tensor after removing -100):",
                    "      " + sample["label_text"].replace("\n", "\n      "),
                    (
                        "    TOKENS: "
                        f"full_input={sample['input_token_count']}, "
                        f"first_label_prefix={sample['prediction_prefix_token_count']}, "
                        f"supervised={sample['label_token_count']}"
                    ),
                ]
            )
    lines.extend(["=" * 88, ""])
    return "\n".join(lines)
