"""One in-place, pre-training Base calibration for label-site LVD+.

This module deliberately does not call the old calibration or pilot gates.
Only the shared numerical normalizer and post-average gradient norm convention
are reused. No optimizer step, reference model or d0/pilot publication is needed.
Recorded-S0 mode binds the already-validated static anchor identity into the plan.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import math
import random
from collections.abc import Iterable, Mapping
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.distributed.fsdp import FullyShardedDataParallel

from sft_conflict.config import LVDLambdaMode, LVDPlusAnchorMode
from sft_conflict.losses import CountAwareMetricAccumulator, LossContext, TargetCrossEntropy
from sft_conflict.lvd_plus_workflow import (
    _sha256_json,
    _source_identity,
    lvd_plus_calibration_batches,
)
from sft_conflict.pilot_stability import calibration_normalizer
from sft_conflict.trainers.gradient_accumulation import CPUGradientAccumulator

if TYPE_CHECKING:
    from sft_conflict.trainers.sft_trainer import SFTConflictTrainer

METHOD = "target_plus_anchor_lvd+"
CENTER_MODE = "labelsite_half_unembedding"


def _s0_provenance(config, plugin) -> dict[str, Any]:
    """Use the trainer-verified artifact identity without rehashing large tables."""
    if config.anchor_mode is not LVDPlusAnchorMode.RECORDED_S0:
        return {}
    identity = getattr(plugin, "s0_identity", None)
    if (
        not isinstance(identity, dict)
        or identity.get("anchor_mode") != "recorded_s0"
        or not isinstance(identity.get("manifest_sha256"), str)
    ):
        raise ValueError("recorded_s0 calibration requires a verified S0 cache identity")
    if config.s0_cache_sha256 is not None and config.s0_cache_sha256 != identity["manifest_sha256"]:
        raise ValueError("S0 calibration identity disagrees with lvd_plus.s0_cache_sha256")
    _sha256_json(identity)
    return {
        "s0_identity": copy.deepcopy(identity),
        "s0_cache_sha256": identity["manifest_sha256"],
    }


def _finite_number(value: Any) -> float | None:
    if not isinstance(value, int | float) or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    return number if math.isfinite(number) else None


def lvd_plus_rms_stability_decision(
    report: Mapping[str, Any], *, minimum_groups: int = 8, allow_single_group: bool = False
) -> dict[str, Any]:
    """Independent, fail-closed LVD+ RMS engineering policy, not a quality test.

    Production uses at least eight positive groups, ordered-half RMS factor <=3
    and peak squared-ratio energy share <=.75. Explicit tiny configurations may
    use minimum_groups=2. A separately labelled single-group test may waive
    spread gates, but still needs finite strictly positive Target and RMS norms.
    All zeros and spikes remain in RMS. A zero ratio is legal ONLY for an actual
    zero auxiliary norm; positive-norm arithmetic underflow fails closed.
    """
    if type(minimum_groups) is not int or minimum_groups < 2:
        raise ValueError("minimum_groups must be an integer >= 2")
    if type(allow_single_group) is not bool:
        raise ValueError("allow_single_group must be boolean")
    reasons: list[str] = []
    decision: dict[str, Any] = {
        "version": 1,
        "method": METHOD,
        "accepted": False,
        "reasons": reasons,
        "policy": "lvd_plus_rms_engineering",
        "thresholds": {
            "minimum_positive_groups": minimum_groups,
            "max_half_rms_factor": 3.0,
            "max_group_energy_share": 0.75,
        },
        "group_count": None,
        "positive_group_count": None,
        "zero_group_count": None,
        "normalization_value": None,
        "half_split": None,
        "rms_half_factor": None,
        "maximum_group_energy_share": None,
        "notes": (
            "Separate LVD+ policy, not legacy response-mode/pilot gates. All measured groups "
            "enter RMS, without epsilon or outlier dropping. Passing is not a statistical "
            "confidence, Base-health, generalization or future-gradient guarantee."
        ),
    }
    if not isinstance(report, Mapping):
        reasons.append("calibration report must be a mapping")
        return decision
    if (
        report.get("method", METHOD) != METHOD
        or report.get("calibration_statistic", "rms") != "rms"
    ):
        reasons.append("LVD+ stability requires its own method identity and RMS statistic")
    try:
        json.dumps(dict(report), allow_nan=False)
    except (TypeError, ValueError, OverflowError, RecursionError):
        reasons.append("entire report must be JSON-serializable without NaN/Infinity")
    arrays: dict[str, list[float]] = {}
    for key in ("auxiliary_to_target_ratios", "target_gradient_norms", "auxiliary_gradient_norms"):
        raw = report.get(key)
        if not isinstance(raw, list):
            reasons.append(f"{key} must be a list")
            continue
        values = [_finite_number(value) for value in raw]
        positive = key == "target_gradient_norms"
        if any(value is None or (value <= 0 if positive else value < 0) for value in values):
            reasons.append(
                f"{key} contains invalid finite {'positive' if positive else 'non-negative'} values"
            )
        else:
            arrays[key] = [float(value) for value in values if value is not None]
    if len(arrays) != 3:
        return decision
    ratios = arrays["auxiliary_to_target_ratios"]
    count = len(ratios)
    decision["group_count"] = count
    if not count or any(len(values) != count for values in arrays.values()):
        reasons.append("gradient norm and ratio arrays must have equal nonzero group counts")
        return decision
    for key in ("calibration_group_count", "calibration_batches"):
        if key in report and (type(report[key]) is not int or report[key] != count):
            reasons.append(f"{key} must match the raw ratio group count")
    for ratio, target, auxiliary in zip(
        ratios, arrays["target_gradient_norms"], arrays["auxiliary_gradient_norms"], strict=True
    ):
        measured = auxiliary / target
        if (
            not math.isfinite(measured)
            or (auxiliary > 0 and measured == 0)
            or (ratio == 0 and auxiliary != 0)
            or not math.isclose(ratio, measured, rel_tol=1e-9, abs_tol=0.0)
        ):
            reasons.append("raw ratios must match actual gradient norms without zero underflow")
            break
    if reasons:
        return decision
    positive_count = sum(value > 0 for value in ratios)
    normalizer = calibration_normalizer(ratios, "rms")
    decision.update(
        positive_group_count=positive_count,
        zero_group_count=count - positive_count,
        normalization_value=normalizer,
    )
    if normalizer <= 0:
        reasons.append("RMS auxiliary gradient ratio is zero; calibration cannot normalize")
    if count == 1 and allow_single_group:
        decision.update(
            policy="single_group_finite_positive_only_not_production",
            thresholds={"minimum_positive_groups": 1, "spread_gates_applied": False},
        )
    else:
        if positive_count < minimum_groups:
            reasons.append(
                f"insufficient positive calibration groups: {positive_count} < {minimum_groups}"
            )
        if count >= 2:
            split = count // 2
            first = calibration_normalizer(ratios[:split], "rms")
            second = calibration_normalizer(ratios[split:], "rms")
            factor = max(first, second) / min(first, second) if min(first, second) > 0 else None
            peak = max(ratios)
            energy = math.fsum((value / peak) ** 2 for value in ratios) if peak else 0.0
            share = 1.0 / energy if energy > 0 else None
            decision.update(
                half_split={
                    "first_group_count": split,
                    "second_group_count": count - split,
                    "first_rms": first,
                    "second_rms": second,
                },
                rms_half_factor=factor if factor is not None and math.isfinite(factor) else None,
                maximum_group_energy_share=share,
            )
            if factor is None or not math.isfinite(factor) or factor > 3.0:
                reasons.append("ordered half RMS values must be positive with factor <= 3")
            if share is None or share > 0.75:
                reasons.append("maximum group energy share must be <= 0.75")
    decision["accepted"] = not reasons
    return decision


def _tensor_digest(digest: Any, tensor: Tensor) -> None:
    value = tensor.detach().cpu().contiguous()
    digest.update(_sha256_json([str(value.dtype), list(value.shape)]).encode())
    # Byte view supports BF16 without a full float32 conversion or bytes copy.
    digest.update(memoryview(value.reshape(-1).view(torch.uint8).numpy()))


def _head_fingerprint(plugin: Any) -> str:
    weight = plugin.unembedding_weight
    if weight is None or weight.device.type != "cpu" or weight.requires_grad:
        raise RuntimeError("LVD+ calibration requires the frozen CPU Unembedding snapshot")
    digest = hashlib.sha256(b"lvd-plus-frozen-head-v1")
    _tensor_digest(digest, weight)
    bias = plugin.unembedding_bias
    if bias is not None:
        if bias.device.type != "cpu" or bias.requires_grad:
            raise RuntimeError("LVD+ calibration requires the frozen CPU Unembedding bias")
        _tensor_digest(digest, bias)
    else:
        digest.update(b"no-bias")
    return digest.hexdigest()


def _batch_fingerprint(batch: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for stream in ("target", "anchor"):
        for key, value in sorted((batch.get(stream) or {}).items()):
            if isinstance(value, Tensor):
                digest.update(f"{stream}/{key}".encode())
                _tensor_digest(digest, value)
    return digest.hexdigest()


def _build_calibration_plan(trainer, plugin, batches, metadata, *, groups, steps, source):
    """Validate local provenance collectively, before any model preparation/backward."""
    from sft_conflict.trainers.sft_trainer import _gather_calibration_objects

    local, error = None, None
    try:
        dataset = getattr(trainer, "dual_stream_dataset", None)
        identities = {
            stream: _source_identity(getattr(dataset, f"{stream}_dataset", None))
            for stream in ("target", "anchor")
        }
        if metadata is not None:
            if (
                metadata["rank"] != trainer.args.process_index
                or metadata["world_size"] != trainer.accelerator.num_processes
            ):
                raise ValueError("LVD+ batch-plan rank/world size does not match the trainer")
            identities = metadata["source_datasets"]
        microbatches = [
            {
                "group_index": index // steps,
                "microbatch_index": index,
                "microbatch_in_group": index % steps,
                "global_slot": index * trainer.accelerator.num_processes
                + trainer.args.process_index,
                "tensor_sha256": _batch_fingerprint(batch),
                **{
                    f"{stream}_sample_ids": list(batch[stream]["sample_ids"])
                    if batch.get(stream) is not None and batch[stream].get("sample_ids") is not None
                    else None
                    if batch.get(stream) is not None
                    else []
                    for stream in ("target", "anchor")
                },
            }
            for index, batch in enumerate(batches)
        ]
        local = {
            "rank": trainer.args.process_index,
            "microbatches": microbatches,
            "source_datasets": identities,
            "frozen_head_sha256": _head_fingerprint(plugin),
            "lvd_plus_config_sha256": _sha256_json(asdict(trainer.lvd_plus_config)),
            "selection_metadata": metadata,
        }
        if trainer.lvd_plus_config.uses_boundary_risk:
            from sft_conflict.trainers.lvd_plus_recording import hybrid_objective_contract

            local["lvd_plus_config_sha256"] = _sha256_json(
                hybrid_objective_contract(trainer.lvd_plus_config)
            )
        s0 = _s0_provenance(trainer.lvd_plus_config, plugin)
        if s0:
            local["s0_identity"] = s0["s0_identity"]
        _sha256_json(local)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    statuses = _gather_calibration_objects({"plan": local, "error": error})
    errors = [
        f"rank {rank}: {status['error']}"
        for rank, status in enumerate(statuses)
        if status["error"] is not None
    ]
    if errors:
        raise RuntimeError("LVD+ calibration plan preflight failed; " + "; ".join(errors))
    ranks = [status["plan"] for status in statuses]
    for key in ("source_datasets", "frozen_head_sha256", "lvd_plus_config_sha256", "s0_identity"):
        if any(rank.get(key) != ranks[0].get(key) for rank in ranks):
            raise RuntimeError(f"LVD+ calibration ranks disagree on {key}")
    if [rank["rank"] for rank in ranks] != list(range(trainer.accelerator.num_processes)):
        raise RuntimeError("LVD+ calibration rank order does not match the distributed world")
    return {
        "plan_version": 1,
        "method": METHOD,
        "group_count": groups,
        "microbatches_per_group": steps,
        "world_size": trainer.accelerator.num_processes,
        "source": source,
        "ranks": ranks,
    }


def _summarize(values: dict[str, float]) -> dict[str, Any]:
    from sft_conflict.losses.lvd_plus import LVD_PLUS_CALIBRATION_METRIC_COUNTS

    return {
        "metrics": {
            name: values.get(name)
            if values.get(numerator) is not None and values.get(denominator, 0) > 0
            else None
            for name, (numerator, denominator) in LVD_PLUS_CALIBRATION_METRIC_COUNTS.items()
        },
        "counts": {
            key: values.get(key)
            for pair in LVD_PLUS_CALIBRATION_METRIC_COUNTS.values()
            for key in pair
        },
    }


def _set_resolved(trainer: SFTConflictTrainer, plugin: Any, value: float) -> None:
    plugin.set_lambda(value)
    trainer.resolved_lvd_plus_lambda = value
    trainer.lvd_plus_config.resolved_lambda = value
    plugin.config.resolved_lambda = value
    resolved = getattr(trainer, "resolved_config", None)
    if resolved is not None and resolved.method.name.uses_lvd_plus:
        resolved.lvd_plus.resolved_lambda = value


def _base_response_active(diagnostics: dict[str, Any]) -> dict[str, Any]:
    from sft_conflict.losses.lvd_plus import LVD_PLUS_CALIBRATION_METRIC_COUNTS

    names = [name for name in LVD_PLUS_CALIBRATION_METRIC_COUNTS if "response_active" in name]
    counts = {key for name in names for key in LVD_PLUS_CALIBRATION_METRIC_COUNTS[name]}
    return {
        "metrics": {name: diagnostics["metrics"].get(name) for name in names},
        "counts": {name: diagnostics["counts"].get(name) for name in sorted(counts)},
    }


@contextlib.contextmanager
def _preserve_calibration_state(trainer: SFTConflictTrainer, plugin: Any):
    """Preserve caller state, including failures and RNG-using explicit iterators."""
    models = (trainer.model, getattr(trainer, "model_wrapped", trainer.model))
    modes = {module: module.training for model in models for module in model.modules()}
    gradients = {
        parameter: (
            parameter.grad,
            parameter.grad.detach().cpu().clone() if parameter.grad is not None else None,
        )
        for model in models
        for parameter in model.parameters()
    }
    torch_rng, python_rng, numpy_rng = (
        torch.get_rng_state(),
        random.getstate(),
        np.random.get_state(),
    )
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    dataset = getattr(trainer, "dual_stream_dataset", None)
    cursor_state = {
        name: copy.deepcopy(getattr(dataset, name))
        for name in (
            "epoch",
            "_resume_epoch_guard",
            "_target_permutation",
            "_anchor_permutation",
            "_anchor_sequence",
        )
        if dataset is not None and hasattr(dataset, name)
    }
    previous_lambda = plugin.lambda_value
    try:
        with torch.enable_grad():
            yield
    finally:
        trainer.model.zero_grad(set_to_none=True)
        with torch.no_grad():
            for parameter, (original, saved) in gradients.items():
                if original is not None:
                    original.copy_(saved.to(original.device))
                parameter.grad = original
        trainer.model.train(modes[models[0]])
        for module, training in modes.items():
            module.training = training
        for module in trainer.model.modules():
            if module not in modes and isinstance(
                module, FullyShardedDataParallel | nn.parallel.DistributedDataParallel
            ):
                module.training = module.module.training
        for name, value in cursor_state.items():
            setattr(dataset, name, value)
        plugin.set_lambda(previous_lambda)
        torch.set_rng_state(torch_rng)
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)


def _prepare_once(trainer: SFTConflictTrainer) -> nn.Module:
    registered = getattr(trainer.accelerator, "_models", ())
    for candidate in (getattr(trainer, "model_wrapped", trainer.model), trainer.model):
        if (
            isinstance(candidate, FullyShardedDataParallel | nn.parallel.DistributedDataParallel)
            or getattr(candidate, "_is_accelerate_prepared", False)
            or candidate is getattr(trainer, "_lvd_plus_prepared_model", None)
            or any(candidate is model for model in registered)
        ):
            prepared = candidate
            break
    else:
        if trainer.is_fsdp_enabled and any(
            parameter.grad is not None for parameter in trainer.model.parameters()
        ):
            raise RuntimeError(
                "prepare FSDP before calibrating with existing gradients; fresh Base has none"
            )
        prepared = trainer.accelerator.prepare_model(trainer.model)
    trainer.model = trainer.model_wrapped = prepared
    trainer._lvd_plus_prepared_model = prepared
    return prepared


@contextlib.contextmanager
def reuse_lvd_plus_prepared_model(trainer: SFTConflictTrainer):
    """Do not let HF 4.48 recursively unwrap/rebuild a calibrated FSDP model.

    Distributed wrappers take Trainer's already-wrapped path. Its optimizer
    creation hook prepares the optimizer separately. CPU/native-AMP models
    retain the ordinary lifecycle, with only repeated model preparation elided.
    The override is scoped to this Trainer's train call, never global.
    """
    prepared = getattr(trainer, "_lvd_plus_prepared_model", None)
    if prepared is None:
        yield
        return
    accelerator = trainer.accelerator
    original_prepare = accelerator.prepare_model
    unwrapped = accelerator.unwrap_model(prepared)  # NOT recursive: keep child FSDP wrappers.
    trainer.model, trainer.model_wrapped = unwrapped, prepared
    trainer._lvd_plus_reuse_distributed_prepared = unwrapped is not prepared

    def prepare_once(model, *args, **kwargs):
        if model is prepared or model is unwrapped:
            if not any(model is prepared for model in accelerator._models):
                accelerator._models.append(prepared)
            return prepared
        return original_prepare(model, *args, **kwargs)

    accelerator.prepare_model = prepare_once
    try:
        yield
    finally:
        accelerator.prepare_model = original_prepare
        trainer._lvd_plus_reuse_distributed_prepared = False


def _write_report(
    trainer: SFTConflictTrainer, output_path: str | Path | None, report: dict[str, Any]
) -> None:
    from sft_conflict.trainers.sft_trainer import _gather_calibration_objects

    error = None
    if output_path is not None and trainer.is_world_process_zero():
        try:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_text(
                json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            temporary.replace(path)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    errors = [value for value in _gather_calibration_objects(error) if value is not None]
    if errors:
        raise RuntimeError("LVD+ calibration report write failed: " + "; ".join(errors))


def _failure_safe_report(report: dict[str, Any]) -> dict[str, Any]:
    nonfinite = {}

    def clean(value, path):
        if isinstance(value, float) and not math.isfinite(value):
            nonfinite[path] = repr(value)
            return None
        if isinstance(value, dict):
            return {key: clean(child, f"{path}.{key}") for key, child in value.items()}
        if isinstance(value, list | tuple):
            return [clean(child, f"{path}[{index}]") for index, child in enumerate(value)]
        return value

    result = clean(report, "report")
    if nonfinite:
        result["nonfinite_measurements"] = nonfinite
    return result


def calibrate_lvd_plus_lambda(
    trainer: SFTConflictTrainer,
    output_path: str | Path | None = None,
    *,
    batches: Iterable[dict[str, Any]] | None = None,
    _defer_resolved: bool = False,
) -> dict[str, Any] | None:
    """Measure RMS of post-average group gradient ratios once at fresh Base.

    Explicit input is a bounded rank-local CPU prefix; the default constructs
    the independent full-training-source plan, never a train DataLoader. Fixed
    and resolved scalars return without consuming batches, preparing a model,
    or overwriting an earlier report. Failed fresh measurements are written on
    rank zero before raising and never commit a candidate lambda.
    The hybrid record workflow defers the automatic result until semantic
    validation and exclusive publication; standalone callers retain their
    existing commit behavior.
    """
    from sft_conflict.trainers.sft_trainer import (
        _calibration_ratio_statistics,
        _explicit_calibration_batches,
    )

    plugin = trainer._lvd_plus_plugin()
    if plugin is None:
        return None
    config = trainer.lvd_plus_config
    report: dict[str, Any] = {
        "calibration_version": 1,
        "method": METHOD,
        "center_mode": (
            "recorded_s0"
            if config is not None and config.anchor_mode is LVDPlusAnchorMode.RECORDED_S0
            else CENTER_MODE
        ),
        "source": "fresh_base",
        "status": "running",
        "accepted": False,
        "calibration_statistic": "rms",
        "gradient_convention": "post_backward_distributed_average_compensated",
        "gradient_accumulation_convention": "mean_of_per_microbatch_losses",
        "target_gradient_norms": [],
        "auxiliary_gradient_norms": [],
        "auxiliary_to_target_ratios": [],
        "group_diagnostics": [],
        "resolved_lambda": None,
        "diagnostic_notes": {
            "gradients": (
                "FSDP sums disjoint reduced shards; DDP replica sqrt(world_size) cancels in "
                "ratios. Each group's gradients sum losses/steps, not norms or pooled pressure."
            ),
            "base_response_active": (
                "Observed on the original Base auxiliary forwards, separately from Target "
                "norms. Missing/empty denominators are null; no Base-health acceptance claim."
            ),
            "normalization": (
                "All groups including actual zero auxiliary norms enter RMS, without epsilon "
                "or dropping. Positive norms/weighted ratios must not underflow. The measured "
                "max weighted ratio <= beta*sqrt(G) is not a future-gradient bound."
            ),
            "data": (
                "Full supplied training sources, not holdouts. No rows are reserved or removed "
                "from training. Explicit tensor hashes bind data as well as ordered sample IDs."
            ),
        },
    }
    stage = "configuration"
    try:
        if config is None:
            raise RuntimeError("LVD+ plugin is missing its configuration")
        config.validate()
        s0 = _s0_provenance(config, plugin)
        report.update(s0)
        if config.uses_boundary_risk:
            if plugin.scale_identity is None:
                raise ValueError("hybrid lambda calibration requires verified SO/SR records")
            report.update(
                objective="pressure_plus_risk",
                scale_identity=plugin.scale_identity,
                outer_scale=config.resolved_outer_scale,
                risk_scale=config.resolved_risk_scale,
                risk_weight=config.risk_weight,
                risk_temperature=config.risk_temperature,
            )
        resolved = config.resolved_lambda
        if resolved is None and config.lambda_mode is not LVDLambdaMode.FIXED:
            resolved = getattr(trainer, "resolved_lvd_plus_lambda", None)
        if resolved is not None or config.lambda_mode is LVDLambdaMode.FIXED:
            source = "resolved_config" if resolved is not None else "fixed_lambda"
            value = resolved if resolved is not None else config.fixed_lambda
            _set_resolved(trainer, plugin, float(value))
            return {
                "method": METHOD,
                "center_mode": report["center_mode"],
                "mode": config.lambda_mode.value,
                "resolved_lambda": value,
                "source": source,
                **s0,
            }
        if getattr(getattr(trainer, "state", None), "global_step", 0) != 0:
            raise RuntimeError("unresolved LVD+ must be calibrated at fresh Base before training")
        if trainer.reference_model is not None or trainer.reference_cache is not None:
            raise ValueError("LVD+ calibration does not accept a reference model or cache")
        groups, steps = config.calibration_batches, config.calibration_accumulation_steps
        report.update(
            mode="gradient_ratio",
            lvd_plus_config=asdict(config),
            target_gradient_ratio=config.target_gradient_ratio,
            calibration_batches=groups,
            calibration_accumulation_steps=steps,
            calibration_seed=config.calibration_seed,
            world_size=trainer.accelerator.num_processes,
            training_gradient_accumulation_steps=trainer.args.gradient_accumulation_steps,
            calibration_batch_source="explicit" if batches is not None else "full_training_sources",
            calibration_loader_restarts=0,
        )
        with _preserve_calibration_state(trainer, plugin):
            stage = "batch_preflight"
            metadata = getattr(batches, "plan_metadata", None)

            def planned_batches():
                nonlocal metadata
                source = trainer.dual_stream_dataset
                generated = lvd_plus_calibration_batches(
                    SimpleNamespace(
                        lvd_plus=config, data=trainer.data_config, method=trainer.method_config
                    ),
                    source.target_dataset,
                    source.anchor_dataset,
                    trainer.pad_token_id,
                )
                metadata = generated.plan_metadata
                yield from generated

            buffered = _explicit_calibration_batches(
                batches if batches is not None else planned_batches(), groups=groups, steps=steps
            )
            if (
                config.uses_boundary_risk
                and isinstance(metadata, Mapping)
                and metadata.get("source") == "full_training_sources"
            ):
                # A bound hybrid plan is passed explicitly for replay, not
                # selected as an arbitrary explicit-batch calibration.
                report["calibration_batch_source"] = "full_training_sources"
            plan = _build_calibration_plan(
                trainer,
                plugin,
                buffered,
                metadata,
                groups=groups,
                steps=steps,
                source=report["calibration_batch_source"],
            )
            identities = plan["ranks"][0]["source_datasets"]
            report.update(
                calibration_plan=plan,
                calibration_plan_sha256=_sha256_json(plan),
                frozen_head_sha256=plan["ranks"][0]["frozen_head_sha256"],
                source_datasets=identities,
                target_dataset_fingerprint=identities["target"]["fingerprint"],
                anchor_dataset_fingerprint=identities["anchor"]["fingerprint"],
            )
            stage = "model_preparation"
            model = _prepare_once(trainer)
            parameters = tuple(
                parameter for parameter in model.parameters() if parameter.requires_grad
            )
            if not parameters:
                raise RuntimeError("LVD+ calibration found no trainable parameters")
            fsdp_plugin = getattr(getattr(trainer.accelerator, "state", None), "fsdp_plugin", None)
            offload = getattr(getattr(fsdp_plugin, "cpu_offload", None), "offload_params", False)
            if isinstance(model, FullyShardedDataParallel):
                offload = offload or model.cpu_offload.offload_params
            accumulator = CPUGradientAccumulator() if steps > 1 and offload else None
            pooled = CountAwareMetricAccumulator()
            plugin.set_lambda(1.0)
            model.eval()
            stage = "group_gradients"
            try:
                for group_index in range(groups):
                    group = buffered[group_index * steps : (group_index + 1) * steps]
                    group_metrics = CountAwareMetricAccumulator()
                    norms = []
                    for auxiliary in (False, True):
                        model.zero_grad(set_to_none=True)
                        for micro, batch in enumerate(group):
                            context = LossContext(
                                model,
                                trainer._prepare_inputs(batch),
                                reference_model=None,
                                capture_anchor_hidden=auxiliary,
                                capture_anchor_label_positions=auxiliary,
                                global_step=max(0, config.lambda_warmup_steps - 1),
                                compensate_distributed_gradient_average=True,
                            )
                            with trainer.compute_loss_context_manager():
                                result = (
                                    plugin(context) if auxiliary else TargetCrossEntropy()(context)
                                )
                            if auxiliary:
                                group_metrics.update(result)
                                pooled.update(result)
                            trainer.accelerator.backward(result.loss / steps)
                            if accumulator is not None:
                                accumulator.accumulate(model, optimizer_boundary=micro == steps - 1)
                            del context, result
                        norms.append(
                            trainer._distributed_gradient_norm(
                                tuple(parameter.grad for parameter in parameters),
                                trainer.accelerator.device,
                            )
                        )
                    model.zero_grad(set_to_none=True)
                    target_norm, auxiliary_norm = norms
                    report["target_gradient_norms"].append(target_norm)
                    report["auxiliary_gradient_norms"].append(auxiliary_norm)
                    report["group_diagnostics"].append(
                        {
                            "group_index": group_index,
                            "microbatch_count": steps,
                            **_summarize(group_metrics.compute(device=trainer.accelerator.device)),
                        }
                    )
                    if (
                        not math.isfinite(target_norm)
                        or target_norm <= 0
                        or not math.isfinite(auxiliary_norm)
                        or auxiliary_norm < 0
                    ):
                        raise RuntimeError(
                            "LVD+ requires finite positive Target and non-negative "
                            "auxiliary gradient norms"
                        )
                    ratio = auxiliary_norm / target_norm
                    if not math.isfinite(ratio) or (ratio <= 0 and auxiliary_norm != 0):
                        raise RuntimeError(
                            "LVD+ gradient ratio is non-finite or "
                            "positive-norm arithmetic underflow"
                        )
                    report["auxiliary_to_target_ratios"].append(ratio)
            finally:
                if accumulator is not None:
                    accumulator.buffers.clear()
            stage = "normalization_and_stability"
            ratios = report["auxiliary_to_target_ratios"]
            pooled_diagnostics = _summarize(pooled.compute(device=trainer.accelerator.device))
            report.update(
                calibration_group_count=len(ratios),
                calibration_microbatch_count=groups * steps,
                calibration_total_microbatch_count=groups
                * steps
                * trainer.accelerator.num_processes,
                pooled_diagnostics=pooled_diagnostics,
                base_response_active=_base_response_active(pooled_diagnostics),
                ratio_statistics=_calibration_ratio_statistics(ratios, allow_zeros=True),
                median_auxiliary_to_target_ratio=calibration_normalizer(ratios),
                zero_auxiliary_groups=sum(norm == 0 for norm in report["auxiliary_gradient_norms"]),
                positive_auxiliary_groups=sum(
                    norm > 0 for norm in report["auxiliary_gradient_norms"]
                ),
            )
            stability = lvd_plus_rms_stability_decision(
                report, minimum_groups=8 if groups >= 8 else 2, allow_single_group=groups == 1
            )
            report["stability"] = stability
            normalizer = calibration_normalizer(ratios, "rms")
            report.update(normalization_value=normalizer, rms_auxiliary_to_target_ratio=normalizer)
            if normalizer <= 0:
                raise RuntimeError(
                    "RMS auxiliary gradient ratio is zero; calibration cannot normalize"
                )
            resolved = config.target_gradient_ratio / normalizer
            if not math.isfinite(resolved) or resolved <= 0:
                raise RuntimeError("LVD+ calibration produced a non-finite or underflowed lambda")
            weighted = [resolved * ratio for ratio in ratios]
            if any(
                not math.isfinite(value) or (value == 0 and raw > 0)
                for value, raw in zip(weighted, ratios, strict=True)
            ):
                raise RuntimeError(
                    "LVD+ calibration produced a non-finite or underflowed weighted gradient ratio"
                )
            weighted_rms = calibration_normalizer(weighted, "rms")
            if weighted_rms <= 0:
                raise RuntimeError("LVD+ calibration produced an invalid weighted RMS")
            report.update(
                candidate_lambda=resolved,
                weighted_auxiliary_to_target_ratios=weighted,
                rms_weighted_auxiliary_to_target_ratio=weighted_rms,
                median_weighted_auxiliary_to_target_ratio=calibration_normalizer(weighted),
            )
            if not stability["accepted"]:
                raise RuntimeError(
                    "LVD+ RMS stability rejected: " + "; ".join(stability["reasons"])
                )
            plugin.set_lambda(resolved)  # Validate before accepting/writing; finally restores it.
        report.update(status="passed", accepted=True, resolved_lambda=resolved)
        json.dumps(report, allow_nan=False)
    except Exception as exc:
        report.update(
            status="failed",
            accepted=False,
            resolved_lambda=None,
            failure_stage=stage,
            error=f"{type(exc).__name__}: {exc}",
        )
        _write_report(trainer, output_path, _failure_safe_report(report))
        raise RuntimeError(f"LVD+ calibration failed at {stage}: {exc}") from exc
    _write_report(trainer, output_path, report)
    if not _defer_resolved:
        _set_resolved(trainer, plugin, resolved)
    return report
