"""Explicit, immutable Base scale/lambda records for the hybrid LVD+ objective.

Records bind actual initial model values, frozen head/S0, data fingerprints,
collated calibration tensors, objective, precision and distributed layout.
No recording operation calls Trainer.train or creates/steps an optimizer.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import stat
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from sft_conflict.config import LVDLambdaMode
from sft_conflict.config.io import _decode_dataclass
from sft_conflict.data.lvd_plus_records import (
    LVDPlusRecord,
    canonical_sha256,
    publish_lvd_plus_record,
)
from sft_conflict.losses import LossContext
from sft_conflict.lvd_plus_workflow import lvd_plus_calibration_batches
from sft_conflict.trainers.lvd_plus_calibration import (
    CENTER_MODE,
    METHOD,
    _build_calibration_plan,
    _finite_number,
    _prepare_once,
    _preserve_calibration_state,
    _set_resolved,
    _tensor_digest,
    _write_report,
    lvd_plus_rms_stability_decision,
)

_RECORD_FIELDS = {
    "resolved_lambda",
    "resolved_outer_scale",
    "resolved_risk_scale",
    "scale_cache_path",
    "scale_cache_sha256",
    "lambda_cache_path",
    "lambda_cache_sha256",
    "s0_cache_path",
    "s0_cache_sha256",
}


def _plain(value: Any) -> Any:
    # Enum values and tuples become canonical plain JSON; record readers reject
    # arbitrary Python objects rather than silently stringify them.
    return json.loads(json.dumps(value, allow_nan=False))


def _normalized_config(config) -> dict[str, Any]:
    """Use YAML's declared numeric types without rounding or mutating the caller."""
    decoded = _decode_dataclass(type(config), _plain(asdict(config)), type(config).__name__)
    return _plain(asdict(decoded))


def hybrid_objective_contract(config) -> dict[str, Any]:
    """Exclude output locations/resolved measurements, never objective settings."""
    return {
        key: value for key, value in _normalized_config(config).items() if key not in _RECORD_FIELDS
    }


def record_recipe(config, world_size: int) -> dict[str, Any]:
    data = _normalized_config(config.data)
    data.pop("validation", None)
    data.pop("steps_per_epoch", None)
    # Worker count is execution detail, not the independently collated plan.
    data.pop("num_workers", None)
    return _plain(
        {
            "version": 1,
            "method": config.method.name.value,
            "model": _normalized_config(config.model),
            "lora": _normalized_config(config.lora),
            "data": data,
            "lvd_plus": hybrid_objective_contract(config.lvd_plus),
            "s0_solver": _normalized_config(config.s0),
            "precision": config.training.mixed_precision,
            "training_seed": config.training.seed,
            "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
            "distributed_strategy": config.training.distributed_strategy,
            "fsdp_cpu_offload": config.training.fsdp_cpu_offload,
            "fsdp_keep_low_precision_params": config.training.fsdp_keep_low_precision_params,
            "fsdp_sync_each_batch": config.training.fsdp_sync_each_batch,
            "world_size": world_size,
        }
    )


def _canonical_record_path(path: str | Path) -> Path:
    # As in exclusive publication, preserve the final directory entry. In
    # particular, a dangling symlink must not redirect a new measurement.
    requested = Path(path).expanduser()
    return requested.parent.resolve() / requested.name


def resolve_record_paths(config, world_size: int) -> tuple[Path, Path]:
    """Derive at the actual worker world size; paths alone never prove binding."""
    plus = config.lvd_plus
    key = canonical_sha256(record_recipe(config, world_size))
    root = Path(config.paths.artifact_root).expanduser().resolve() / "lvd_plus_records" / key
    outer = (
        _canonical_record_path(plus.scale_cache_path)
        if plus.scale_cache_path
        else root / "scales.json"
    )
    coefficient = (
        _canonical_record_path(plus.lambda_cache_path)
        if plus.lambda_cache_path
        else root / "lambda.json"
    )
    if outer == coefficient:
        raise ValueError("LVD+ scale and lambda records must have different paths")
    plus.scale_cache_path, plus.lambda_cache_path = str(outer), str(coefficient)
    return outer, coefficient


def _model_fingerprint(model) -> str:
    digest = hashlib.sha256(b"lvd-plus-initial-model-v1")
    for name, value in model.state_dict().items():
        if not isinstance(value, torch.Tensor):
            raise ValueError("LVD+ initial model state must contain tensors only")
        digest.update(name.encode())
        _tensor_digest(digest, value)
    return digest.hexdigest()


def _collective_result(operation):
    """Propagate a local read/binding error before any rank enters a forward."""
    from sft_conflict.trainers.sft_trainer import _gather_calibration_objects

    value, error = None, None
    try:
        value = operation()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    statuses = _gather_calibration_objects(error)
    if any(status is not None for status in statuses):
        raise ValueError(
            "LVD+ record preflight failed: "
            + "; ".join(
                f"rank {rank}: {message}" for rank, message in enumerate(statuses) if message
            )
        )
    return value


def _collective_record_exists(path: str | Path) -> bool:
    """Choose reuse/measurement together, including at world size one.

    Only the local filesystem probe is caught. A later record read must use
    its own collective error path, never fall back to fresh measurement.
    Publication remains exclusive if the destination appears after this check.
    """
    from sft_conflict.trainers.sft_trainer import _gather_calibration_objects

    local = {"path": None, "exists": False, "error": None}
    try:
        candidate = _canonical_record_path(path)
        local["path"] = str(candidate)
        local["exists"] = os.path.lexists(candidate)
        if local["exists"]:
            # stat follows a final symlink: broken links and non-files fail,
            # while valid links to immutable files may be read normally.
            if not stat.S_ISREG(candidate.stat().st_mode):
                raise ValueError("LVD+ record path must be a regular file")
        else:
            # lexists suppresses OSError. Confirm genuine absence rather than
            # treating a permission/I/O error (or a concurrent creation) as it.
            try:
                candidate.lstat()
            except FileNotFoundError:
                pass
            else:
                raise ValueError("LVD+ record path changed during availability preflight")
    except Exception as exc:
        local["error"] = f"{type(exc).__name__}: {exc}"
    statuses = _gather_calibration_objects(local)
    errors = [
        f"rank {rank}: {status['error']}"
        for rank, status in enumerate(statuses)
        if status["error"] is not None
    ]
    if errors:
        raise ValueError("LVD+ record availability preflight failed: " + "; ".join(errors))
    if len({status["path"] for status in statuses}) != 1:
        raise ValueError("LVD+ ranks disagree on the canonical record path")
    if len({status["exists"] for status in statuses}) != 1:
        raise ValueError("LVD+ ranks disagree on record existence; refusing reuse/measurement")
    return statuses[0]["exists"]


def _context(trainer):
    plugin = trainer._lvd_plus_plugin()
    config = trainer.resolved_config
    if plugin is None or config is None or not config.lvd_plus.uses_boundary_risk:
        raise ValueError("recording requires hybrid LVD+ with a complete resolved_config")
    config.validate()
    if trainer.state.global_step != 0:
        raise ValueError("LVD+ records must be prepared at the fresh initial model")
    existing = getattr(trainer, "_lvd_plus_record_context", None)
    contract = canonical_sha256(record_recipe(config, trainer.accelerator.num_processes))
    if existing is not None:
        if existing["recipe_sha256"] != contract:
            raise ValueError("LVD+ recording recipe changed after initial binding")
        return existing
    if getattr(trainer, "_lvd_plus_prepared_model", None) is not None:
        raise ValueError("bind hybrid records before preparing the model for distributed execution")
    source = trainer.dual_stream_dataset

    def prepare_local():
        batches = lvd_plus_calibration_batches(
            config, source.target_dataset, source.anchor_dataset, trainer.pad_token_id
        )
        return batches, _model_fingerprint(trainer.model)

    batches, fingerprint = _collective_result(prepare_local)
    plus = config.lvd_plus
    plan = _build_calibration_plan(
        trainer,
        plugin,
        batches,
        batches.plan_metadata,
        groups=plus.calibration_batches,
        steps=plus.calibration_accumulation_steps,
        source="full_training_sources",
    )
    tokenizer = trainer.preview_tokenizer
    tokenization = {
        "pad_token_id": trainer.pad_token_id,
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "padding_side": getattr(tokenizer, "padding_side", "right"),
        "vocabulary_sha256": canonical_sha256(tokenizer.get_vocab())
        if tokenizer is not None and hasattr(tokenizer, "get_vocab")
        else None,
    }
    binding = {
        "contract_version": 1,
        "recipe": record_recipe(config, trainer.accelerator.num_processes),
        "initial_model_sha256": fingerprint,
        "frozen_head_sha256": plan["ranks"][0]["frozen_head_sha256"],
        "s0_identity": copy.deepcopy(plugin.s0_identity),
        "tokenization": tokenization,
        "calibration_plan": plan,
        "calibration_plan_sha256": canonical_sha256(plan),
    }
    from sft_conflict.trainers.sft_trainer import _gather_calibration_objects

    identities = _gather_calibration_objects(canonical_sha256(binding))
    if len(set(identities)) != 1:
        raise ValueError("LVD+ ranks disagree on actual initial model or record binding")
    context = {
        "batches": batches,
        "binding": binding,
        "recipe_sha256": contract,
        "paths": _collective_result(
            lambda: resolve_record_paths(config, trainer.accelerator.num_processes)
        ),
    }
    trainer._lvd_plus_record_context = context
    return context


def _load_record(trainer, path, kind, binding, expected_sha):
    def read():
        if not Path(path).is_file():
            action = (
                "record_lvd_plus_scales"
                if kind == "lvd_plus_scales"
                else "calibrate_lvd_plus_lambda"
            )
            raise FileNotFoundError(f"Missing {kind}: {path}; run METHOD='{action}' first")
        record = LVDPlusRecord(path)
        if record.kind != kind:
            raise ValueError(f"Expected {kind}, not {record.kind}")
        record.validate_binding(binding, expected_sha256=expected_sha)
        return record

    record = _collective_result(read)
    from sft_conflict.trainers.sft_trainer import _gather_calibration_objects

    if len(set(_gather_calibration_objects(record.sha256))) != 1:
        raise ValueError("LVD+ ranks loaded different immutable records")
    return record


def _validate_lambda_report_semantics(trainer, report, binding) -> None:
    """Local hybrid workflow checks; callers serialize failures collectively.

    The generic record reader owns arithmetic/integrity, not experiment beta,
    provenance or RMS policy. Do not apply this private contract to standalone
    pressure-only reports, and do not repair contradictory report fields here.
    """
    if not isinstance(report, dict):
        raise ValueError("hybrid lambda measurement requires an accepted fresh report")
    recipe, plan = binding["recipe"], binding["calibration_plan"]
    objective = recipe["lvd_plus"]
    if (
        not trainer.lvd_plus_config.uses_boundary_risk
        or recipe["method"] != METHOD
        or objective["objective_mode"] != "pressure_plus_risk"
        or objective["lambda_mode"] != "gradient_ratio"
    ):
        raise ValueError("LVD+ lambda record requires the automatic hybrid objective")
    actual_recipe = record_recipe(trainer.resolved_config, trainer.accelerator.num_processes)
    if canonical_sha256(actual_recipe) != canonical_sha256(recipe):
        raise ValueError("LVD+ lambda recipe changed after initial binding")
    groups = objective["calibration_batches"]
    steps = objective["calibration_accumulation_steps"]
    world = recipe["world_size"]
    for key, expected in {
        "method": METHOD,
        "source": "full_training_sources",
        "group_count": groups,
        "microbatches_per_group": steps,
        "world_size": world,
    }.items():
        if canonical_sha256(plan.get(key)) != canonical_sha256(expected):
            raise ValueError(f"LVD+ bound calibration plan {key} differs from its recipe")
    plan_sha = binding["calibration_plan_sha256"]
    if (
        canonical_sha256(plan) != plan_sha
        or canonical_sha256(report.get("calibration_plan")) != plan_sha
    ):
        raise ValueError("LVD+ lambda report calibration_plan differs from the bound plan")
    sources = plan["ranks"][0]["source_datasets"]
    expected_fields = {
        "calibration_version": 1,
        "method": METHOD,
        "objective": objective["objective_mode"],
        "mode": objective["lambda_mode"],
        "center_mode": "recorded_s0"
        if objective["anchor_mode"] == "recorded_s0"
        else CENTER_MODE,
        "source": "fresh_base",
        "status": "passed",
        "accepted": True,
        "calibration_statistic": "rms",
        "gradient_convention": "post_backward_distributed_average_compensated",
        "gradient_accumulation_convention": "mean_of_per_microbatch_losses",
        "target_gradient_ratio": objective["target_gradient_ratio"],
        "risk_weight": objective["risk_weight"],
        "risk_temperature": objective["risk_temperature"],
        "outer_scale": binding["scales"]["outer_scale"],
        "risk_scale": binding["scales"]["risk_scale"],
        "scale_identity": binding["scale_identity"],
        "frozen_head_sha256": binding["frozen_head_sha256"],
        "source_datasets": sources,
        "target_dataset_fingerprint": sources["target"]["fingerprint"],
        "anchor_dataset_fingerprint": sources["anchor"]["fingerprint"],
        "calibration_plan_sha256": plan_sha,
        "calibration_batch_source": "full_training_sources",
        "calibration_batches": groups,
        "calibration_group_count": groups,
        "calibration_accumulation_steps": steps,
        "calibration_seed": objective["calibration_seed"],
        "world_size": world,
        "training_gradient_accumulation_steps": recipe["gradient_accumulation_steps"],
        "calibration_microbatch_count": groups * steps,
        "calibration_total_microbatch_count": groups * steps * world,
        "calibration_loader_restarts": 0,
        "s0_identity": binding["s0_identity"],
        "s0_cache_sha256": binding["s0_identity"]["manifest_sha256"]
        if binding["s0_identity"] is not None
        else None,
    }
    float_fields = {
        "target_gradient_ratio",
        "risk_weight",
        "risk_temperature",
        "outer_scale",
        "risk_scale",
    }
    for key, expected in expected_fields.items():
        actual = report.get(key)
        if key in float_fields:
            # Config floats accept integer input, never bool or rounding. This
            # semantic comparison does not relax the generic JSON/hash reader.
            actual, expected = _finite_number(actual), _finite_number(expected)
        if canonical_sha256(actual) != canonical_sha256(expected):
            raise ValueError(f"LVD+ lambda report {key} differs from the bound calibration")
    report_config = _decode_dataclass(
        type(trainer.lvd_plus_config), report["lvd_plus_config"], "report.lvd_plus_config"
    )
    plugin = trainer._lvd_plus_plugin()
    for config in (report_config, trainer.lvd_plus_config, plugin.config):
        if canonical_sha256(hybrid_objective_contract(config)) != canonical_sha256(objective):
            raise ValueError("LVD+ lambda report/config lvd_plus_config differs from its recipe")
        for name, expected in (
            ("resolved_outer_scale", binding["scales"]["outer_scale"]),
            ("resolved_risk_scale", binding["scales"]["risk_scale"]),
        ):
            if _finite_number(getattr(config, name)) != _finite_number(expected):
                raise ValueError(f"LVD+ lambda report/config {name} differs from its scale record")
        if config.scale_cache_sha256 != binding["scale_identity"]["manifest_sha256"]:
            raise ValueError("LVD+ lambda report/config scale_cache_sha256 differs from its record")
        if config.s0_cache_sha256 != expected_fields["s0_cache_sha256"]:
            raise ValueError("LVD+ lambda report/config s0_cache_sha256 differs from its identity")
    if canonical_sha256(plugin.scale_identity) != canonical_sha256(binding["scale_identity"]):
        raise ValueError("LVD+ lambda plugin scale_identity differs from its record")
    if canonical_sha256(plugin.s0_identity) != canonical_sha256(binding["s0_identity"]):
        raise ValueError("LVD+ lambda plugin s0_identity differs from its binding")
    if trainer.args.gradient_accumulation_steps != recipe["gradient_accumulation_steps"]:
        raise ValueError("LVD+ lambda trainer accumulation differs from its recipe")
    stability = lvd_plus_rms_stability_decision(
        report, minimum_groups=8 if groups >= 8 else 2, allow_single_group=groups == 1
    )
    if not stability["accepted"]:
        raise ValueError("LVD+ lambda RMS stability rejected: " + "; ".join(stability["reasons"]))
    if canonical_sha256(report.get("stability")) != canonical_sha256(stability):
        raise ValueError("LVD+ lambda recorded stability differs from recomputed RMS policy")


def _validate_lambda_record_semantics(trainer, record: LVDPlusRecord, binding) -> None:
    """Reject incompatible or stale accepted evidence on every rank before binding lambda."""

    def validate():
        _validate_lambda_report_semantics(trainer, record.report, binding)
        value = record.values["resolved_lambda"]
        for resolved in (
            trainer.lvd_plus_config.resolved_lambda,
            trainer.resolved_config.lvd_plus.resolved_lambda,
            trainer.resolved_lvd_plus_lambda,
        ):
            if resolved is not None and resolved != value:
                raise ValueError("resolved lambda differs from the bound calibration record")

    _collective_result(validate)


def _publish(trainer, path, kind, binding, values, report):
    # Calibration reports contain str-Enum config values; convert explicitly
    # at the producer boundary rather than weakening the immutable reader.
    binding, values, report = _plain(binding), _plain(values), _plain(report)

    def write():
        if trainer.is_world_process_zero():
            publish_lvd_plus_record(path, kind=kind, binding=binding, values=values, report=report)

    _collective_result(write)
    return _load_record(trainer, path, kind, binding, None)


def bind_lvd_plus_scales(trainer) -> LVDPlusRecord:
    context = _context(trainer)
    plus = trainer.lvd_plus_config
    record = _load_record(
        trainer, context["paths"][0], "lvd_plus_scales", context["binding"], plus.scale_cache_sha256
    )

    def bind():
        trainer._lvd_plus_plugin().bind_scales(
            record.values["outer_scale"], record.values["risk_scale"], record.identity()
        )
        resolved = trainer.resolved_config.lvd_plus
        resolved.resolved_outer_scale = record.values["outer_scale"]
        resolved.resolved_risk_scale = record.values["risk_scale"]
        resolved.scale_cache_sha256 = record.sha256

    _collective_result(bind)
    return record


def record_lvd_plus_scales(trainer, output_path: str | Path | None = None) -> dict[str, Any]:
    """Record arithmetic means of global per-micro losses, before any updates."""
    if trainer.resolved_config.training.resume_from_checkpoint:
        raise ValueError("record_lvd_plus_scales does not accept a training resume checkpoint")
    context = _context(trainer)
    path = context["paths"][0]
    plugin = trainer._lvd_plus_plugin()
    if _collective_record_exists(path):
        record = bind_lvd_plus_scales(trainer)
        result = {
            "action": "record_lvd_plus_scales",
            "reused": True,
            "path": str(path),
            "identity": record.identity(),
            **record.values,
        }
        _write_report(trainer, output_path, result)
        return result
    report = {
        "status": "running",
        "accepted": False,
        "method": "target_plus_anchor_lvd+",
        "source": "fresh_base",
        "aggregation": "arithmetic_mean_of_global_micro_losses",
        "per_micro_outer": [],
        "per_micro_risk": [],
        "micro_position_counts": [],
        "microbatch_count": len(context["batches"]),
        "optimizer_created": False,
        "parameter_updates": 0,
        "gradients_computed": False,
        "calibration_plan_sha256": context["binding"]["calibration_plan_sha256"],
    }
    try:
        with _preserve_calibration_state(trainer, plugin):
            model = _prepare_once(trainer)
            model.eval()
            with torch.no_grad():
                for index, batch in enumerate(context["batches"]):
                    loss_context = LossContext(
                        model,
                        trainer._prepare_inputs(batch),
                        reference_model=None,
                        capture_anchor_hidden=True,
                        capture_anchor_label_positions=True,
                    )
                    with trainer.compute_loss_context_manager():
                        raw = plugin.raw_components(loss_context)
                    outer, risk, count = (
                        float(raw.outer_loss),
                        float(raw.risk_loss),
                        raw.global_position_count,
                    )
                    if count <= 0 or not all(math.isfinite(x) and x >= 0 for x in (outer, risk)):
                        raise ValueError(
                            "scale recording requires nonempty finite global micro losses"
                        )
                    report["per_micro_outer"].append(outer)
                    report["per_micro_risk"].append(risk)
                    report["micro_position_counts"].append(count)
                    del raw, loss_context
                    if trainer.is_world_process_zero() and (index + 1) % 32 == 0:
                        print(
                            f"LVD+ scales: {index + 1}/{len(context['batches'])} micros", flush=True
                        )
        values = {
            "outer_scale": math.fsum(report["per_micro_outer"]) / report["microbatch_count"],
            "risk_scale": math.fsum(report["per_micro_risk"]) / report["microbatch_count"],
        }
        if not all(math.isfinite(value) and value > 0 for value in values.values()):
            raise ValueError("both recorded LVD+ scales must be finite and strictly positive")
        report.update(status="passed", accepted=True)
        record = _publish(trainer, path, "lvd_plus_scales", context["binding"], values, report)
        plugin.bind_scales(values["outer_scale"], values["risk_scale"], record.identity())
    except Exception as exc:
        report.update(status="failed", accepted=False, error=f"{type(exc).__name__}: {exc}")
        _write_report(trainer, output_path, report)
        raise
    result = {
        "action": "record_lvd_plus_scales",
        "reused": False,
        "path": str(path),
        "identity": record.identity(),
        **values,
        "recording": report,
    }
    _write_report(trainer, output_path, result)
    return result


def calibrate_lvd_plus_record(trainer, output_path: str | Path | None = None) -> dict[str, Any]:
    """Validate scales, then reuse or measure/publish the complete hybrid lambda."""
    context = _context(trainer)
    scale = bind_lvd_plus_scales(trainer)
    plus = trainer.lvd_plus_config
    plugin = trainer._lvd_plus_plugin()
    if plus.lambda_mode is LVDLambdaMode.FIXED:
        # Explicit fixed-lambda experiments still require verified scales but
        # do not manufacture an automatic-RMS record or run any backward.
        from sft_conflict.trainers.lvd_plus_calibration import calibrate_lvd_plus_lambda

        return calibrate_lvd_plus_lambda(trainer, output_path)
    binding = {**context["binding"], "scale_identity": scale.identity(), "scales": scale.values}
    path = context["paths"][1]
    if _collective_record_exists(path):
        record = _load_record(trainer, path, "lvd_plus_lambda", binding, plus.lambda_cache_sha256)
        _validate_lambda_record_semantics(trainer, record, binding)
        value = record.values["resolved_lambda"]
        _collective_result(lambda: _set_resolved(trainer, plugin, value))
        for conf in (plus, trainer.resolved_config.lvd_plus):
            conf.lambda_cache_sha256 = record.sha256
        result = {
            **record.report,
            "source": "verified_lambda_record",
            "reused_calibration": True,
            "record_path": str(path),
            "lambda_identity": record.identity(),
            "scale_identity": scale.identity(),
        }
        _write_report(trainer, output_path, result)
        return result
    if (
        plus.lambda_cache_sha256 is not None
        or trainer.resolved_config.training.resume_from_checkpoint
    ):
        raise FileNotFoundError(
            "the pinned hybrid lambda record is missing; cannot recalibrate a resume"
        )
    if plus.resolved_lambda is not None or trainer.resolved_lvd_plus_lambda is not None:
        raise ValueError(
            "hybrid lambda must come from its verified record, not a bare resolved scalar"
        )
    # Existing calibrator owns exact grouped parameter gradients and stability
    # gates; it calls the newly bound combined plugin, never the old outer alone.
    from sft_conflict.trainers.lvd_plus_calibration import calibrate_lvd_plus_lambda

    report = calibrate_lvd_plus_lambda(
        trainer, output_path, batches=context["batches"], _defer_resolved=True
    )
    _collective_result(lambda: _validate_lambda_report_semantics(trainer, report, binding))
    record = _publish(
        trainer,
        path,
        "lvd_plus_lambda",
        binding,
        {"resolved_lambda": report["resolved_lambda"]},
        report,
    )
    _validate_lambda_record_semantics(trainer, record, binding)
    _collective_result(lambda: _set_resolved(trainer, plugin, record.values["resolved_lambda"]))
    for conf in (plus, trainer.resolved_config.lvd_plus):
        conf.lambda_cache_sha256 = record.sha256
    result = {
        **report,
        "record_path": str(path),
        "lambda_identity": record.identity(),
        "reused_calibration": False,
    }
    _write_report(trainer, output_path, result)
    return result