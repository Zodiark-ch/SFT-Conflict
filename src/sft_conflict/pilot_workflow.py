"""Health-gated pilots and reproducible disjoint calibration/training streams."""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import math
import os
import statistics
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch.distributed as distributed
from transformers import TrainerCallback

from sft_conflict.config import (
    ConfigError,
    LVDResponseMode,
    MethodName,
    ReferenceConfig,
    dump_config,
)
from sft_conflict.data import CausalLMCollator, ReferenceCache
from sft_conflict.losses import build_loss_plugins
from sft_conflict.models import get_unembedding, load_model_bundle
from sft_conflict.pilot_protocol import (
    IndexedDatasetView,
    build_anchor_plan,
    evaluate_anchor_health,
    find_accepted_pilot,
    pilot_health_decision,
    write_accepted_pilot_registry,
)
from sft_conflict.pilot_stability import calibration_normalizer, calibration_stability_decision
from sft_conflict.tracking import prepare_run
from sft_conflict.tracking.metadata import write_json_atomic


def _world() -> tuple[int, int]:
    if distributed.is_available() and distributed.is_initialized():
        return distributed.get_rank(), distributed.get_world_size()
    return int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))


def anchor_plan_for_config(config, anchor) -> dict[str, Any]:
    if config.lvd.calibration_seed is None:
        raise ConfigError("guarded pilot generation requires lvd.calibration_seed")
    _, world_size = _world()
    return build_anchor_plan(
        anchor,
        seed=config.lvd.calibration_seed,
        calibration_samples=(
            config.lvd.calibration_batches
            * config.lvd.calibration_accumulation_steps
            * world_size
            * config.data.anchor_batch_size
        ),
        health_samples=config.lvd.pilot_health_samples,
        validation_samples=config.lvd.pilot_validation_samples,
    )


def resolve_pilot_checkpoint(config) -> str:
    if config.lvd.calibration_checkpoint:
        return config.lvd.calibration_checkpoint
    try:
        return find_accepted_pilot(config)["checkpoint"]
    except ValueError as error:
        raise ConfigError(str(error)) from error


def _require_dataset_fingerprint(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(
            f"accepted pilot publication requires a valid {name}; "
            "run generate_pilot with the matching formal source config"
        )
    return value


def accepted_pilot_calibration(config) -> dict[str, Any] | None:
    """Rescale an exact-key publication's measurements without loading a model.

    Explicit checkpoints and legacy unseeded calibration keep their existing
    route. Automatic lookup must pass the full publication validator; invalid
    or old evidence never silently falls back to loading/recalibrating a pilot.
    The caller must also validate the actual loaded Target/Anchor fingerprints
    before constructing the training view. Only a validated scalar changes config.
    The configured statistic must match the measurement (absence means median).
    RMS reuse retains raw medians and all groups, including legitimate zeros.
    """
    if config.lvd.calibration_checkpoint is not None or config.lvd.calibration_seed is None:
        return None
    statistic = config.lvd.calibration_statistic
    if not isinstance(statistic, str) or statistic not in {"median", "rms"}:
        raise ConfigError("lvd.calibration_statistic must be 'median' or 'rms'")
    if statistic == "rms" and config.lvd.response_mode != LVDResponseMode.THREE_SEGMENT:
        raise ConfigError("lvd.calibration_statistic='rms' requires response_mode='three_segment'")
    ratio = config.lvd.target_gradient_ratio
    try:
        valid_ratio = (
            isinstance(ratio, int | float)
            and not isinstance(ratio, bool)
            and math.isfinite(ratio)
            and ratio > 0
        )
    except OverflowError:
        valid_ratio = False
    if not valid_ratio:
        raise ConfigError("lvd.target_gradient_ratio must be a finite positive number")
    try:
        manifest = find_accepted_pilot(config)
    except ValueError as error:
        raise ConfigError(str(error)) from error
    publication = manifest["report"]
    target_fingerprint = _require_dataset_fingerprint(
        publication.get("target_dataset_fingerprint"), "target_dataset_fingerprint"
    )
    anchor_fingerprint = _require_dataset_fingerprint(
        publication["baseline"].get("dataset_fingerprint"), "anchor_dataset_fingerprint"
    )
    cache_fingerprint = publication["baseline"]["cache_fingerprint"]
    try:
        if not config.lvd.d0_cache_path:
            raise ValueError("lvd.d0_cache_path is missing")
        cache_path = Path(config.lvd.d0_cache_path).expanduser() / "manifest.json"
        cache_manifest = json.loads(cache_path.read_text(encoding="utf-8"))
        if not isinstance(cache_manifest, dict):
            raise ValueError("manifest must be a JSON object")
        # Match the health report's exact canonical JSON SHA, including numeric
        # types; config-key numeric normalization is NOT valid for cache evidence.
        current_fingerprint = hashlib.sha256(
            json.dumps(
                cache_manifest, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
        ).hexdigest()
    except (OSError, TypeError, ValueError, OverflowError) as error:
        raise ConfigError(f"cannot validate pilot calibration cache manifest: {error}") from error
    if current_fingerprint != cache_fingerprint:
        raise ConfigError(
            "accepted pilot calibration cache fingerprint changed; "
            "run generate_pilot with the matching formal source config"
        )

    report = copy.deepcopy(publication["calibration"])
    reported_statistic = report.get("calibration_statistic", "median")
    if not isinstance(reported_statistic, str) or reported_statistic not in {"median", "rms"}:
        raise ConfigError("accepted pilot calibration_statistic must be 'median' or 'rms'")
    if reported_statistic != statistic:
        raise ConfigError(
            "accepted pilot calibration_statistic does not match lvd.calibration_statistic; "
            "run generate_pilot with the matching formal source config"
        )
    stability = calibration_stability_decision(
        report, minimum_groups=max(2, min(8, config.lvd.calibration_batches))
    )
    if not stability["accepted"]:
        raise ConfigError(
            f"accepted pilot calibration failed stability checks: {stability['reasons']}"
        )
    try:
        raw_ratios = report["auxiliary_to_target_ratios"]
        median = (
            calibration_normalizer(raw_ratios)
            if statistic == "rms"
            else statistics.median(raw_ratios)
        )
        normalizer = calibration_normalizer(raw_ratios, statistic) if statistic == "rms" else median
        resolved = ratio / normalizer
        weighted = [resolved * raw for raw in raw_ratios]
        weighted_median = (
            calibration_normalizer(weighted) if statistic == "rms" else statistics.median(weighted)
        )
        if statistic == "rms":
            weighted_rms = calibration_normalizer(weighted, statistic)
            positive_values = (normalizer, resolved, weighted_rms)
        else:
            positive_values = (median, resolved, weighted_median)
        if not all(math.isfinite(value) and value > 0 for value in positive_values) or any(
            not math.isfinite(value) or value < 0 for value in weighted
        ):
            raise ValueError("non-finite or underflowed lambda/weighted ratios")
    except (ArithmeticError, ValueError) as error:
        raise ConfigError(f"accepted pilot calibration cannot be rescaled: {error}") from error
    report.update(
        source="accepted_pilot_publication",
        pilot_checkpoint=manifest["checkpoint"],
        publication_id=manifest["publication_id"],
        pilot_key=manifest["pilot_key"],
        measurement_target_gradient_ratio=report["target_gradient_ratio"],
        measurement_resolved_lambda=report["resolved_lambda"],
        reused_calibration=True,
        target_dataset_fingerprint=target_fingerprint,
        anchor_dataset_fingerprint=anchor_fingerprint,
        cache_fingerprint=cache_fingerprint,
        target_gradient_ratio=ratio,
        resolved_lambda=resolved,
        median_auxiliary_to_target_ratio=median,
        weighted_auxiliary_to_target_ratios=weighted,
        median_weighted_auxiliary_to_target_ratio=weighted_median,
        pilot_stability=stability,
    )
    if statistic == "rms":
        report.update(
            calibration_statistic="rms",
            normalization_value=normalizer,
            rms_auxiliary_to_target_ratio=normalizer,
            rms_weighted_auxiliary_to_target_ratio=weighted_rms,
        )
    config.lvd.resolved_lambda = resolved
    return report


def validate_pilot_calibration_datasets(report: dict[str, Any], target, anchor) -> None:
    """Check a reused report against full loaded datasets, without mutating either.

    Call only for reused publication calibration, before training_anchor_view;
    explicit/legacy fresh-calibration reports do not carry these fingerprints.
    """
    for stream, dataset in (("target", target), ("anchor", anchor)):
        expected = _require_dataset_fingerprint(
            report.get(f"{stream}_dataset_fingerprint"), f"{stream}_dataset_fingerprint"
        )
        if getattr(dataset, "fingerprint", None) != expected:
            raise ConfigError(
                f"{stream} dataset fingerprint does not match accepted pilot calibration; "
                "run generate_pilot with the matching formal source config"
            )


def training_anchor_view(config, anchor):
    if anchor is None or not config.method.name.uses_lvd or config.lvd.calibration_seed is None:
        return anchor
    plan = anchor_plan_for_config(config, anchor)
    return IndexedDatasetView(anchor, plan["training"]["indices"])


def calibration_batches_for_config(config, target, anchor, pad_token_id):
    if config.lvd.calibration_seed is None:
        return None
    if anchor is None:
        raise ConfigError("calibration requires positioned Anchor data")
    rank, world_size = _world()
    plan = anchor_plan_for_config(config, anchor)
    collator = CausalLMCollator(pad_token_id, config.data.pad_to_multiple_of)
    target_indices = sorted(
        range(len(target)),
        key=lambda index: hashlib.sha256(
            f"target-calibration:{config.lvd.calibration_seed}:{target[index]['sample_id']}".encode()
        ).hexdigest(),
    )
    micros = config.lvd.calibration_batches * config.lvd.calibration_accumulation_steps
    if micros * world_size * config.data.target_batch_size > len(target_indices):
        raise ConfigError("Target data is too small for disjoint nonrepeating calibration")
    result = []
    for micro in range(micros):
        slot = micro * world_size + rank
        target_start = slot * config.data.target_batch_size
        anchor_start = slot * config.data.anchor_batch_size
        result.append(
            {
                "target": collator(
                    [
                        target[index]
                        for index in target_indices[
                            target_start : target_start + config.data.target_batch_size
                        ]
                    ]
                ),
                "anchor": collator(
                    [
                        anchor[index]
                        for index in plan["calibration"]["indices"][
                            anchor_start : anchor_start + config.data.anchor_batch_size
                        ]
                    ]
                ),
            }
        )
    return result


class _PilotHealthCallback(TrainerCallback):
    def __init__(self, trainer, health, validation, cache, steps, interval, output):
        self.trainer = trainer
        self.health = health
        self.validation = validation
        self.cache = cache
        self.steps = steps
        self.interval = interval
        self.output = output
        self.baseline = None
        self.validation_baseline = None
        self.history = []
        self.failed = False

    def _measure(self, dataset):
        return evaluate_anchor_health(
            self.trainer.model_wrapped,
            dataset,
            self.cache,
            pad_token_id=self.trainer.pad_token_id,
            device=self.trainer.accelerator.device,
        )

    def on_train_begin(self, args, state, control, **kwargs):
        self.baseline = self._measure(self.health)
        self.validation_baseline = self._measure(self.validation)
        if self.trainer.is_world_process_zero():
            write_json_atomic(
                self.output,
                {"status": "running", "baseline": self.baseline, "history": []},
            )

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.interval == 0 or state.global_step >= self.steps:
            decision = pilot_health_decision(
                self.baseline,
                self._measure(self.health),
                minimum_relative_displacement=(1e-5 if state.global_step >= self.steps else 0.0),
            )
            self.history.append({"step": state.global_step, **decision})
            self.failed = not decision["accepted"]
            if self.trainer.is_world_process_zero():
                write_json_atomic(
                    self.output,
                    {"status": "rejected" if self.failed else "running", "history": self.history},
                )
                print(
                    "PILOT HEALTH",
                    state.global_step,
                    decision["accepted"],
                    decision["candidate"]["cached_base_correct_retention"],
                    decision["reasons"],
                    flush=True,
                )
        if self.failed or state.global_step >= self.steps:
            control.should_training_stop = True
        return control

    def on_epoch_end(self, args, state, control, **kwargs):
        control.should_save = control.should_evaluate = False
        return control


def _publish_pilot_on_rank_zero(trainer, publish: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    """Broadcast the publication outcome after collective model export."""
    payload: list[dict[str, Any] | None] = [None]
    publication_error = None
    if trainer.is_world_process_zero():
        try:
            payload[0] = {"result": publish()}
        except Exception as error:
            publication_error = error
            payload[0] = {"error": f"{type(error).__name__}: {error}"}
    if distributed.is_available() and distributed.is_initialized():
        distributed.broadcast_object_list(payload, src=0)
    outcome = payload[0]
    if outcome is None:
        raise RuntimeError("pilot publication did not receive a rank-zero outcome")
    if "error" in outcome:
        raise RuntimeError(
            f"pilot publication failed on rank 0: {outcome['error']}"
        ) from publication_error
    return outcome["result"]


def generate_pilot(config, *, steps: int, health_interval: int) -> dict[str, Any]:
    """Keep the full schedule, stop early, validate, calibrate, then publish."""
    if (
        type(steps) is not int
        or type(health_interval) is not int
        or min(steps, health_interval) <= 0
    ):
        raise ConfigError("pilot steps and health interval must be positive integers")
    if config.training.resume_from_checkpoint or config.training.max_steps != -1:
        raise ConfigError("generate_pilot requires fresh Base and max_steps=-1 (full schedule)")
    if config.data.steps_per_epoch is not None:
        raise ConfigError("generate_pilot requires the natural full-epoch dataset plan")
    if not config.method.name.uses_lvd or config.data.anchor is None:
        raise ConfigError("generate_pilot requires the formal LVD config and health Anchor")
    destination = (
        Path(config.paths.output_dir)
        if config.paths.output_dir
        else Path(config.paths.artifact_root) / "checkpoints" / config.paths.run_name
    )
    if destination.exists():
        raise FileExistsError(f"pilot output already exists; choose a new run name: {destination}")
    cli = importlib.import_module("sft_conflict.cli.main")
    cli._preflight(config)
    cli._validate_lvd_training_cache_metadata(config)
    bundle = load_model_bundle(
        config.model, config.lora, config.training, cache_dir=cli._cache_paths(config)[0]
    )
    target, anchor, _, _ = cli._load_datasets(config, bundle.tokenizer, include_eval=False)
    plan = anchor_plan_for_config(config, anchor)
    health = IndexedDatasetView(anchor, plan["health"]["indices"])
    validation = IndexedDatasetView(anchor, plan["validation"]["indices"])
    cache = ReferenceCache(config.lvd.d0_cache_path)
    cache.validate_integrity()
    cache.validate_binding(
        model_id=config.model.model_id,
        model_revision=config.model.revision,
        vocabulary_size=int(get_unembedding(bundle.model).weight.shape[0]),
        dataset_fingerprint=anchor.fingerprint,
        position_config=anchor.selector.config.to_dict(),
        normal_epsilon=config.lvd.normal_epsilon,
        filter_mode=config.lvd.d0_filter_mode,
        storage_dtype=config.lvd.d0_storage_dtype,
        minimum_confidence=config.lvd.d0_minimum_confidence,
        minimum_margin=config.lvd.d0_minimum_margin,
    )
    head = get_unembedding(bundle.model)
    head_weight = head.weight.detach().cpu().clone()
    bias = getattr(head, "bias", None)
    head_bias = bias.detach().cpu().clone() if bias is not None else None
    pilot_config = copy.deepcopy(config)
    pilot_config.method.name = MethodName.TARGET_ONLY
    pilot_config.data.anchor = None
    pilot_config.data.anchor_batches_per_target = 0.0
    pilot_config.data.steps_per_epoch = None
    pilot_config.data.validation = None
    pilot_config.reference = ReferenceConfig()
    pilot_config.training.recovery_save_steps = None
    pilot_config.training.save_strategy = "no"
    pilot_config.evaluation.run_after_training = False
    paths = prepare_run(pilot_config, bundle.audit, allow_existing=_world()[1] > 1)
    trainer = cli._build_trainer(
        pilot_config, bundle, target, None, None, paths, with_reference=False
    )
    trainer.args.load_best_model_at_end = False
    monitor = _PilotHealthCallback(
        trainer,
        health,
        validation,
        cache,
        steps,
        health_interval,
        paths.result_dir / "pilot-health.json",
    )
    trainer.add_callback(monitor)
    if trainer.is_world_process_zero():
        dump_config(config, paths.result_dir / "pilot-source-config.yaml")
        dump_config(pilot_config, paths.result_dir / "resolved-config.yaml")
        dump_config(pilot_config, paths.checkpoint_dir / "resolved-config.yaml")
        write_json_atomic(paths.result_dir / "anchor-plan.json", plan)
    result = trainer.train()
    if monitor.failed or not monitor.history or trainer.state.global_step != steps:
        raise RuntimeError("pilot failed the health gate; no accepted export was published")
    decision = monitor.history[-1]
    validation_decision = pilot_health_decision(
        monitor.validation_baseline, monitor._measure(validation)
    )
    decision = {**decision, "validation": validation_decision}
    if trainer.is_world_process_zero():
        write_json_atomic(paths.result_dir / "pilot-health.json", decision)
    if not validation_decision["accepted"]:
        raise RuntimeError("pilot failed independent Anchor validation; no accepted export")
    trainer.lvd_config = copy.deepcopy(config.lvd)
    trainer.lvd_config.resolved_lambda = None
    trainer.reference_cache = cache
    trainer.loss_plugins = build_loss_plugins(
        config.method,
        lvd_config=trainer.lvd_config,
        reference_cache=cache,
        resolved_lvd_lambda=1.0,
        unembedding_weight=head_weight,
        unembedding_bias=head_bias,
    )
    calibration = trainer.calibrate_lvd_lambda(
        paths.result_dir / "lvd-lambda-calibration.json",
        batches=calibration_batches_for_config(config, target, anchor, trainer.pad_token_id),
    )
    if calibration is None:
        raise RuntimeError("pilot calibration did not produce a report")
    stability = calibration_stability_decision(
        calibration, minimum_groups=max(2, min(8, config.lvd.calibration_batches))
    )
    if trainer.is_world_process_zero():
        write_json_atomic(paths.result_dir / "pilot-stability.json", stability)
    if not stability["accepted"]:
        raise RuntimeError(
            f"pilot calibration failed stability checks; no accepted export: {stability['reasons']}"
        )
    final = paths.checkpoint_dir / "final"
    trainer.save_model(str(final))
    report = {
        **decision,
        "pilot_steps": steps,
        "scheduler_steps": trainer.state.max_steps,
        "world_size": _world()[1],
        "anchor_plan_sha256": plan["plan_sha256"],
        "target_dataset_fingerprint": target.fingerprint,
        "calibration": calibration,
        "calibration_stability": stability,
    }

    def publish():
        bundle.tokenizer.save_pretrained(final)
        dump_config(pilot_config, final / "resolved-config.yaml")
        write_json_atomic(final / "pilot-health.json", report)
        registry = write_accepted_pilot_registry(config, final, report)
        write_json_atomic(
            paths.result_dir / "summary.json",
            {
                "status": "accepted",
                "registry": str(registry),
                "report": report,
                "train": result.metrics,
            },
        )
        return {
            "action": "generate-pilot",
            "status": "accepted",
            "global_step": trainer.state.global_step,
            "final_checkpoint": str(final),
            "result_dir": str(paths.result_dir),
            "calibration": calibration,
        }

    return _publish_pilot_on_rank_zero(trainer, publish)
