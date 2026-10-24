"""Post-CE calibration and resumable dynamic-geometry phase state."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as distributed
from torch.distributed.fsdp import FullyShardedDataParallel
from transformers import TrainerCallback

from sft_conflict.data import CausalLMCollator
from sft_conflict.data.anchor_geometry_reference import json_digest
from sft_conflict.losses import LossContext
from sft_conflict.losses.plugins import AnchorCrossEntropy
from sft_conflict.trainers.gradient_accumulation import CPUGradientAccumulator

STATE_FILE = "anchor-geometry-state.json"
GRADIENT_NORM_CHUNK_ELEMENTS = 1 << 20


def _gather(value):
    if not distributed.is_available() or not distributed.is_initialized():
        return [value]
    values = [None] * distributed.get_world_size()
    distributed.all_gather_object(values, value)
    return values


def geometry_recipe(trainer) -> str:
    options = asdict(trainer.anchor_geometry_config)
    if options["objective"] == "centered_v1":
        options = {
            name: value
            for name, value in options.items()
            if name != "objective" and not name.startswith("band_")
        }
    for name in (
        "reference_cache_path",
        "reference_sha256",
        "resolved_lambda",
        "calibrated_at_step",
        "calibration_sha256",
    ):
        options.pop(name)
    return json_digest(
        {
            "method": trainer.method_config.name.value,
            "anchor_weight": trainer.method_config.anchor_weight,
            "options": options,
            "accumulation_steps": trainer.args.gradient_accumulation_steps,
            "world_size": trainer.accelerator.num_processes,
            "training_seed": trainer.args.seed,
            "mixed_precision": trainer.resolved_config.training.mixed_precision,
            "fsdp_use_orig_params": bool(trainer.is_fsdp_enabled),
            "sources": {
                stream: getattr(
                    getattr(trainer.dual_stream_dataset, f"{stream}_dataset"), "fingerprint", None
                )
                for stream in ("target", "anchor")
            },
            "data": asdict(trainer.data_config),
            "model": asdict(trainer.resolved_config.model),
            "training": {
                name: getattr(trainer.args, name)
                for name in (
                    "learning_rate",
                    "weight_decay",
                    "max_steps",
                    "num_train_epochs",
                    "warmup_ratio",
                    "max_grad_norm",
                    "data_seed",
                )
            },
            "optimizer": trainer.args.optim.value,
        }
    )


def geometry_calibration_batches(trainer):
    config = trainer.anchor_geometry_config
    source = trainer.dual_stream_dataset.anchor_dataset
    steps = trainer.args.gradient_accumulation_steps
    world = trainer.accelerator.num_processes
    rank = trainer.accelerator.process_index
    batch_size = trainer.data_config.anchor_batch_size
    required = config.calibration_groups * steps * world * batch_size
    if len(source) < required:
        raise ValueError(
            f"geometry calibration requires {required} distinct Anchor samples; "
            f"received {len(source)}; lower groups for tiny tests (no cycling)"
        )
    ordered = sorted(
        range(len(source)),
        key=lambda index: json_digest(
            [
                "geometry-calibration-v1",
                config.calibration_seed,
                source[index]["sample_id"],
            ]
        ),
    )[:required]
    collator = CausalLMCollator(trainer.pad_token_id, trainer.data_config.pad_to_multiple_of)
    batches, local_ids = [], []
    for micro in range(config.calibration_groups * steps):
        start = (micro * world + rank) * batch_size
        records = [source[index] for index in ordered[start : start + batch_size]]
        batch = collator(records)
        batches.append({"target": batch, "anchor": batch})
        local_ids.append([record["sample_id"] for record in records])
    return batches, {
        "rank": rank,
        "sample_ids": local_ids,
        "anchor_fingerprint": source.fingerprint,
    }


@contextmanager
def _preserve_state(trainer, plugin, model):
    modes = {module: module.training for module in model.modules()}
    gradients = {
        parameter: (
            parameter.grad,
            None if parameter.grad is None else parameter.grad.detach().cpu().clone(),
        )
        for parameter in model.parameters()
    }
    rng = (torch.get_rng_state(), random.getstate(), np.random.get_state())
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    previous = plugin.calibrating
    try:
        plugin.calibrating = True
        model.eval()
        with torch.enable_grad():
            yield
    finally:
        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            for parameter, (original, saved) in gradients.items():
                if original is not None:
                    original.copy_(saved.to(original.device))
                parameter.grad = original
        for module, training in modes.items():
            module.training = training
        plugin.calibrating = previous
        torch.set_rng_state(rng[0])
        random.setstate(rng[1])
        np.random.set_state(rng[2])
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)


def _local_parameter_digest(model) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        value = parameter.detach().cpu().contiguous()
        digest.update(json_digest([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(memoryview(value.reshape(-1).view(torch.uint8).numpy()))
    return digest.hexdigest()


@torch.no_grad()
def _gradient_norm(model, device) -> float:
    squared = torch.zeros((), dtype=torch.float64, device=device)
    for parameter in model.parameters():
        if parameter.grad is not None:
            gradient = parameter.grad.detach().reshape(-1)
            for chunk in gradient.split(GRADIENT_NORM_CHUNK_ELEMENTS):
                squared += chunk.double().square().sum().to(device)
    if isinstance(model, FullyShardedDataParallel):
        distributed.all_reduce(squared)
    return float(squared.sqrt())


def _write_collectively(trainer, path: Path, payload) -> None:
    error = None
    if trainer.is_world_process_zero():
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_text(
                json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            temporary.replace(path)
        except (OSError, TypeError, ValueError) as exc:
            error = str(exc)
    errors = [value for value in _gather(error) if value is not None]
    if errors:
        raise RuntimeError("geometry state publication failed: " + "; ".join(errors))


def calibrate_anchor_geometry(trainer) -> dict:
    config = trainer.anchor_geometry_config
    plugin = trainer._anchor_geometry_plugin()
    if trainer.state.global_step != config.geometry_start_step:
        raise ValueError("geometry calibration must run exactly at the CE warmup boundary")
    if config.resolved_lambda is not None:
        return trainer.anchor_geometry_calibration_report
    batches, plan, error = None, None, None
    try:
        batches, plan = geometry_calibration_batches(trainer)
    except (ValueError, RuntimeError, KeyError) as exc:
        error = str(exc)
    statuses = _gather({"error": error, "plan": plan})
    errors = [value["error"] for value in statuses if value["error"] is not None]
    if errors:
        raise ValueError("geometry calibration batch preflight failed: " + "; ".join(errors))
    model = trainer.model_wrapped
    steps = trainer.args.gradient_accumulation_steps
    weighted_ce = AnchorCrossEntropy(weight=trainer.method_config.anchor_weight)
    fsdp_plugin = getattr(trainer.accelerator.state, "fsdp_plugin", None)
    offload = getattr(getattr(fsdp_plugin, "cpu_offload", None), "offload_params", False)
    accumulator = CPUGradientAccumulator() if offload and steps > 1 else None
    before = _local_parameter_digest(model)
    norms, ratios = [], []
    with _preserve_state(trainer, plugin, model):
        try:
            for group in range(config.calibration_groups):
                measured = []
                for auxiliary in (False, True):
                    model.zero_grad(set_to_none=True)
                    for micro, batch in enumerate(batches[group * steps : (group + 1) * steps]):
                        context = LossContext(
                            model,
                            trainer._prepare_inputs(batch),
                            reference_model=None,
                            capture_anchor_hidden=auxiliary,
                            capture_anchor_label_positions=auxiliary,
                            global_step=trainer.state.global_step,
                            anchor_geometry_plugin=plugin if auxiliary else None,
                            anchor_geometry_epoch=int(trainer.state.epoch or 0),
                        )
                        with trainer.compute_loss_context_manager():
                            result = plugin(context) if auxiliary else weighted_ce(context)
                        trainer.accelerator.backward(result.loss / steps)
                        if accumulator is not None:
                            accumulator.accumulate(model, optimizer_boundary=micro == steps - 1)
                        del context, result
                    measured.append(_gradient_norm(model, trainer.accelerator.device))
                ce_norm, geometry_norm = measured
                if (
                    not math.isfinite(ce_norm)
                    or ce_norm <= 0
                    or not math.isfinite(geometry_norm)
                    or geometry_norm < 0
                ):
                    raise RuntimeError("geometry calibration needs finite positive Anchor CE norms")
                ratio = geometry_norm / ce_norm
                if not math.isfinite(ratio) or (geometry_norm > 0 and ratio == 0):
                    raise RuntimeError("non-finite or underflowed geometry calibration ratio")
                norms.append(measured)
                ratios.append(ratio)
        finally:
            if accumulator is not None:
                accumulator.buffers.clear()
    maximum = max(ratios)
    rms = (
        maximum * math.sqrt(math.fsum((value / maximum) ** 2 for value in ratios) / len(ratios))
        if maximum
        else 0.0
    )
    if not math.isfinite(rms) or rms <= 0:
        raise RuntimeError("geometry calibration has zero effective gradient; cannot normalize")
    if config.objective == "base_band_v2" and rms < config.band_min_calibration_rms:
        raise RuntimeError(
            "band geometry calibration gradient is too small; inspect active-band rates "
            "and width settings instead of amplifying a near-zero gradient"
        )
    resolved = config.geometry_gradient_ratio / rms
    if not math.isfinite(resolved) or resolved <= 0:
        raise RuntimeError("geometry calibration produced an invalid coefficient")
    unchanged = _local_parameter_digest(model) == before
    if not all(_gather(unchanged)):
        raise RuntimeError("calibration unexpectedly changed model parameters")
    report = {
        "version": 1,
        "method": trainer.method_config.name.value,
        "at_step": trainer.state.global_step,
        "statistic": "rms",
        "reference_sha256": trainer.anchor_geometry_reference.sha256,
        "recipe_sha256": geometry_recipe(trainer),
        "warmup_parameter_shards": _gather(before),
        "plan": [value["plan"] for value in statuses],
        "microbatches_per_group": steps,
        "group_count": len(ratios),
        "weighted_anchor_ce_gradient_norms": [row[0] for row in norms],
        "geometry_gradient_norms": [row[1] for row in norms],
        "geometry_to_weighted_anchor_ce_ratios": ratios,
        "rms_ratio": rms,
        "gradient_budget": config.geometry_gradient_ratio,
        "resolved_lambda": resolved,
        "training_rows_removed": 0,
    }
    digest = json_digest(report)
    _write_collectively(
        trainer, Path(trainer.args.output_dir) / "anchor-geometry-calibration.json", report
    )
    config.resolved_lambda = resolved
    config.calibrated_at_step = trainer.state.global_step
    config.calibration_sha256 = digest
    trainer.anchor_geometry_calibration_report = report
    if trainer.is_world_process_zero():
        print(
            f"Dynamic geometry calibrated at step {trainer.state.global_step}: "
            f"lambda={resolved:.12g}, weighted-Anchor gradient budget="
            f"{config.geometry_gradient_ratio:.6g}",
            flush=True,
        )
    return report


def validate_calibration_report(trainer, report: dict) -> None:
    config = trainer.anchor_geometry_config
    if (
        report.get("version") != 1
        or report.get("method") != trainer.method_config.name.value
        or report.get("statistic") != "rms"
        or report.get("gradient_budget") != config.geometry_gradient_ratio
        or report.get("group_count") != config.calibration_groups
        or report.get("microbatches_per_group") != trainer.args.gradient_accumulation_steps
        or report.get("training_rows_removed") != 0
    ):
        raise ValueError("geometry calibration report protocol mismatch")
    arrays = [
        report.get(name)
        for name in (
            "weighted_anchor_ce_gradient_norms",
            "geometry_gradient_norms",
            "geometry_to_weighted_anchor_ce_ratios",
        )
    ]
    if any(
        not isinstance(values, list) or len(values) != config.calibration_groups
        for values in arrays
    ):
        raise ValueError("geometry calibration report norm/ratio arrays mismatch")
    for index, values in enumerate(arrays):
        if any(
            not isinstance(value, int | float)
            or isinstance(value, bool)
            or not math.isfinite(value)
            or (value <= 0 if index == 0 else value < 0)
            for value in values
        ):
            raise ValueError("geometry calibration report invalid gradient norm/ratio")
    ce_norms, geometry_norms, ratios = arrays
    for ce_norm, geometry_norm, ratio in zip(ce_norms, geometry_norms, ratios, strict=True):
        if not math.isclose(ratio, geometry_norm / ce_norm, rel_tol=1e-10, abs_tol=0):
            raise ValueError("geometry calibration report ratios do not match gradient norms")
    peak = max(ratios)
    rms = (
        peak * math.sqrt(math.fsum((ratio / peak) ** 2 for ratio in ratios) / len(ratios))
        if peak
        else 0.0
    )
    if not math.isfinite(rms) or rms <= 0:
        raise ValueError("geometry calibration report has zero effective RMS")
    if config.objective == "base_band_v2" and rms < config.band_min_calibration_rms:
        raise ValueError("band geometry calibration report has a near-zero effective RMS")
    for name, expected in (
        ("rms_ratio", rms),
        ("resolved_lambda", config.geometry_gradient_ratio / rms),
    ):
        actual = report.get(name)
        if (
            not isinstance(actual, int | float)
            or isinstance(actual, bool)
            or not math.isfinite(actual)
            or not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=0)
        ):
            raise ValueError(f"geometry calibration report inconsistent {name}")
    plan = report.get("plan")
    if not isinstance(plan, list) or len(plan) != trainer.accelerator.num_processes:
        raise ValueError("geometry calibration report rank plan mismatch")
    expected_micros = config.calibration_groups * trainer.args.gradient_accumulation_steps
    sample_ids = []
    for rank, entry in enumerate(plan):
        if (
            not isinstance(entry, dict)
            or entry.get("rank") != rank
            or entry.get("anchor_fingerprint")
            != trainer.dual_stream_dataset.anchor_dataset.fingerprint
            or len(entry.get("sample_ids", [])) != expected_micros
        ):
            raise ValueError("geometry calibration report sample plan mismatch")
        for batch in entry["sample_ids"]:
            if (
                not isinstance(batch, list)
                or len(batch) != trainer.data_config.anchor_batch_size
                or any(not isinstance(value, str) or not value for value in batch)
            ):
                raise ValueError("geometry calibration report batch layout mismatch")
            sample_ids.extend(batch)
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("geometry calibration report repeats training samples")


def save_geometry_state(trainer, checkpoint_dir: Path) -> None:
    config = trainer.anchor_geometry_config
    _write_collectively(
        trainer,
        checkpoint_dir / STATE_FILE,
        {
            "version": 1,
            "method": trainer.method_config.name.value,
            "global_step": trainer.state.global_step,
            "reference_sha256": trainer.anchor_geometry_reference.sha256,
            "recipe_sha256": geometry_recipe(trainer),
            "resolved_lambda": config.resolved_lambda,
            "calibrated_at_step": config.calibrated_at_step,
            "calibration_sha256": config.calibration_sha256,
            "calibration_report": trainer.anchor_geometry_calibration_report,
        },
    )


def restore_geometry_state(trainer, checkpoint_dir: str | Path) -> None:
    state = None
    error = None
    try:
        path = Path(checkpoint_dir) / STATE_FILE
        if not path.is_file():
            raise ValueError(
                "dynamic geometry cannot resume a legacy checkpoint without phase state"
            )
        state = json.loads(path.read_text(encoding="utf-8"))
        if (
            state.get("version") != 1
            or state.get("method") != trainer.method_config.name.value
            or state.get("reference_sha256") != trainer.anchor_geometry_reference.sha256
            or state.get("recipe_sha256") != geometry_recipe(trainer)
        ):
            raise ValueError("geometry resume method/reference/recipe mismatch")
        saved_step = json.loads((Path(checkpoint_dir) / "trainer_state.json").read_text())[
            "global_step"
        ]
        if state["global_step"] != saved_step:
            raise ValueError("geometry phase state does not match the checkpoint step")
        config = copy.deepcopy(trainer.anchor_geometry_config)
        for name in ("resolved_lambda", "calibrated_at_step", "calibration_sha256"):
            setattr(config, name, state[name])
        config.validate()
        report = state["calibration_report"]
        if config.resolved_lambda is None:
            if saved_step > config.geometry_start_step or report is not None:
                raise ValueError("checkpoint is missing the completed geometry calibration")
        elif (
            not isinstance(report, dict)
            or json_digest(report) != config.calibration_sha256
            or report.get("resolved_lambda") != config.resolved_lambda
            or report.get("at_step") != config.calibrated_at_step
            or report.get("recipe_sha256") != geometry_recipe(trainer)
            or report.get("reference_sha256") != trainer.anchor_geometry_reference.sha256
        ):
            raise ValueError("geometry calibration report identity mismatch")
        if report is not None:
            validate_calibration_report(trainer, report)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        error = str(exc)
    errors = [value for value in _gather(error) if value is not None]
    if errors:
        raise ValueError("geometry resume rejected: " + "; ".join(errors))
    for name in ("resolved_lambda", "calibrated_at_step", "calibration_sha256"):
        setattr(trainer.anchor_geometry_config, name, state[name])
    trainer.anchor_geometry_calibration_report = state["calibration_report"]


class GeometryCalibrationCallback(TrainerCallback):
    def __init__(self, trainer) -> None:
        self.trainer = trainer

    def on_step_begin(self, args, state, control, **kwargs):
        config = self.trainer.anchor_geometry_config
        if state.global_step >= config.geometry_start_step and config.resolved_lambda is None:
            calibrate_anchor_geometry(self.trainer)
        return control
