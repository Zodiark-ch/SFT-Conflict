"""One standard Trainer lifecycle with target and anchor data streams."""

from __future__ import annotations

import contextlib
import gc
import hashlib
import json
import math
import os
import shutil
import statistics
import warnings
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as distributed
from accelerate.data_loader import BatchSamplerShard, DataLoaderShard
from accelerate.utils.constants import FSDP_MODEL_NAME
from torch import Tensor, nn
from torch.distributed.fsdp import FullyShardedDataParallel, StateDictType
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from transformers import Trainer
from transformers.modeling_utils import load_sharded_checkpoint
from transformers.trainer import SCHEDULER_NAME, TRAINING_ARGS_NAME
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, get_last_checkpoint
from transformers.utils import (
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
)

from sft_conflict.callbacks import ParameterInvariantCallback
from sft_conflict.callbacks.recovery import RecoveryCheckpointCallback
from sft_conflict.config import (
    DataConfig,
    ExperimentConfig,
    LVDConfig,
    LVDPlusAnchorMode,
    LVDPlusConfig,
    LVDResponseMode,
    MethodConfig,
    dump_config,
)
from sft_conflict.data import (
    CausalLMCollator,
    DualStreamCollator,
    DualStreamDataset,
    ReferenceCache,
    TargetStreamCollator,
)
from sft_conflict.data.dual_stream import EpochAwareSequentialSampler
from sft_conflict.evaluation.native import sharded_classification_counts
from sft_conflict.losses import (
    AnchorLVDLoss,
    CountAwareMetricAccumulator,
    LossContext,
    LossResult,
    TargetCrossEntropy,
    build_loss_plugins,
)
from sft_conflict.losses.plugins import (
    AnchorCrossEntropy,
    AnchorKLDistillation,
    NullLVDLoss,
    lvd_calibration_metric_counts,
)
from sft_conflict.models import get_unembedding
from sft_conflict.pilot_stability import calibration_normalizer
from sft_conflict.tracking.tensor_preview import (
    build_random_tensor_preview,
    render_tensor_preview,
)
from sft_conflict.trainers.gradient_accumulation import CPUGradientAccumulator
from sft_conflict.trainers.lvd_plus_calibration import (
    calibrate_lvd_plus_lambda as _calibrate_lvd_plus_lambda,
)
from sft_conflict.trainers.lvd_plus_calibration import reuse_lvd_plus_prepared_model
from sft_conflict.trainers.native_selection import NativeSelection

if TYPE_CHECKING:
    from sft_conflict.losses.lvd_plus import AnchorLVDPlusLoss

RANK_LOCAL_FSDP_MANIFEST = "optimizer-rank-local.json"


def _gather_calibration_objects(value: Any) -> list[Any]:
    if not distributed.is_available() or not distributed.is_initialized():
        return [value]
    values = [None] * distributed.get_world_size()
    distributed.all_gather_object(values, value)
    return values


def _calibration_cpu_batch(batch: dict[str, Any], *, require_cpu: bool) -> dict[str, Any]:
    """Snapshot data, not graphs; replay one microbatch at a time on the device."""
    if not isinstance(batch, dict) or not isinstance(batch.get("target"), dict):
        raise ValueError("calibration batches must contain a collated target dictionary")
    if "anchor" not in batch or not isinstance(batch["anchor"], dict | type(None)):
        raise ValueError("calibration batches must contain a collated anchor dictionary or None")

    def snapshot(value: Any) -> Any:
        if isinstance(value, Tensor):
            if require_cpu and value.device.type != "cpu":
                raise ValueError("explicit calibration batch tensors must be on CPU")
            return value.detach().to("cpu", copy=True)
        if isinstance(value, dict):
            return {key: snapshot(child) for key, child in value.items()}
        if isinstance(value, list | tuple):
            return type(value)(snapshot(child) for child in value)
        return value

    return snapshot(batch)


def _explicit_calibration_batches(
    batches: Iterable[dict[str, Any]], *, groups: int, steps: int
) -> list[dict[str, Any]]:
    """Bounded CPU preflight: a short rank must fail before any collective backward."""
    count = groups * steps
    buffered: list[dict[str, Any]] = []
    error = None
    try:
        iterator = iter(batches)
        for _ in range(count):
            try:
                batch = next(iterator)
            except StopIteration as exc:
                raise RuntimeError(
                    f"insufficient explicit calibration batches: required {count}, "
                    f"received {len(buffered)} (no cycling)"
                ) from exc
            buffered.append(_calibration_cpu_batch(batch, require_cpu=True))
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    statuses = _gather_calibration_objects({"groups": groups, "steps": steps, "error": error})
    errors = [
        f"rank {rank}: {status['error']}" for rank, status in enumerate(statuses) if status["error"]
    ]
    if len({(status["groups"], status["steps"]) for status in statuses}) != 1:
        errors.append("ranks requested different calibration group layouts")
    if errors:
        raise RuntimeError("LVD calibration batch preflight failed; " + "; ".join(errors))
    return buffered


def _calibration_ratio_statistics(
    ratios: list[float], *, allow_zeros: bool = False
) -> dict[str, Any]:
    """Describe all raw ratios; only opt-in log statistics exclude legitimate zeros."""
    if allow_zeros and any(not math.isfinite(ratio) or ratio < 0 for ratio in ratios):
        raise ValueError("calibration ratios must be finite and non-negative")

    def quantiles(values: list[float]) -> dict[str, float]:
        ordered = sorted(values)
        result = {}
        for name, probability in (
            ("min", 0.0),
            ("p05", 0.05),
            ("p25", 0.25),
            ("median", 0.5),
            ("p75", 0.75),
            ("p95", 0.95),
            ("max", 1.0),
        ):
            index = (len(ordered) - 1) * probability
            lower, upper = math.floor(index), math.ceil(index)
            weight = index - lower
            result[name] = ordered[lower] + weight * (ordered[upper] - ordered[lower])
        return result

    logs = [math.log10(ratio) for ratio in ratios if not allow_zeros or ratio > 0]
    log_quantiles = quantiles(logs) if logs else None
    split = len(ratios) // 2
    first = quantiles(ratios[:split])["median"] if split else None
    second = quantiles(ratios[split:])["median"]
    result = {
        "quantile_method": "linear interpolation at (n - 1) * probability",
        "quantiles": quantiles(ratios),
        "log10_quantiles": log_quantiles,
        "log10_spread": max(logs) - min(logs) if logs else None,
        "log10_iqr": log_quantiles["p75"] - log_quantiles["p25"] if log_quantiles else None,
        "half_split": {
            "first_group_count": split,
            "second_group_count": len(ratios) - split,
            "first_median": first,
            "second_median": second,
            "relative_median_gap": (
                abs(second - first) / max(first, second)
                if first is not None and max(first, second) > 0
                else None
            ),
            "second_to_first_median_log10_ratio": (
                math.log10(second) - math.log10(first)
                if first is not None and first > 0 and second > 0
                else None
            ),
        },
    }
    if allow_zeros:
        result.update(
            zero_group_count=len(ratios) - len(logs),
            positive_group_count=len(logs),
            log10_notes=(
                "Log10 statistics use strictly positive raw group ratios only. Zero ratios "
                "remain in raw quantiles and ordered half-split medians, without epsilon. "
                "Empty positive subsets and undefined half-median log ratios are null. "
                "Descriptive only, not a pilot acceptance test."
            ),
        )
    return result


def _rank_local_optimizer_name(rank: int, world_size: int) -> str:
    return f"optimizer.rank-{rank:05d}-of-{world_size:05d}.pt"


def _set_fsdp_state_dict_type(plugin: Any, state_dict_type: StateDictType) -> None:
    plugin.state_dict_type = state_dict_type
    plugin.state_dict_config = None
    plugin.optim_state_dict_config = None
    plugin.set_state_dict_type()


@contextlib.contextmanager
def _temporary_fsdp_state_dict_type(plugin: Any, state_dict_type: StateDictType):
    previous = (
        plugin.state_dict_type,
        plugin.state_dict_config,
        plugin.optim_state_dict_config,
    )
    _set_fsdp_state_dict_type(plugin, state_dict_type)
    try:
        yield
    finally:
        (
            plugin.state_dict_type,
            plugin.state_dict_config,
            plugin.optim_state_dict_config,
        ) = previous


class SFTConflictTrainer(Trainer):
    def __init__(
        self,
        *args: Any,
        target_dataset: Dataset,
        anchor_dataset: Dataset | None,
        data_config: DataConfig,
        method_config: MethodConfig,
        pad_token_id: int,
        preview_tokenizer: Any | None = None,
        reference_model: nn.Module | None = None,
        resolved_config: ExperimentConfig | None = None,
        anchor_composition_dir: str | Path | None = None,
        fsdp_sync_each_batch: bool = True,
        lvd_config: LVDConfig | None = None,
        reference_cache: ReferenceCache | None = None,
        resolved_lvd_lambda: float | None = None,
        lvd_plus_config: LVDPlusConfig | None = None,
        resolved_lvd_plus_lambda: float | None = None,
        native_selection: NativeSelection | None = None,
        **kwargs: Any,
    ) -> None:
        if method_config.name.uses_lvd_plus:
            if reference_model is not None or reference_cache is not None:
                raise ValueError("LVD+ does not accept an online reference model or cache")
            if lvd_plus_config is None:
                lvd_plus_config = (
                    resolved_config.lvd_plus if resolved_config is not None else LVDPlusConfig()
                )
            if resolved_lvd_plus_lambda is not None:
                lvd_plus_config.resolved_lambda = resolved_lvd_plus_lambda
            lvd_plus_config.validate()
            if lvd_plus_config.anchor_mode is LVDPlusAnchorMode.RECORDED_S0:
                if resolved_config is None:
                    raise ValueError("recorded_s0 requires resolved_config for S0 cache binding")
                if (
                    resolved_config.method.name is not method_config.name
                    or resolved_config.lvd_plus.anchor_mode is not LVDPlusAnchorMode.RECORDED_S0
                ):
                    raise ValueError(
                        "recorded_s0 requires matching resolved_config method/anchor_mode"
                    )
                resolved_config.lvd_plus.validate()
                resolved_config.s0.validate()
                path = lvd_plus_config.s0_cache_path
                resolved_path = resolved_config.lvd_plus.s0_cache_path
                if path is None or resolved_path is None:
                    raise ValueError("recorded_s0 requires a resolved lvd_plus.s0_cache_path")
                if (
                    Path(path).expanduser().absolute()
                    != Path(resolved_path).expanduser().absolute()
                ):
                    raise ValueError("recorded_s0 s0_cache_path disagrees with resolved_config")
            if resolved_lvd_plus_lambda is None:
                resolved_lvd_plus_lambda = lvd_plus_config.resolved_lambda
                if (
                    resolved_lvd_plus_lambda is None
                    and lvd_plus_config.lambda_mode.value == "fixed"
                ):
                    resolved_lvd_plus_lambda = lvd_plus_config.fixed_lambda
        token_collator = CausalLMCollator(
            pad_token_id=pad_token_id,
            pad_to_multiple_of=data_config.pad_to_multiple_of,
        )
        kwargs["train_dataset"] = target_dataset
        kwargs.setdefault("data_collator", TargetStreamCollator(token_collator))
        super().__init__(*args, **kwargs)
        # Custom compute_loss already returns per-microbatch means and ignores
        # num_items_in_batch, even when model.forward accepts **kwargs. Let HF
        # normalize fallback training steps for gradient accumulation.
        self.model_accepts_loss_kwargs = False
        self.pad_token_id = int(pad_token_id)
        if (
            method_config.name.uses_dynamic_geometry
            and self.is_fsdp_enabled
            and not self.accelerator.state.fsdp_plugin.use_orig_params
        ):
            raise ValueError("dynamic geometry FSDP requires use_orig_params=True")
        if self.is_fsdp_enabled:
            _set_fsdp_state_dict_type(
                self.accelerator.state.fsdp_plugin,
                StateDictType.SHARDED_STATE_DICT,
            )
        self.fsdp_sync_each_batch = bool(
            fsdp_sync_each_batch
            and self.is_fsdp_enabled
            and self.args.gradient_accumulation_steps > 1
        )
        if self.fsdp_sync_each_batch:
            self.accelerator.gradient_state.plugin_kwargs["sync_each_batch"] = True
            # Transformers 4.48 calls Accelerator.no_sync directly, bypassing
            # Accelerator.accumulate and its sync_each_batch policy.
            original_no_sync = self.accelerator.no_sync

            def no_sync_unless_fsdp(model):
                if isinstance(model, FullyShardedDataParallel):
                    return contextlib.nullcontext()
                return original_no_sync(model)

            self.accelerator.no_sync = no_sync_unless_fsdp
        self._cpu_gradient_accumulator = (
            CPUGradientAccumulator()
            if self.fsdp_sync_each_batch
            and self.accelerator.state.fsdp_plugin.cpu_offload.offload_params
            else None
        )
        self.data_config = data_config
        self.preview_tokenizer = preview_tokenizer
        self.native_selection = native_selection
        self._evaluation_tensor_preview_printed = False
        self.method_config = method_config
        self.resolved_config = resolved_config
        self.reference_model = reference_model
        self.lvd_config = lvd_config
        self.reference_cache = reference_cache
        self.lvd_plus_config = lvd_plus_config
        self.resolved_lvd_plus_lambda = resolved_lvd_plus_lambda
        self.anchor_geometry_config = None
        self.anchor_geometry_reference = None
        self.anchor_geometry_calibration_report = None
        self.capture_anchor_label_positions = bool(
            method_config.name.uses_lvd_plus or method_config.name.uses_dynamic_geometry
        )
        self.anchor_composition_path = (
            Path(anchor_composition_dir) / f"anchor-batches.rank-{self.args.process_index}.jsonl"
            if anchor_composition_dir is not None
            else None
        )
        self._anchor_micro_batch = 0
        if self.reference_model is not None:
            self.reference_model.requires_grad_(False)
            self.reference_model.eval()
            self._move_model_to_device(self.reference_model, self.args.device)

        self.dual_stream_dataset = DualStreamDataset(
            target_dataset,
            anchor_dataset,
            target_batch_size=data_config.target_batch_size,
            anchor_batch_size=data_config.anchor_batch_size,
            anchor_batches_per_target=data_config.anchor_batches_per_target,
            world_size=max(1, self.args.world_size),
            seed=data_config.seed,
            steps_per_epoch=data_config.steps_per_epoch,
            anchor_sampling_strategy=data_config.anchor_sampling_strategy,
        )
        self.dual_stream_collator = DualStreamCollator(token_collator)
        lvd_unembedding_weight = None
        lvd_unembedding_bias = None
        if method_config.name.uses_lvd_geometry:
            output_head = get_unembedding(self.model)
            if any(parameter.requires_grad for parameter in output_head.parameters()):
                raise ValueError("LVD geometry requires a frozen Unembedding")
            lvd_unembedding_weight = output_head.weight.detach().cpu().clone()
            output_bias = getattr(output_head, "bias", None)
            if output_bias is not None:
                lvd_unembedding_bias = output_bias.detach().cpu().clone()
        plugin_kwargs = {
            "lvd_config": lvd_config,
            "reference_cache": reference_cache,
            "resolved_lvd_lambda": resolved_lvd_lambda,
            "unembedding_weight": lvd_unembedding_weight,
            "unembedding_bias": lvd_unembedding_bias,
        }
        if method_config.name.uses_dynamic_geometry:
            from sft_conflict.data.anchor_geometry_reference import (
                AnchorGeometryReference,
                model_digest,
                reference_binding,
            )

            if resolved_config is None or preview_tokenizer is None or anchor_dataset is None:
                raise ValueError("dynamic geometry requires resolved config, tokenizer and Anchor")
            resolved_config.validate()
            geometry = resolved_config.anchor_geometry
            cache = AnchorGeometryReference(
                geometry.reference_cache_path,
                expected_sha256=geometry.reference_sha256,
            )
            cache.validate(
                reference_binding(resolved_config, anchor_dataset, preview_tokenizer),
                initial_model_sha256=model_digest(self.model),
            )
            cache.validate_dataset(anchor_dataset)
            geometry.reference_sha256 = cache.sha256
            self.anchor_geometry_config = geometry
            self.anchor_geometry_reference = cache
            plugin_kwargs.update(anchor_geometry_config=geometry, anchor_geometry_reference=cache)
        # Legacy callers and factory test doubles keep their original signature.
        if method_config.name.uses_lvd_plus:
            plugin_kwargs.update(
                lvd_plus_config=lvd_plus_config,
                resolved_lvd_plus_lambda=resolved_lvd_plus_lambda,
            )
            if lvd_plus_config.anchor_mode is LVDPlusAnchorMode.RECORDED_S0:
                from sft_conflict.data.s0_cache import S0Cache

                cache = S0Cache(lvd_plus_config.s0_cache_path)
                # Bind the actual snapshot's FP32 geometry values. Widening a
                # BF16 head cannot recover pre-rounding FP32 weights: a cache
                # recorded from different values must fail its head SHA check.
                # Keep the original snapshot dtype and legacy fingerprint intact.
                cache.validate_binding(
                    weight=lvd_unembedding_weight.float(),
                    bias=lvd_unembedding_bias.float() if lvd_unembedding_bias is not None else None,
                    model_id=resolved_config.model.model_id,
                    model_revision=resolved_config.model.revision,
                    solver_config=asdict(resolved_config.s0),
                )
                identity = cache.identity()
                manifest_sha256 = identity["manifest_sha256"]
                for config in (lvd_plus_config, resolved_config.lvd_plus):
                    if (
                        config.s0_cache_sha256 is not None
                        and config.s0_cache_sha256 != manifest_sha256
                    ):
                        raise ValueError("S0 cache does not match lvd_plus.s0_cache_sha256")
                # Obtain one defensive CPU table from the cache, never per loss.
                centers = cache.sites
                for config in (lvd_plus_config, resolved_config.lvd_plus):
                    config.s0_cache_sha256 = manifest_sha256
                plugin_kwargs.update(
                    lvd_plus_anchor_centers=centers,
                    lvd_plus_s0_identity=identity,
                )
        self.loss_plugins = build_loss_plugins(method_config, **plugin_kwargs)
        self.capture_anchor_hidden = any(
            plugin.requires_anchor_hidden for plugin in self.loss_plugins
        )
        self._metric_accumulators = {
            phase: CountAwareMetricAccumulator() for phase in ("train", "eval")
        }
        self._in_prediction_step = False
        self._metric_totals: defaultdict[str, float] = self._metric_accumulators["train"].totals
        self._metric_observations: defaultdict[str, int] = self._metric_accumulators[
            "train"
        ].observations
        self.parameter_invariant_callback = ParameterInvariantCallback(self.model)
        self.add_callback(self.parameter_invariant_callback)
        if resolved_config is not None and resolved_config.training.recovery_save_steps is not None:
            self.add_callback(
                RecoveryCheckpointCallback(resolved_config.training.recovery_save_steps)
            )
        if method_config.name.uses_dynamic_geometry:
            from sft_conflict.trainers.anchor_geometry import GeometryCalibrationCallback

            self.add_callback(GeometryCalibrationCallback(self))

    def _anchor_geometry_plugin(self):
        if not self.method_config.name.uses_dynamic_geometry:
            return None
        from sft_conflict.losses.anchor_geometry import AnchorGeometryLoss

        return next(
            plugin for plugin in self.loss_plugins if isinstance(plugin, AnchorGeometryLoss)
        )

    def _lvd_plus_plugin(self) -> AnchorLVDPlusLoss | None:
        if not self.method_config.name.uses_lvd_plus:
            return None
        from sft_conflict.losses.lvd_plus import AnchorLVDPlusLoss

        return next(
            (plugin for plugin in self.loss_plugins if isinstance(plugin, AnchorLVDPlusLoss)),
            None,
        )

    def calibrate_lvd_plus_lambda(
        self,
        output_path: str | Path | None = None,
        *,
        batches: Iterable[dict[str, Any]] | None = None,
    ) -> dict[str, Any] | None:
        """Independent, state-preserving Base RMS calibration; no legacy pilot."""
        if self.lvd_plus_config is not None and self.lvd_plus_config.uses_boundary_risk:
            if batches is not None:
                raise ValueError("hybrid records require their bound full-training-source plan")
            from sft_conflict.trainers.lvd_plus_recording import calibrate_lvd_plus_record

            return calibrate_lvd_plus_record(self, output_path)
        return _calibrate_lvd_plus_lambda(self, output_path, batches=batches)

    def record_lvd_plus_scales(self, output_path: str | Path | None = None) -> dict[str, Any]:
        """Record immutable Base loss scales without optimizer steps or gradients."""
        from sft_conflict.trainers.lvd_plus_recording import record_lvd_plus_scales

        return record_lvd_plus_scales(self, output_path)

    def _lvd_plugin(self) -> AnchorLVDLoss | None:
        return next(
            (plugin for plugin in self.loss_plugins if isinstance(plugin, AnchorLVDLoss)),
            None,
        )

    @staticmethod
    def _distributed_gradient_norm(
        gradients: tuple[Tensor | None, ...],
        reduction_device: torch.device,
    ) -> float:
        squared = torch.zeros((), dtype=torch.float64, device=reduction_device)
        for gradient in gradients:
            if gradient is not None:
                local_squared = gradient.detach().double().square().sum()
                squared = squared + local_squared.to(reduction_device)
        if distributed.is_available() and distributed.is_initialized():
            distributed.all_reduce(squared, op=distributed.ReduceOp.SUM)
        return float(squared.sqrt().item())

    def calibrate_lvd_lambda(
        self,
        output_path: str | Path | None = None,
        *,
        batches: Iterable[dict[str, Any]] | None = None,
    ) -> dict[str, Any] | None:
        """Measure gradients of mean microbatch losses, never pooled LVD pressure.

        Explicit batches are rank-local, finite, collated CPU dictionaries. Only
        the requested prefix is consumed; unlike the default loader it never
        cycles. Fixed/resolved lambda returns before touching batches or models.
        """
        plugin = self._lvd_plugin()
        if plugin is None:
            return None
        if self.lvd_config is None:
            raise RuntimeError("LVD plugin is missing its configuration")
        if self.lvd_config.resolved_lambda is not None:
            plugin.set_lambda(self.lvd_config.resolved_lambda)
            return {
                "mode": self.lvd_config.lambda_mode.value,
                "resolved_lambda": self.lvd_config.resolved_lambda,
                "source": "resolved_config",
            }
        if self.lvd_config.lambda_mode.value == "fixed":
            if self.lvd_config.fixed_lambda is None:
                raise RuntimeError("fixed LVD lambda is missing")
            resolved = float(self.lvd_config.fixed_lambda)
            plugin.set_lambda(resolved)
            self.lvd_config.resolved_lambda = resolved
            return {"mode": "fixed", "resolved_lambda": resolved, "source": "fixed_lambda"}

        response_mode = LVDResponseMode(self.lvd_config.response_mode)
        allow_zero_auxiliary = response_mode is LVDResponseMode.THREE_SEGMENT
        statistic = self.lvd_config.calibration_statistic
        if not isinstance(statistic, str) or statistic not in {"median", "rms"}:
            raise ValueError("lvd.calibration_statistic must be 'median' or 'rms'")
        if statistic == "rms" and not allow_zero_auxiliary:
            raise ValueError(
                "lvd.calibration_statistic='rms' requires response_mode='three_segment'"
            )
        metric_counts = lvd_calibration_metric_counts(response_mode)
        steps = self.lvd_config.calibration_accumulation_steps
        if type(steps) is not int or steps <= 0:
            raise ValueError("lvd.calibration_accumulation_steps must be a positive integer")
        grouped_report = steps != 1 or batches is not None
        explicit_batches = (
            _explicit_calibration_batches(
                batches, groups=self.lvd_config.calibration_batches, steps=steps
            )
            if batches is not None
            else None
        )
        # prepare_model is not idempotent for DDP/native AMP in pinned Accelerate.
        # Trainer may retain the bare module in self.model after preparing its wrapper.
        registered_models = getattr(self.accelerator, "_models", ())
        for candidate in (getattr(self, "model_wrapped", self.model), self.model):
            if (
                isinstance(
                    candidate, FullyShardedDataParallel | nn.parallel.DistributedDataParallel
                )
                or getattr(candidate, "_is_accelerate_prepared", False)
                or any(candidate is prepared for prepared in registered_models)
            ):
                prepared_model = candidate
                break
        else:
            prepared_model = self.accelerator.prepare_model(self.model)
        self.model = self.model_wrapped = prepared_model
        parameters = tuple(
            parameter for parameter in self.model.parameters() if parameter.requires_grad
        )
        if not parameters:
            raise RuntimeError("LVD lambda calibration found no trainable parameters")
        loader = self.get_train_dataloader() if explicit_batches is None else explicit_batches
        iterator = iter(loader)
        loader_restarts = 0
        microbatch_count = 0
        local_plan: list[dict[str, Any]] = []
        ratios: list[float] = []
        target_norms: list[float] = []
        auxiliary_norms: list[float] = []
        batch_diagnostics: list[dict[str, Any]] = []
        batch_metrics = CountAwareMetricAccumulator()
        pooled_metrics = CountAwareMetricAccumulator()
        fsdp_plugin = getattr(getattr(self.accelerator, "state", None), "fsdp_plugin", None)
        cpu_offload = getattr(fsdp_plugin, "cpu_offload", None)
        # A calibration group may be >1 even when training accumulation is 1.
        # Use a private accumulator, never the training window's live buffers.
        cpu_accumulator = (
            CPUGradientAccumulator()
            if steps > 1
            and (
                getattr(cpu_offload, "offload_params", False)
                or (
                    isinstance(self.model, FullyShardedDataParallel)
                    and self.model.cpu_offload.offload_params
                )
            )
            else None
        )

        def summarize_diagnostics(values: dict[str, float]) -> dict[str, Any]:
            measurements: dict[str, float | None] = {}
            for name, (numerator_name, denominator_name) in metric_counts.items():
                numerator = values.get(numerator_name)
                denominator = values.get(denominator_name)
                # Fake/older plugins can return LossResult with default empty
                # counts. Missing details and unobserved ratios are null, not 1.
                measurements[name] = (
                    values.get(name)
                    if numerator is not None and denominator is not None and denominator > 0
                    else None
                )
            return {
                "metrics": measurements,
                "counts": {
                    name: values.get(name) for pair in metric_counts.values() for name in pair
                },
            }

        was_training = self.model.training
        previous_lambda = plugin.lambda_value
        plugin.set_lambda(1.0)
        self.model.eval()
        try:
            for group_index in range(self.lvd_config.calibration_batches):
                group = []
                for micro_index in range(steps):
                    try:
                        batch = next(iterator)
                    except StopIteration:
                        if explicit_batches is not None:
                            raise RuntimeError(
                                "explicit calibration batches cannot cycle"
                            ) from None
                        loader_restarts += 1
                        iterator = iter(loader)
                        try:
                            batch = next(iterator)
                        except StopIteration:
                            raise RuntimeError("LVD calibration training loader is empty") from None
                    if steps > 1 and explicit_batches is None:
                        batch = _calibration_cpu_batch(batch, require_cpu=False)
                    group.append(batch)
                    if grouped_report:
                        local_plan.append(
                            {
                                "group_index": group_index,
                                "microbatch_index": microbatch_count,
                                "microbatch_in_group": micro_index,
                                "loader_pass": loader_restarts,
                                **{
                                    f"{stream}_sample_ids": (
                                        list(batch[stream]["sample_ids"])
                                        if batch.get(stream) is not None
                                        and batch[stream].get("sample_ids") is not None
                                        else None
                                        if batch.get(stream) is not None
                                        else []
                                    )
                                    for stream in ("target", "anchor")
                                },
                            }
                        )
                    microbatch_count += 1

                self.model.zero_grad(set_to_none=True)
                for micro_index, batch in enumerate(group):
                    target_context = LossContext(
                        self.model,
                        self._prepare_inputs(batch),
                        reference_model=self.reference_model,
                        capture_anchor_hidden=False,
                    )
                    target_loss = TargetCrossEntropy()(target_context).loss
                    self.accelerator.backward(target_loss / steps if steps > 1 else target_loss)
                    if cpu_accumulator is not None:
                        cpu_accumulator.accumulate(
                            self.model, optimizer_boundary=micro_index == steps - 1
                        )
                    del target_context, target_loss
                target_gradients = tuple(parameter.grad for parameter in parameters)
                target_norm = self._distributed_gradient_norm(
                    target_gradients,
                    self.accelerator.device,
                )
                self.model.zero_grad(set_to_none=True)
                del target_gradients

                for micro_index, batch in enumerate(group):
                    auxiliary_context = LossContext(
                        self.model,
                        self._prepare_inputs(batch),
                        reference_model=self.reference_model,
                        capture_anchor_hidden=True,
                        global_step=max(0, self.lvd_config.lambda_warmup_steps - 1),
                        # Synchronized backwards through the prepared DDP/FSDP
                        # model average gradients just as training does.
                        compensate_distributed_gradient_average=True,
                    )
                    auxiliary_result = plugin(auxiliary_context)
                    auxiliary_loss = auxiliary_result.loss
                    with torch.no_grad():
                        # Reuse exact auxiliary forwards; pool counts, not rates.
                        batch_metrics.update(auxiliary_result)
                        pooled_metrics.update(auxiliary_result)
                    del auxiliary_result
                    self.accelerator.backward(
                        auxiliary_loss / steps if steps > 1 else auxiliary_loss
                    )
                    if cpu_accumulator is not None:
                        cpu_accumulator.accumulate(
                            self.model, optimizer_boundary=micro_index == steps - 1
                        )
                    del auxiliary_context, auxiliary_loss
                auxiliary_gradients = tuple(parameter.grad for parameter in parameters)
                auxiliary_norm = self._distributed_gradient_norm(
                    auxiliary_gradients,
                    self.accelerator.device,
                )
                self.model.zero_grad(set_to_none=True)
                del auxiliary_gradients
                with torch.no_grad():
                    batch_diagnostics.append(
                        {
                            "batch_index": len(batch_diagnostics),
                            **summarize_diagnostics(
                                batch_metrics.compute(device=self.accelerator.device)
                            ),
                        }
                    )
                    batch_metrics.reset()
                if not math.isfinite(target_norm) or not math.isfinite(auxiliary_norm):
                    raise RuntimeError("LVD lambda calibration produced a non-finite gradient norm")
                if target_norm <= 0 or (
                    auxiliary_norm < 0 if allow_zero_auxiliary else auxiliary_norm <= 0
                ):
                    raise RuntimeError(
                        "LVD lambda calibration requires positive Target and non-negative "
                        "auxiliary gradients"
                        if allow_zero_auxiliary
                        else "LVD lambda calibration requires non-zero Target and auxiliary "
                        "gradients; record more eligible d0 positions or inspect the Anchor batch"
                    )
                ratio = auxiliary_norm / target_norm
                # Only an actual zero auxiliary norm may produce a zero ratio;
                # positive-norm arithmetic underflow is not safety-core evidence.
                if not math.isfinite(ratio) or (
                    ratio <= 0 and (not allow_zero_auxiliary or auxiliary_norm != 0)
                ):
                    raise RuntimeError(
                        "LVD calibration produced a non-finite or zero gradient ratio"
                    )
                target_norms.append(target_norm)
                auxiliary_norms.append(auxiliary_norm)
                ratios.append(ratio)
        finally:
            self.model.zero_grad(set_to_none=True)
            if cpu_accumulator is not None:
                cpu_accumulator.buffers.clear()
            self.model.train(was_training)
            plugin.set_lambda(previous_lambda)
        median_ratio = (
            calibration_normalizer(ratios) if statistic == "rms" else statistics.median(ratios)
        )
        if statistic == "median":
            if allow_zero_auxiliary and median_ratio <= 0:
                raise RuntimeError(
                    "median auxiliary gradient ratio is zero; "
                    "pilot drift remains inside safety core, "
                    "increase covered groups or use validated later pilot"
                )
            if not math.isfinite(median_ratio) or median_ratio <= 0:
                raise RuntimeError(
                    "LVD lambda calibration produced an invalid median gradient ratio"
                )
            normalizer = median_ratio
        else:
            normalizer = calibration_normalizer(ratios, statistic)
            if normalizer <= 0:
                raise RuntimeError(
                    "RMS auxiliary gradient ratio is zero; calibration cannot normalize"
                )
        resolved = self.lvd_config.target_gradient_ratio / normalizer
        if not math.isfinite(resolved) or resolved < 0 or (statistic == "rms" and resolved == 0):
            raise RuntimeError(f"LVD lambda calibration produced invalid lambda: {resolved}")
        weighted_ratios = [resolved * ratio for ratio in ratios]
        if not all(math.isfinite(ratio) for ratio in weighted_ratios):
            raise RuntimeError("LVD calibration produced a non-finite weighted gradient ratio")
        with torch.no_grad():
            pooled_diagnostics = summarize_diagnostics(
                pooled_metrics.compute(device=self.accelerator.device)
            )
        report = {
            "calibration_version": 3 if grouped_report else 2,
            "gradient_convention": "post_backward_distributed_average_compensated",
            "world_size": self.accelerator.num_processes,
            "mode": "gradient_ratio",
            "target_gradient_ratio": self.lvd_config.target_gradient_ratio,
            "calibration_batches": self.lvd_config.calibration_batches,
            "calibration_accumulation_steps": steps,
            "calibration_group_count": len(ratios),
            "calibration_microbatch_count": microbatch_count,
            "calibration_total_microbatch_count": microbatch_count * self.accelerator.num_processes,
            "calibration_batch_source": "explicit"
            if explicit_batches is not None
            else "train_dataloader",
            "calibration_loader_restarts": loader_restarts,
            "training_gradient_accumulation_steps": getattr(
                self.args, "gradient_accumulation_steps", 1
            ),
            "gradient_accumulation_convention": "mean_of_per_microbatch_losses",
            "target_gradient_norms": target_norms,
            "auxiliary_gradient_norms": auxiliary_norms,
            "auxiliary_to_target_ratios": ratios,
            "median_auxiliary_to_target_ratio": median_ratio,
            "ratio_statistics": _calibration_ratio_statistics(
                ratios, allow_zeros=allow_zero_auxiliary
            ),
            "weighted_auxiliary_to_target_ratios": weighted_ratios,
            "median_weighted_auxiliary_to_target_ratio": calibration_normalizer(weighted_ratios)
            if statistic == "rms"
            else statistics.median(weighted_ratios),
            "resolved_lambda": resolved,
            "diagnostics_version": 2 if allow_zero_auxiliary else 1,
            "response_mode": response_mode.value,
            "batch_diagnostics": batch_diagnostics,
            "pooled_diagnostics": pooled_diagnostics,
            "diagnostic_notes": {
                "aggregation": (
                    "Global summed counts/per-position sums; "
                    "zero or missing denominators are null. "
                    "Batch indices identify calibration groups and align with the "
                    "gradient-norm/ratio arrays; group size 1 retains the legacy meaning."
                ),
                "active_exponent_cap_rate": (
                    "Valid pairs with the active exp3 (1 <= rho < 1/gamma0) or exp4 "
                    "(rho >= 1/gamma0) strictly above the cap / all valid pairs. "
                    "Unlike legacy exponent_clamp_count, no inactive/double counting."
                ),
                "zero_response_gradient_crossing_rate": (
                    "Legacy FOUR_SEGMENT active-cap saturation AND d_new <= 0 / actual "
                    "d_new <= 0 crossings. Linear-tail and other modes contribute no saturated "
                    "crossings. This is not a count of zero full-objective gradients."
                ),
                "capacity_means": (
                    "Per-cached-position means of 1/sum(alpha^2) and max(alpha); "
                    "capacity_position_count equals cached positions when weights are available."
                ),
                "hidden_norm_means": (
                    "Cached-position means; norm ratio averages ||h|| / "
                    "max(||h_reference||, normal_epsilon), not a ratio of pooled norms."
                ),
                "gradient_norms": (
                    "Post-backward norms retain distributed-average compensation. FSDP sums "
                    "disjoint shards; DDP replica sums scale both norms by sqrt(world_size), "
                    "which cancels in the ratio. Each group sums gradients of the original "
                    "microbatch losses divided by calibration_accumulation_steps; LVD pressure "
                    "is never pooled across microbatches. No diagnostic rejection threshold or "
                    "automatic pilot/target-ratio selection is applied."
                ),
                "ratio_statistics": (
                    "Unweighted group ratios. Log spread is max(log10(r)) - min(log10(r)); "
                    "half-splits preserve group order (floor(n/2), remainder). Relative median "
                    "gap divides by the larger median; log10 ratio avoids overflow. "
                    "A missing first half is null. Descriptive only, not an acceptance claim."
                ),
            },
        }
        if allow_zero_auxiliary:
            zero_groups = sum(norm == 0 for norm in auxiliary_norms)
            report.update(
                response_region_schema={
                    "version": 1,
                    "count": response_mode.region_count,
                    "names": list(response_mode.region_names),
                },
                zero_auxiliary_groups=zero_groups,
                positive_auxiliary_groups=len(ratios) - zero_groups,
            )
            report["diagnostic_notes"].update(
                active_exponent_cap_rate=(
                    "Valid crossed-region pairs (rho >= 1/gamma0) with "
                    "mu3 * (rho - 1/gamma0) strictly above the cap / all valid pairs. "
                    "The linear continuation retains a nonzero response derivative."
                ),
                zero_response_gradient_crossing_rate=(
                    "Three-segment crossings do not saturate at the exponential cap. "
                    "This legacy saturation counter is not the number of zero auxiliary "
                    "gradient groups or zero safety-core responses."
                ),
                ratio_statistics=(
                    "All raw group ratios, including zeros, enter quantiles and ordered "
                    "half-split medians (floor(n/2), remainder). Log10 statistics use only "
                    "strictly positive ratios; zero/positive group counts are explicit. "
                    "Undefined half-median log ratios are null. Descriptive only."
                ),
                auxiliary_gradient_groups=(
                    "Zero auxiliary norms are legal for the three-segment safety core. "
                    "Every group is retained without skipping, mutation, or epsilon; "
                    "Target norms must be finite and strictly positive. Lambda uses the "
                    "strictly positive median of all raw group ratios, not just active groups."
                ),
            )
        if statistic == "rms":
            weighted_rms = calibration_normalizer(weighted_ratios, statistic)
            if weighted_rms <= 0:
                raise RuntimeError(
                    "LVD calibration produced an invalid weighted RMS gradient ratio"
                )
            report.update(
                calibration_statistic="rms",
                normalization_value=normalizer,
                rms_auxiliary_to_target_ratio=normalizer,
                rms_weighted_auxiliary_to_target_ratio=weighted_rms,
            )
            report["diagnostic_notes"].update(
                auxiliary_gradient_groups=(
                    "Zero auxiliary norms are legal for the three-segment safety core. "
                    "Every group is retained without skipping, mutation, or epsilon; "
                    "Target norms must be finite and strictly positive. Lambda uses RMS "
                    "of all raw group ratios, including zeros, not just active groups. "
                    "The reported raw and weighted medians remain actual medians."
                ),
                normalization=(
                    "lambda = target_gradient_ratio / RMS(raw group ratios). For these G "
                    "measured groups, max(lambda * ratio) <= target_gradient_ratio * sqrt(G). "
                    "This algebraic bound is not statistical robustness, outlier-proofing, "
                    "a guarantee of pilot health, or a bound on future training groups."
                ),
            )
        if grouped_report:
            rank_plans = _gather_calibration_objects(
                {
                    "rank": self.args.process_index,
                    "microbatch_count": microbatch_count,
                    "loader_restarts": loader_restarts,
                    "microbatches": local_plan,
                }
            )
            plan = {
                "plan_version": 1,
                "source": report["calibration_batch_source"],
                "world_size": self.accelerator.num_processes,
                "group_count": len(ratios),
                "microbatches_per_group": steps,
                "ranks": rank_plans,
            }
            report["calibration_plan"] = plan
            report["calibration_plan_sha256"] = hashlib.sha256(
                json.dumps(plan, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
                    "utf-8"
                )
            ).hexdigest()
            report["group_diagnostics"] = [
                {**row, "group_index": row["batch_index"], "microbatch_count": steps}
                for row in batch_diagnostics
            ]
            report["diagnostic_notes"]["calibration_plan"] = (
                "SHA-256 of canonical JSON (sort_keys=True, separators=(',', ':'), UTF-8). "
                "Ordered rank/group/microbatch sample IDs and loader passes, not tensor or "
                "checkpoint contents. Missing IDs are null; absent streams are empty lists. "
                "Explicit input is rank-local and never cycled; extra batches are not consumed. "
                "Microbatch counts are per rank; total count includes all ranks."
            )
        if output_path is not None and self.is_world_process_zero():
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
            temporary.replace(path)
        plugin.set_lambda(resolved)
        self.lvd_config.resolved_lambda = resolved
        return report

    def train(
        self,
        resume_from_checkpoint: str | bool | None = None,
        trial: Any = None,
        ignore_keys_for_eval: list[str] | None = None,
        **kwargs: Any,
    ):
        if self.method_config.name.uses_dynamic_geometry:
            from sft_conflict.trainers.anchor_geometry import restore_geometry_state

            if resume_from_checkpoint is True:
                resume_from_checkpoint = get_last_checkpoint(self.args.output_dir)
                if resume_from_checkpoint is None:
                    raise ValueError("dynamic geometry resume requested without a checkpoint")
            if resume_from_checkpoint:
                restore_geometry_state(self, resume_from_checkpoint)
            elif self.anchor_geometry_config.resolved_lambda is not None:
                raise ValueError("fresh dynamic geometry cannot reuse a post-warmup lambda")
        if self.method_config.name.uses_lvd_plus:
            config = self.lvd_plus_config
            if (
                resume_from_checkpoint
                and config is not None
                and config.lambda_mode.value != "fixed"
                and config.resolved_lambda is None
                and self.resolved_lvd_plus_lambda is None
            ):
                raise ValueError("LVD+ resume requires the checkpoint's resolved lambda")
            # Main normally calls this before its step-zero audit. Direct Trainer
            # users get the same Base-only workflow, never a silent unit lambda.
            self.calibrate_lvd_plus_lambda(
                Path(self.args.output_dir) / "lvd-plus-lambda-calibration.json"
            )
            with reuse_lvd_plus_prepared_model(self):
                return super().train(
                    resume_from_checkpoint=resume_from_checkpoint,
                    trial=trial,
                    ignore_keys_for_eval=ignore_keys_for_eval,
                    **kwargs,
                )
        return super().train(
            resume_from_checkpoint=resume_from_checkpoint,
            trial=trial,
            ignore_keys_for_eval=ignore_keys_for_eval,
            **kwargs,
        )

    def get_train_dataloader(self) -> DataLoader:
        data_loader = DataLoader(
            self.dual_stream_dataset,
            batch_size=self.data_config.target_batch_size,
            sampler=EpochAwareSequentialSampler(self.dual_stream_dataset),
            collate_fn=self.dual_stream_collator,
            num_workers=self.data_config.num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            persistent_workers=self.data_config.num_workers > 0,
        )
        return self.accelerator.prepare(data_loader)

    def create_optimizer(self):
        optimizer = super().create_optimizer()
        self._configure_optimizer_backend(optimizer)
        self.parameter_invariant_callback.validate(optimizer)
        return optimizer

    def create_optimizer_and_scheduler(self, num_training_steps: int) -> None:
        super().create_optimizer_and_scheduler(num_training_steps)
        if getattr(self, "_lvd_plus_reuse_distributed_prepared", False):
            # Trainer's already-wrapped branch skips accelerator.prepare(model,
            # optimizer). Keep its ordinary scheduler and prepare only optimizer.
            prepared = self._lvd_plus_prepared_model
            if not any(model is prepared for model in self.accelerator._models):
                self.accelerator._models.append(prepared)
            self.optimizer = self.accelerator.prepare_optimizer(self.optimizer)

    def _configure_optimizer_backend(self, optimizer: Any) -> None:
        underlying_optimizer = getattr(optimizer, "optimizer", optimizer)
        if self.is_fsdp_enabled and isinstance(underlying_optimizer, torch.optim.AdamW):
            underlying_optimizer.defaults["foreach"] = False
            for group in underlying_optimizer.param_groups:
                group["foreach"] = False

    def _uses_rank_local_fsdp_optimizer_state(self) -> bool:
        return self.is_fsdp_enabled and self.resolved_config is not None

    def save_model(self, output_dir: str | None = None, _internal_call: bool = False) -> None:
        if not self.is_fsdp_enabled:
            super().save_model(output_dir, _internal_call=_internal_call)
            return

        destination = Path(output_dir or self.args.output_dir)
        plugin = self.accelerator.state.fsdp_plugin
        if not _internal_call:
            self._reclaim_fsdp_checkpoint_memory()
            with _temporary_fsdp_state_dict_type(plugin, StateDictType.FULL_STATE_DICT):
                super().save_model(str(destination), _internal_call=False)
            self._reclaim_fsdp_checkpoint_memory()
            return

        self._wait_for_checkpoint_processes()
        self._save_sharded_fsdp_model(destination)
        if self.args.should_save:
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_model.config.save_pretrained(destination)
            generation_config = getattr(unwrapped_model, "generation_config", None)
            if generation_config is not None:
                generation_config.save_pretrained(destination)
            if self.processing_class is not None:
                self.processing_class.save_pretrained(destination)
            torch.save(self.args, destination / TRAINING_ARGS_NAME)
        self._wait_for_checkpoint_processes()

    def _get_fsdp_checkpoint_process_group(self):
        if not distributed.is_available() or not distributed.is_initialized():
            return None
        if distributed.get_world_size() <= 1:
            return None
        process_group = getattr(self, "_fsdp_checkpoint_process_group", None)
        if process_group is None:
            if not distributed.is_gloo_available():
                raise RuntimeError("FSDP checkpointing requires the Gloo distributed backend")
            process_group = distributed.new_group(backend="gloo")
            self._fsdp_checkpoint_process_group = process_group
            if self.args.should_save:
                print("FSDP checkpoint metadata backend: gloo", flush=True)
        return process_group

    def _wait_for_checkpoint_processes(self) -> None:
        process_group = self._get_fsdp_checkpoint_process_group()
        if process_group is None:
            self.accelerator.wait_for_everyone()
        else:
            distributed.barrier(group=process_group)

    def _save_sharded_fsdp_model(self, destination: Path) -> None:
        import torch.distributed.checkpoint as distributed_checkpoint
        from torch.distributed.checkpoint.default_planner import DefaultSavePlanner
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        plugin = self.accelerator.state.fsdp_plugin
        if plugin.state_dict_type is not StateDictType.SHARDED_STATE_DICT:
            raise RuntimeError("internal FSDP checkpoints must use SHARDED_STATE_DICT")
        checkpoint_dir = destination / f"{FSDP_MODEL_NAME}_0"
        staging_dir = destination / ".model-shards.incomplete"
        if self.args.should_save:
            shutil.rmtree(staging_dir, ignore_errors=True)
            staging_dir.mkdir(parents=True, exist_ok=True)
        self._wait_for_checkpoint_processes()
        process_group = self._get_fsdp_checkpoint_process_group()
        with FSDP.state_dict_type(
            self.model,
            plugin.state_dict_type,
            plugin.state_dict_config,
            plugin.optim_state_dict_config,
        ):
            state_dict = {"model": self.model.state_dict()}
            distributed_checkpoint.save(
                state_dict=state_dict,
                storage_writer=distributed_checkpoint.FileSystemWriter(staging_dir),
                planner=DefaultSavePlanner(),
                process_group=process_group,
            )
        self._wait_for_checkpoint_processes()
        if self.args.should_save:
            if not (staging_dir / ".metadata").is_file():
                raise RuntimeError("distributed FSDP checkpoint did not write metadata")
            shutil.rmtree(checkpoint_dir, ignore_errors=True)
            os.replace(staging_dir, checkpoint_dir)
        self._wait_for_checkpoint_processes()

    def _load_sharded_fsdp_model(self, source: Path, model: nn.Module) -> None:
        import torch.distributed.checkpoint as distributed_checkpoint
        from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        checkpoint_dir = source / f"{FSDP_MODEL_NAME}_0"
        if not (checkpoint_dir / ".metadata").is_file():
            raise FileNotFoundError(f"incomplete sharded FSDP checkpoint: {checkpoint_dir}")
        plugin = self.accelerator.state.fsdp_plugin
        process_group = self._get_fsdp_checkpoint_process_group()
        with FSDP.state_dict_type(
            model,
            plugin.state_dict_type,
            plugin.state_dict_config,
            plugin.optim_state_dict_config,
        ):
            state_dict = {"model": model.state_dict()}
            distributed_checkpoint.load(
                state_dict=state_dict,
                storage_reader=distributed_checkpoint.FileSystemReader(checkpoint_dir),
                planner=DefaultLoadPlanner(),
                process_group=process_group,
            )
            model.load_state_dict(state_dict["model"])

    def _save_optimizer_and_scheduler(self, output_dir: str) -> None:
        if not self._uses_rank_local_fsdp_optimizer_state():
            super()._save_optimizer_and_scheduler(output_dir)
            return

        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        rank = self.args.process_index
        world_size = self.args.world_size
        optimizer_path = destination / _rank_local_optimizer_name(rank, world_size)
        temporary_optimizer = optimizer_path.with_suffix(optimizer_path.suffix + ".tmp")
        torch.save(self.optimizer.state_dict(), temporary_optimizer)
        os.replace(temporary_optimizer, optimizer_path)
        self._wait_for_checkpoint_processes()

        if self.args.should_save:
            expected_files = [
                _rank_local_optimizer_name(file_rank, world_size) for file_rank in range(world_size)
            ]
            missing_files = [
                file_name for file_name in expected_files if not (destination / file_name).is_file()
            ]
            if missing_files:
                raise RuntimeError(
                    f"rank-local FSDP optimizer checkpoint is missing files: {missing_files}"
                )
            scheduler_path = destination / SCHEDULER_NAME
            temporary_scheduler = scheduler_path.with_suffix(scheduler_path.suffix + ".tmp")
            torch.save(self.lr_scheduler.state_dict(), temporary_scheduler)
            os.replace(temporary_scheduler, scheduler_path)
            manifest = {
                "schema_version": 3,
                "format": "rank_local_fsdp",
                "world_size": world_size,
                "optimizer": self.resolved_config.training.optimizer,
                "training_mode": self.resolved_config.model.training_mode.value,
                "use_orig_params": self.accelerator.state.fsdp_plugin.use_orig_params,
                "files": expected_files,
            }
            manifest_path = destination / RANK_LOCAL_FSDP_MANIFEST
            temporary_manifest = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
            temporary_manifest.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary_manifest, manifest_path)
        self._wait_for_checkpoint_processes()

    def _load_optimizer_and_scheduler(self, checkpoint: str | None) -> None:
        if checkpoint is None:
            return
        source = Path(checkpoint)
        manifest_path = source / RANK_LOCAL_FSDP_MANIFEST
        if not manifest_path.is_file():
            if self.is_fsdp_enabled and (source / f"{FSDP_MODEL_NAME}.bin").is_file():
                with _temporary_fsdp_state_dict_type(
                    self.accelerator.state.fsdp_plugin,
                    StateDictType.FULL_STATE_DICT,
                ):
                    super()._load_optimizer_and_scheduler(checkpoint)
            else:
                super()._load_optimizer_and_scheduler(checkpoint)
            self._configure_optimizer_backend(self.optimizer)
            return
        if not self._uses_rank_local_fsdp_optimizer_state():
            raise ValueError(
                "rank-local FSDP optimizer checkpoint requires the original "
                "Full FT/FSDP/use_orig_params=False configuration"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_format = (manifest.get("schema_version"), manifest.get("format"))
        supported_formats = {
            (1, "rank_local_bnb_fsdp"),
            (2, "rank_local_flat_parameter_fsdp"),
            (3, "rank_local_fsdp"),
        }
        if manifest_format not in supported_formats:
            raise ValueError("unsupported rank-local FSDP optimizer checkpoint format")
        if int(manifest.get("world_size", 0)) != self.args.world_size:
            raise ValueError(
                "rank-local FSDP optimizer checkpoint world size differs from the current run"
            )
        expected_use_orig_params = self.accelerator.state.fsdp_plugin.use_orig_params
        if manifest.get("optimizer") != self.resolved_config.training.optimizer:
            raise ValueError("rank-local FSDP optimizer checkpoint configuration is incompatible")
        if manifest_format == (3, "rank_local_fsdp"):
            if (
                manifest.get("training_mode") != self.resolved_config.model.training_mode.value
                or manifest.get("use_orig_params") is not expected_use_orig_params
            ):
                raise ValueError(
                    "rank-local FSDP optimizer checkpoint configuration is incompatible"
                )
        elif (
            self.resolved_config.model.training_mode.value != "full_ft"
            or expected_use_orig_params
            or manifest.get("use_orig_params") is not False
        ):
            raise ValueError("legacy rank-local FSDP optimizer checkpoint is incompatible")
        expected_files = [
            _rank_local_optimizer_name(file_rank, self.args.world_size)
            for file_rank in range(self.args.world_size)
        ]
        if manifest.get("files") != expected_files or any(
            not (source / file_name).is_file() for file_name in expected_files
        ):
            raise FileNotFoundError(f"rank-local FSDP optimizer checkpoint is incomplete: {source}")
        optimizer_path = source / _rank_local_optimizer_name(
            self.args.process_index,
            self.args.world_size,
        )
        scheduler_path = source / SCHEDULER_NAME
        if not scheduler_path.is_file():
            raise FileNotFoundError(f"rank-local FSDP optimizer checkpoint is incomplete: {source}")
        optimizer_state = torch.load(
            optimizer_path,
            # CPU-offloaded FSDP can have multi-GB Adam shards. Let the optimizer
            # place state with its parameters instead of staging it all on CUDA.
            map_location="cpu",
            weights_only=True,
        )
        self.optimizer.load_state_dict(optimizer_state)
        self._configure_optimizer_backend(self.optimizer)
        scheduler_state = torch.load(scheduler_path, map_location="cpu", weights_only=True)
        self.lr_scheduler.load_state_dict(scheduler_state)
        self._wait_for_checkpoint_processes()

    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys=None,
        metric_key_prefix: str = "eval",
    ):
        if not self._evaluation_tensor_preview_printed:
            self._evaluation_tensor_preview_printed = True
            preview_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
            if (
                self.is_world_process_zero()
                and preview_dataset is not None
                and self.preview_tokenizer is not None
            ):
                preview = build_random_tensor_preview(
                    preview_dataset,
                    self.preview_tokenizer,
                    stage="evaluation",
                    pad_token_id=self.preview_tokenizer.pad_token_id,
                    pad_to_multiple_of=self.data_config.pad_to_multiple_of,
                )
                print(render_tensor_preview(preview), flush=True)
        accumulator = self._metric_accumulators["eval"]
        accumulator.reset()
        try:
            metrics = super().evaluate(
                eval_dataset=eval_dataset,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
        finally:
            # Failed/repeated evaluations must not leak a partial window. Never
            # reset the pending training window (including the resume path).
            accumulator.reset()
        if self.native_selection is not None and (
            eval_dataset is None or eval_dataset is self.eval_dataset
        ):
            # The native metric belongs to the training validation split only; other
            # datasets (the held-out evaluation split) keep the plain loss metrics.
            native_metrics = self._native_selection_metrics(metric_key_prefix)
            # Bypass the window-aware log override: this is a standalone record.
            Trainer.log(self, native_metrics)
            metrics.update(native_metrics)
        return metrics

    def _native_selection_metrics(self, prefix: str) -> dict[str, float]:
        selection = self.native_selection
        model = self.model
        was_training = model.training
        model.eval()
        try:
            correct, total = sharded_classification_counts(
                selection.evaluator,
                model,
                selection.tokenizer,
                selection.dataset,
                batch_size=selection.batch_size,
                rank=self.accelerator.process_index,
                world_size=self.accelerator.num_processes,
                inference_mode=not self.is_fsdp_enabled,
            )
        finally:
            model.train(was_training)
        counts = torch.tensor([correct, total], dtype=torch.float64, device=self.args.device)
        counts = self.accelerator.reduce(counts, reduction="sum")
        correct, total = (float(value) for value in counts.tolist())
        if total != len(selection.dataset):
            raise RuntimeError(
                f"native selection scored {total:g} of {len(selection.dataset)} validation samples"
            )
        return {
            f"{prefix}_native_accuracy": correct / total,
            f"{prefix}_native_correct": correct,
            f"{prefix}_native_sample_count": total,
        }

    def _record_loss_result(self, result: LossResult) -> None:
        phase = "train" if self.model.training else "eval"
        accumulator = self._metric_accumulators[phase]
        # Keep _record_total_loss on the same window without changing the loss or
        # training path. Evaluation must neither consume nor reset train counts.
        self._metric_totals = accumulator.totals
        self._metric_observations = accumulator.observations
        accumulator.update(result)

    def _record_total_loss(self, total_loss: Tensor) -> None:
        self._metric_totals["loss/total"] += float(total_loss.detach().float().item())
        self._metric_observations["loss/total"] += 1

    def _evaluation_valid_rows(self, batch_size: int) -> int | None:
        """Rank-local prefix retained by Accelerate's rank-ordered tail gather.

        Only the standard fixed-size, sequential, evenly padded evaluation
        shards have a verified prefix rule. Unknown layouts must not advertise
        raw padded observations as duplicate-free global counts.
        """
        world_size = self.accelerator.num_processes
        reduction_world_size = (
            distributed.get_world_size()
            if distributed.is_available() and distributed.is_initialized()
            else 1
        )
        if world_size != reduction_world_size:
            return None
        if world_size == 1:
            return batch_size
        state = self.accelerator.gradient_state
        loader = state.active_dataloader
        shard = getattr(loader, "batch_sampler", None)
        if (
            self.args.use_legacy_prediction_loop
            or not isinstance(loader, DataLoaderShard)
            or not isinstance(shard, BatchSamplerShard)
            or not isinstance(getattr(shard.batch_sampler, "sampler", None), SequentialSampler)
            or not shard.even_batches
            or shard.num_processes != world_size
            or shard.process_index != self.accelerator.process_index
            or loader.total_batch_size != batch_size * world_size
        ):
            return None
        if shard.drop_last:
            return batch_size
        if not 0 <= state.remainder < loader.total_batch_size:
            return None
        if not state.end_of_dataloader or state.remainder == 0:
            return batch_size
        return max(0, min(batch_size, state.remainder - shard.process_index * batch_size))

    def _record_evaluation_metrics(
        self, context: LossContext, results: list[LossResult], total_loss: Tensor
    ) -> None:
        target = context.stream_batch("target")
        if target is None:
            raise ValueError("evaluation metrics require a target batch")
        batch_size = target["input_ids"].shape[0]
        valid_rows = self._evaluation_valid_rows(batch_size)
        accumulator = self._metric_accumulators["eval"]
        supported_plugins = (AnchorCrossEntropy, AnchorKLDistillation, NullLVDLoss, AnchorLVDLoss)
        if getattr(self, "capture_anchor_label_positions", False):
            from sft_conflict.losses.anchor_geometry import AnchorGeometryLoss
            from sft_conflict.losses.lvd_plus import AnchorLVDPlusLoss

            supported_plugins = (*supported_plugins, AnchorLVDPlusLoss, AnchorGeometryLoss)
        if valid_rows is None or (
            self.accelerator.num_processes > 1
            and (
                context.stream_batch("anchor") is not None
                or "regression_labels" in target
                or type(self.loss_plugins[0]) is not TargetCrossEntropy
                or any(type(plugin) not in supported_plugins for plugin in self.loss_plugins[1:])
            )
        ):
            # Reduce this sentinel with the existing log-time metrics collective:
            # one unsupported rank invalidates the ENTIRE diagnostic window.
            accumulator.sum_totals["_eval_metrics_unsupported"] += 1
            return
        if valid_rows == 0:
            # In particular, do not add a zero loss observation to legacy means.
            return
        if valid_rows < batch_size:
            # Only diagnostics are sliced, AFTER the original loss/forward. CE
            # must be recomputed, not scaled by a row fraction: label counts and
            # losses differ per example. The cache prevents another model pass.
            metric_context = LossContext(
                context.model,
                {"target": {"labels": target["labels"][:valid_rows]}},
                reference_model=None,
                capture_anchor_hidden=False,
            )
            metric_context.current_outputs["target"] = SimpleNamespace(
                logits=context.current_outputs["target"].logits[:valid_rows]
            )
            results[0] = TargetCrossEntropy()(metric_context)
            # All other supported plugins have absent Anchor and zero loss.
            total_loss = results[0].loss
        for result in results:
            accumulator.update(result)
        accumulator.sum_totals["samples/target"] += valid_rows
        accumulator.totals["loss/total"] += float(total_loss.detach().float().item())
        accumulator.observations["loss/total"] += 1

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Tensor | None = None,
    ):
        del num_items_in_batch
        anchor_batch = inputs.get("anchor")
        if anchor_batch is not None and self.anchor_composition_path is not None:
            task_counts = Counter(
                str(
                    metadata.get("task_name")
                    or metadata.get("task")
                    or metadata.get("task_source")
                    or "unknown"
                )
                for metadata in anchor_batch.get("source_metadata", [])
            )
            self.anchor_composition_path.parent.mkdir(parents=True, exist_ok=True)
            with self.anchor_composition_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "optimizer_step": self.state.global_step,
                            "micro_batch": self._anchor_micro_batch,
                            "task_counts": dict(sorted(task_counts.items())),
                            "sample_ids": anchor_batch.get("sample_ids", []),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
            self._anchor_micro_batch += 1
        context = LossContext(
            model,
            inputs,
            reference_model=self.reference_model,
            capture_anchor_hidden=self.capture_anchor_hidden,
            global_step=self.state.global_step,
            combine_current_streams=(
                self.is_fsdp_enabled
                and len(self.loss_plugins) >= 2
                and inputs.get("anchor") is not None
            ),
            **(
                {"capture_anchor_label_positions": True}
                if getattr(self, "capture_anchor_label_positions", False)
                else {}
            ),
            **(
                {
                    "anchor_geometry_plugin": self._anchor_geometry_plugin(),
                    "anchor_geometry_epoch": int(self.state.epoch or 0),
                }
                if self.method_config.name.uses_dynamic_geometry
                else {}
            ),
        )
        total_loss: Tensor | None = None
        evaluation_results: list[LossResult] | None = [] if self._in_prediction_step else None
        for plugin in self.loss_plugins:
            result = plugin(context)
            total_loss = result.loss if total_loss is None else total_loss + result.loss
            if evaluation_results is None:
                self._record_loss_result(result)
            else:
                evaluation_results.append(result)
        if total_loss is None:
            raise RuntimeError("no loss plugins were registered")
        if evaluation_results is None:
            self._record_total_loss(total_loss)
        else:
            self._record_evaluation_metrics(context, evaluation_results, total_loss)
        outputs = context.current_outputs.get("target")
        return (total_loss, outputs) if return_outputs else total_loss

    def training_step(
        self,
        model: nn.Module,
        inputs: dict[str, Any],
        num_items_in_batch: Tensor | None = None,
    ) -> Tensor:
        loss = self._training_step_impl(model, inputs, num_items_in_batch)
        if self._cpu_gradient_accumulator is not None:
            self._cpu_gradient_accumulator.accumulate(
                model, optimizer_boundary=self.accelerator.sync_gradients
            )
        return loss

    def _training_step_impl(
        self,
        model: nn.Module,
        inputs: dict[str, Any],
        num_items_in_batch: Tensor | None = None,
    ) -> Tensor:
        if not self.is_fsdp_enabled or len(self.loss_plugins) < 2 or inputs.get("anchor") is None:
            return super().training_step(model, inputs, num_items_in_batch)

        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()
        inputs = self._prepare_inputs(inputs)
        with self.compute_loss_context_manager():
            total_loss = self.compute_loss(model, inputs)
        detached_total = total_loss.detach()
        self.accelerator.backward(total_loss / self.args.gradient_accumulation_steps)
        del total_loss
        return detached_total / self.args.gradient_accumulation_steps

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys=None,
    ):
        del prediction_loss_only, ignore_keys
        inputs = self._prepare_inputs(inputs)
        # Match Trainer.prediction_step: eval() alone does not disable autograd.
        # In FSDP, grad-enabled evaluation also installs backward/unshard hooks
        # even though evaluation never executes their matching backward pass.
        previous_prediction_step = getattr(self, "_in_prediction_step", False)
        self._in_prediction_step = True
        try:
            with torch.no_grad(), self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
        finally:
            self._in_prediction_step = previous_prediction_step
        return loss.detach().mean(), None, None

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        # Trainer can emit the final train summary while the model is still in
        # eval mode after validation. Ordinary eval logs (also custom prefixes)
        # follow model.eval(); the explicit train log keys take precedence.
        training_log = "loss" in logs or any(name.startswith("train_") for name in logs)
        evaluation_log = any(name.startswith("eval_") for name in logs)
        phase = "train" if training_log or (self.model.training and not evaluation_log) else "eval"
        accumulator = self._metric_accumulators[phase]
        values = accumulator.compute(device=self.args.device)
        if phase == "eval" and values.pop("_eval_metrics_unsupported", 0.0):
            warnings.warn(
                "Custom evaluation metrics omitted: duplicate-free distributed diagnostics "
                "require Target-only CE with standard sequential, evenly padded Accelerate "
                "shards and a known remainder. HF evaluation loss is unchanged.",
                RuntimeWarning,
                stacklevel=2,
            )
            values = {}
        if "loss/target_ce" in values:
            # Compatibility alias remains an unweighted rank/microbatch mean;
            # loss/target_ce_token_mean is the explicitly token-weighted value.
            values["loss/target_ce_microbatch_mean"] = values["loss/target_ce"]
        prefix = ""
        if phase == "eval":
            prefix = next((name[:-5] for name in logs if name.endswith("_loss")), "eval") + "_"
        for name, value in values.items():
            logs.setdefault(f"{prefix}{name}", value)
        accumulator.reset()
        super().log(logs, start_time=start_time)

    def _checkpoint_state_path(self, checkpoint_dir: str | Path) -> Path:
        return Path(checkpoint_dir) / f"dual_stream_state.rank-{self.args.process_index}.json"

    def _reclaim_fsdp_checkpoint_memory(self) -> None:
        if not self.is_fsdp_enabled:
            return
        self._wait_for_checkpoint_processes()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._wait_for_checkpoint_processes()

    def _save_checkpoint(self, model, trial):
        self._reclaim_fsdp_checkpoint_memory()
        super()._save_checkpoint(model, trial)
        run_dir = Path(self._get_output_dir(trial=trial))
        checkpoint_dir = run_dir / f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        if self.resolved_config is not None and self.is_world_process_zero():
            dump_config(self.resolved_config, checkpoint_dir / "resolved-config.yaml")
        if self.method_config.name.uses_dynamic_geometry:
            from sft_conflict.trainers.anchor_geometry import save_geometry_state

            save_geometry_state(self, checkpoint_dir)
        state = self.dual_stream_dataset.state_dict()
        completed_micro_batches = self.state.global_step * self.args.gradient_accumulation_steps
        cursor = completed_micro_batches % self.dual_stream_dataset.per_rank_steps
        state.update(
            {
                "global_step": self.state.global_step,
                "trainer_epoch": self.state.epoch,
                "current_cursor": cursor,
                "consumed_target_examples": completed_micro_batches
                * self.data_config.target_batch_size,
                "gradient_accumulation_micro_step": 0,
                "rank": self.args.process_index,
                "local_rank": self.args.local_process_index,
                "rank_global_batch_offset": self.args.process_index,
                "rank_global_batch_stride": self.args.world_size,
            }
        )
        path = self._checkpoint_state_path(checkpoint_dir)
        temporary_path = path.with_suffix(".json.tmp")
        temporary_path.write_text(
            json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary_path.replace(path)
        self._reclaim_fsdp_checkpoint_memory()

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        state_path = self._checkpoint_state_path(resume_from_checkpoint)
        if not state_path.is_file():
            raise ValueError(f"resume checkpoint is missing dual-stream state: {state_path}")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        self.dual_stream_dataset.validate_resume_state(state)
        trainer_epoch = state.get("trainer_epoch")
        if trainer_epoch is not None:
            epoch = math.floor(float(trainer_epoch))
            self.dual_stream_dataset.set_epoch(epoch)
            if state.get("current_cursor", 0) > 0 and not self.args.ignore_data_skip:
                self.dual_stream_dataset.preserve_epoch_for_resume(epoch)
        source = Path(resume_from_checkpoint)
        return self._load_checkpoint_model_only(source, model or self.model)

    def _load_checkpoint_model_only(self, source: Path, model: nn.Module):
        if self.is_fsdp_enabled and (source / f"{FSDP_MODEL_NAME}_0").is_dir():
            self._load_sharded_fsdp_model(source, model)
            return None
        if self.is_fsdp_enabled and (source / f"{FSDP_MODEL_NAME}.bin").is_file():
            with _temporary_fsdp_state_dict_type(
                self.accelerator.state.fsdp_plugin,
                StateDictType.FULL_STATE_DICT,
            ):
                return super()._load_from_checkpoint(str(source), model=model)
        if self.is_fsdp_enabled and any(
            (source / file_name).is_file()
            for file_name in (
                SAFE_WEIGHTS_NAME,
                SAFE_WEIGHTS_INDEX_NAME,
                WEIGHTS_NAME,
                WEIGHTS_INDEX_NAME,
            )
        ):
            return self._load_standard_model_checkpoint(source, model)
        return super()._load_from_checkpoint(str(source), model=model)

    def _load_standard_model_checkpoint(self, source: Path, model: nn.Module):
        if (source / SAFE_WEIGHTS_INDEX_NAME).is_file() or (source / WEIGHTS_INDEX_NAME).is_file():
            load_result = load_sharded_checkpoint(
                model,
                source,
                strict=False,
                prefer_safe=True,
            )
        elif (source / SAFE_WEIGHTS_NAME).is_file():
            from safetensors.torch import load_file

            state_dict = load_file(source / SAFE_WEIGHTS_NAME, device="cpu")
            load_result = model.load_state_dict(state_dict, strict=False)
            del state_dict
        elif (source / WEIGHTS_NAME).is_file():
            state_dict = torch.load(
                source / WEIGHTS_NAME,
                map_location="cpu",
                weights_only=True,
            )
            load_result = model.load_state_dict(state_dict, strict=False)
            del state_dict
        else:
            raise ValueError(f"standard model weights not found in checkpoint: {source}")
        self._issue_warnings_after_load(load_result)
        return load_result

    def _is_complete_fsdp_checkpoint(self, checkpoint: Path) -> bool:
        sharded_model = checkpoint / f"{FSDP_MODEL_NAME}_0"
        if sharded_model.is_dir():
            return (sharded_model / ".metadata").is_file()
        return any(
            (checkpoint / file_name).is_file()
            for file_name in (
                f"{FSDP_MODEL_NAME}.bin",
                SAFE_WEIGHTS_NAME,
                SAFE_WEIGHTS_INDEX_NAME,
                WEIGHTS_NAME,
                WEIGHTS_INDEX_NAME,
            )
        )

    def _best_complete_checkpoint(self) -> tuple[Path, float] | None:
        metric_name = self.args.metric_for_best_model
        if not metric_name:
            return None
        metric_name = metric_name if metric_name.startswith("eval_") else f"eval_{metric_name}"
        candidates: list[tuple[Path, float]] = []
        for record in self.state.log_history:
            step = record.get("step")
            metric = record.get(metric_name)
            if not isinstance(step, int) or not isinstance(metric, int | float):
                continue
            checkpoint = Path(self.args.output_dir) / f"{PREFIX_CHECKPOINT_DIR}-{step}"
            if self._is_complete_fsdp_checkpoint(checkpoint):
                candidates.append((checkpoint, float(metric)))
        if not candidates:
            return None
        return (max if self.args.greater_is_better else min)(candidates, key=lambda item: item[1])

    def _load_best_model(self):
        checkpoint = Path(self.state.best_model_checkpoint)
        if self.is_fsdp_enabled and not self._is_complete_fsdp_checkpoint(checkpoint):
            fallback = self._best_complete_checkpoint()
            if fallback is None:
                raise FileNotFoundError(
                    f"best FSDP checkpoint is incomplete and no complete evaluated checkpoint "
                    f"is available in the current run: {checkpoint}"
                )
            fallback_checkpoint, fallback_metric = fallback
            warnings.warn(
                f"Best FSDP checkpoint {checkpoint} is incomplete; loading the best complete "
                f"checkpoint from the current run instead: {fallback_checkpoint}",
                RuntimeWarning,
                stacklevel=2,
            )
            checkpoint = fallback_checkpoint
            self.state.best_model_checkpoint = str(checkpoint)
            self.state.best_metric = fallback_metric
        if self.is_fsdp_enabled and (checkpoint / f"{FSDP_MODEL_NAME}_0").is_dir():
            self._reclaim_fsdp_checkpoint_memory()
            self._load_sharded_fsdp_model(checkpoint, self.model)
            self._reclaim_fsdp_checkpoint_memory()
            return None
        if self.is_fsdp_enabled and (checkpoint / f"{FSDP_MODEL_NAME}.bin").is_file():
            with _temporary_fsdp_state_dict_type(
                self.accelerator.state.fsdp_plugin,
                StateDictType.FULL_STATE_DICT,
            ):
                return super()._load_best_model()
        return super()._load_best_model()

    def load_checkpoint_for_evaluation(self, checkpoint: str | Path) -> None:
        self._load_checkpoint_model_only(Path(checkpoint), self.model)
