"""Dataclass schema for reproducible training runs."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath


class TrainingMode(str, Enum):
    FULL_FT = "full_ft"
    BODY_FT_FROZEN_HEAD = "body_ft_frozen_head"
    LORA = "lora"


class MethodName(str, Enum):
    TARGET_ONLY = "target_only"
    TARGET_PLUS_ANCHOR_CE = "target_plus_anchor_ce"
    TARGET_PLUS_ANCHOR_KL = "target_plus_anchor_kl"
    TARGET_PLUS_ANCHOR_LVD = "target_plus_anchor_lvd"
    TARGET_PLUS_ANCHOR_CE_LVD = "target_plus_anchor_ce_lvd"
    TARGET_PLUS_NULL_LVD = "target_plus_null_lvd"
    FILTERING_ANCHOR = "filtering_anchor"
    RECORD_ANCHOR_D0 = "record_anchor_d0"
    TEST_INTERFERENCE = "test_interference"
    TEST_CROSSING = "test_crossing"
    TARGET_PLUS_ANCHOR_LVD_PLUS = "target_plus_anchor_lvd+"
    RECORD_S0 = "record_s0"
    RECORD_LVD_PLUS_SCALES = "record_lvd_plus_scales"
    CALIBRATE_LVD_PLUS_LAMBDA = "calibrate_lvd_plus_lambda"
    TARGET_PLUS_ANCHOR_CE_GEOMETRY = "target_plus_anchor_ce_geometry"
    PREPARE_ANCHOR_GEOMETRY_REFERENCE = "prepare_anchor_geometry_reference"

    @classmethod
    def _missing_(cls, value: object) -> MethodName | None:
        if value == "target_plus_anchor_lvd_plus":
            return cls.TARGET_PLUS_ANCHOR_LVD_PLUS
        return None

    @property
    def uses_lvd(self) -> bool:
        return self in {self.TARGET_PLUS_ANCHOR_LVD, self.TARGET_PLUS_ANCHOR_CE_LVD}

    @property
    def uses_lvd_plus(self) -> bool:
        return self is self.TARGET_PLUS_ANCHOR_LVD_PLUS

    @property
    def is_lvd_plus_preparation(self) -> bool:
        """Action selectors that must normalize to the formal LVD+ method."""
        return self in {self.RECORD_LVD_PLUS_SCALES, self.CALIBRATE_LVD_PLUS_LAMBDA}

    @property
    def uses_lvd_geometry(self) -> bool:
        return self.uses_lvd or self.uses_lvd_plus

    @property
    def uses_dynamic_geometry(self) -> bool:
        return self is self.TARGET_PLUS_ANCHOR_CE_GEOMETRY


class ReferenceMode(str, Enum):
    NONE = "none"
    ONLINE = "online_reference"
    CACHED = "cached_reference"


class AnchorSamplingStrategy(str, Enum):
    TASK_BALANCED = "task_balanced"
    MIXTURE_PROPORTIONAL = "mixture_proportional"
    RANDOM = "random"


class LVDNeighborStrategy(str, Enum):
    RANDOM = "random"
    CURRENT_TOPK = "current_topk"


class LVDResponseMode(str, Enum):
    FOUR_SEGMENT = "four_segment"
    FOUR_SEGMENT_LINEAR_TAIL = "four_segment_linear_tail"
    LINEAR = "linear"
    SOFTPLUS = "softplus"
    HINGE = "hinge"
    THREE_SEGMENT = "three_segment"

    @property
    def region_names(self) -> tuple[str, ...]:
        if self is self.THREE_SEGMENT:
            return ("safety_core", "buffer", "crossed")
        return ("non_eroding", "safety_core", "buffer", "crossed")

    @property
    def region_count(self) -> int:
        return len(self.region_names)


class LVDPressureMode(str, Enum):
    WEIGHTED_SUM = "weighted_sum"
    ASINH_WEIGHTED_SUM = "asinh_weighted_sum"
    MAX_BOUNDARY = "max_boundary"
    RAW_DISPLACEMENT = "raw_displacement"


class LVDMomentMode(str, Enum):
    UNCENTERED_SECOND_MOMENT = "uncentered_second_moment"
    CENTERED_COVARIANCE = "centered_covariance"


class LVDLambdaMode(str, Enum):
    GRADIENT_RATIO = "gradient_ratio"
    FIXED = "fixed"


class LVDPlusAnchorMode(str, Enum):
    HALF_UNEMBEDDING = "half_unembedding"
    RECORDED_S0 = "recorded_s0"


class LVDPlusObjectiveMode(str, Enum):
    PRESSURE_ONLY = "pressure_only"
    PRESSURE_PLUS_RISK = "pressure_plus_risk"


@dataclass
class PathsConfig:
    artifact_root: str = "/data1/chenhang/SFT_Conflict/artifacts"
    run_name: str | None = None
    output_dir: str | None = None
    estimated_run_bytes: int = 0
    minimum_free_bytes: int = 2 * 1024**4
    free_space_multiplier: float = 2.0


@dataclass
class ModelConfig:
    model_id: str
    revision: str
    tokenizer_revision: str | None = None
    model_variant: str = "base"
    training_mode: TrainingMode = TrainingMode.BODY_FT_FROZEN_HEAD
    freeze_unembedding: bool = True
    untie_tied_embeddings: bool | None = None
    trust_remote_code: bool = False
    local_files_only: bool = True
    gradient_checkpointing: bool = False
    use_flash_attention_2: bool = False


@dataclass
class LoraConfig:
    rank: int = 8
    alpha: int = 32
    dropout: float = 0.05
    bias: str = "none"
    target_modules: tuple[str, ...] = ("q_proj", "v_proj")
    modules_to_save: tuple[str, ...] = ()
    train_input_embeddings: bool = False


@dataclass
class DatasetConfig:
    source: str
    path: str
    split: str = "train"
    revision: str | None = None
    name: str | None = None
    data_file: str | None = None
    adapter: str = "prompt_response"
    max_samples: int | None = None
    metadata_path: str | None = None
    selection: str = "all"
    selection_seed: int = 0
    validation_fraction: float = 0.1
    stratify_column: str | None = None
    stratify_regression: bool = False


@dataclass
class DataConfig:
    target: DatasetConfig
    validation: DatasetConfig | None = None
    anchor: DatasetConfig | None = None
    max_length: int = 512
    target_batch_size: int = 1
    anchor_batch_size: int = 1
    anchor_batches_per_target: float = 1.0
    anchor_sampling_strategy: AnchorSamplingStrategy = AnchorSamplingStrategy.TASK_BALANCED
    steps_per_epoch: int | None = None
    num_workers: int = 0
    seed: int = 42
    pad_to_multiple_of: int | None = 8


@dataclass
class MethodConfig:
    name: MethodName = MethodName.TARGET_ONLY
    anchor_weight: float = 1.0
    kl_temperature: float = 1.0
    lvd_weight: float = 0.0


@dataclass
class LVDConfig:
    config_version: int = 1
    d0_cache_path: str | None = None
    position_strategy: str = "all_selected_tokens"
    positions_per_sample: int = 4
    position_seed: int = 42
    d0_filter_mode: str = "base_correct"
    d0_minimum_confidence: float | None = None
    d0_minimum_margin: float | None = None
    d0_storage_dtype: str = "float32"
    d0_samples_per_shard: int = 32
    neighbor_strategy: LVDNeighborStrategy = LVDNeighborStrategy.CURRENT_TOPK
    top_k: int = 512
    position_chunk_size: int = 4
    gamma0: float = 0.2
    mu3: float = 1.0
    mu4: float = 2.0
    capacity_exponent: float = 1.0
    epsilon_d: float = 1e-6
    normal_epsilon: float = 1e-12
    baseline_margin_floor: float = 1e-6
    baseline_negative_tolerance: float = 1e-6
    exponent_clamp: float = 6.0
    response_mode: LVDResponseMode = LVDResponseMode.FOUR_SEGMENT
    use_capacity_weights: bool = True
    pressure_mode: LVDPressureMode = LVDPressureMode.WEIGHTED_SUM
    moment_mode: LVDMomentMode = LVDMomentMode.UNCENTERED_SECOND_MOMENT
    lambda_mode: LVDLambdaMode = LVDLambdaMode.GRADIENT_RATIO
    calibration_checkpoint: str | None = None
    fixed_lambda: float | None = None
    resolved_lambda: float | None = None
    target_gradient_ratio: float = 0.5
    calibration_batches: int = 8
    lambda_warmup_steps: int = 0
    calibration_accumulation_steps: int = 1
    calibration_seed: int | None = None
    pilot_health_samples: int = 128
    pilot_validation_samples: int = 128
    core_transition_fraction: float = 0.1
    calibration_statistic: str = "median"

    def validate(self) -> None:
        if self.config_version != 1:
            raise ValueError("lvd.config_version must be 1")
        if type(self.top_k) is not int or self.top_k <= 0:
            raise ValueError("lvd.top_k must be positive")
        if type(self.position_chunk_size) is not int or self.position_chunk_size <= 0:
            raise ValueError("lvd.position_chunk_size must be a positive integer")
        if self.positions_per_sample <= 0:
            raise ValueError("lvd.positions_per_sample must be positive")
        if self.position_strategy not in {
            "last_non_padding",
            "all_selected_tokens",
            "sample_k_positions",
        }:
            raise ValueError("lvd.position_strategy is invalid")
        if self.position_strategy == "last_non_padding" and self.d0_filter_mode == "base_correct":
            raise ValueError(
                "last_non_padding has no gold next-token target and cannot use base_correct"
            )
        if self.position_seed < 0:
            raise ValueError("lvd.position_seed must be non-negative")
        if self.d0_filter_mode not in {
            "all_reference",
            "base_correct",
            "minimum_confidence",
            "minimum_margin",
        }:
            raise ValueError("lvd.d0_filter_mode is invalid")
        if self.d0_filter_mode == "minimum_confidence":
            if self.d0_minimum_confidence is None or not 0 <= self.d0_minimum_confidence <= 1:
                raise ValueError(
                    "lvd.d0_filter_mode=minimum_confidence requires a threshold in [0, 1]"
                )
        elif self.d0_minimum_confidence is not None:
            raise ValueError("lvd.d0_minimum_confidence requires minimum_confidence mode")
        if self.d0_filter_mode == "minimum_margin":
            if self.d0_minimum_margin is None or self.d0_minimum_margin < 0:
                raise ValueError(
                    "lvd.d0_filter_mode=minimum_margin requires a non-negative threshold"
                )
        elif self.d0_minimum_margin is not None:
            raise ValueError("lvd.d0_minimum_margin requires minimum_margin mode")
        if self.d0_storage_dtype not in {"float32", "bfloat16", "float16"}:
            raise ValueError("lvd.d0_storage_dtype is invalid")
        if self.d0_samples_per_shard <= 0:
            raise ValueError("lvd.d0_samples_per_shard must be positive")
        three_segment = self.response_mode == LVDResponseMode.THREE_SEGMENT
        if not isinstance(self.calibration_statistic, str) or self.calibration_statistic not in {
            "median",
            "rms",
        }:
            raise ValueError("lvd.calibration_statistic must be 'median' or 'rms'")
        if self.calibration_statistic == "rms" and not three_segment:
            raise ValueError(
                "lvd.calibration_statistic='rms' requires response_mode='three_segment'"
            )
        if three_segment:
            for name in ("gamma0", "mu3", "exponent_clamp", "core_transition_fraction"):
                value = getattr(self, name)
                if (
                    not isinstance(value, int | float)
                    or isinstance(value, bool)
                    or not math.isfinite(value)
                ):
                    raise ValueError(f"lvd.{name} must be a finite number")
        if not 0.0 < self.gamma0 < 1.0:
            raise ValueError("lvd.gamma0 must be in (0, 1)")
        if three_segment:
            if self.mu3 <= 0:
                raise ValueError("lvd.mu3 must be positive")
            if not 0.0 < self.core_transition_fraction <= 1.0:
                raise ValueError("lvd.core_transition_fraction must be in (0, 1]")
            if not 0.0 < self.exponent_clamp <= 60.0:
                raise ValueError("lvd.exponent_clamp must be in (0, 60] for three_segment")
        elif self.mu3 <= 0 or self.mu4 <= self.mu3:
            raise ValueError("lvd parameters must satisfy mu4 > mu3 > 0")
        if self.capacity_exponent < 1.0:
            raise ValueError("lvd.capacity_exponent must be at least one")
        if self.epsilon_d <= 0 or self.normal_epsilon <= 0 or self.baseline_margin_floor <= 0:
            raise ValueError("lvd margin stabilizers must be positive")
        if self.baseline_negative_tolerance < 0 or self.exponent_clamp <= 0:
            raise ValueError("lvd tolerances and exponent clamp must be non-negative/positive")
        if not three_segment:
            minimum_c1_clamp = self.mu3 * (1.0 / self.gamma0 - 1.0)
            if self.exponent_clamp < minimum_c1_clamp:
                raise ValueError(
                    "lvd.exponent_clamp must preserve the C1 join at x_bound; "
                    f"requires at least {minimum_c1_clamp}"
                )
        if self.lambda_mode is LVDLambdaMode.FIXED:
            if (
                self.fixed_lambda is None
                or isinstance(self.fixed_lambda, bool)
                or not math.isfinite(self.fixed_lambda)
                or self.fixed_lambda < 0
            ):
                raise ValueError("fixed LVD lambda mode requires fixed_lambda >= 0 and finite")
        elif self.fixed_lambda is not None:
            raise ValueError("fixed_lambda must be null when lambda_mode=gradient_ratio")
        if self.resolved_lambda is not None and (
            isinstance(self.resolved_lambda, bool)
            or not math.isfinite(self.resolved_lambda)
            or self.resolved_lambda < 0
        ):
            raise ValueError("lvd.resolved_lambda must be non-negative and finite")
        if (
            isinstance(self.target_gradient_ratio, bool)
            or not math.isfinite(self.target_gradient_ratio)
            or self.target_gradient_ratio <= 0
        ):
            raise ValueError("lvd.target_gradient_ratio must be positive and finite")
        if type(self.calibration_batches) is not int or self.calibration_batches <= 0:
            raise ValueError("lvd.calibration_batches must be a positive integer")
        if (
            type(self.calibration_accumulation_steps) is not int
            or self.calibration_accumulation_steps <= 0
        ):
            raise ValueError("lvd.calibration_accumulation_steps must be a positive integer")
        if self.lambda_warmup_steps < 0:
            raise ValueError("lvd.lambda_warmup_steps must be non-negative")
        if self.calibration_seed is not None and (
            type(self.calibration_seed) is not int or self.calibration_seed < 0
        ):
            raise ValueError("lvd.calibration_seed must be a non-negative integer or null")
        for name in ("pilot_health_samples", "pilot_validation_samples"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"lvd.{name} must be a positive integer")


@dataclass
class S0Config:
    """Full-vocabulary S0 solver on the radius-bounded hidden-state domain.

    Radius and step size use physical hidden-state units. The audit tolerance
    scales FP64 cell/certificate checks; neither a global unbounded optimum nor
    membership inside every label cell is guaranteed by these settings.
    """

    config_version: int = 1
    radius: float = 512.0
    token_batch_size: int = 128
    vocabulary_chunk_size: int = 32768
    max_iterations: int = 500
    initial_step_size: float = 16.0
    gap_tolerance: float = 1e-4
    audit_tolerance: float = 1e-5

    def validate(self) -> None:
        if type(self.config_version) is not int or self.config_version != 1:
            raise ValueError("s0.config_version must be the integer 1")
        for name in ("token_batch_size", "vocabulary_chunk_size", "max_iterations"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"s0.{name} must be a positive integer")
        for name in ("radius", "initial_step_size", "gap_tolerance", "audit_tolerance"):
            value = getattr(self, name)
            try:
                valid = (
                    isinstance(value, int | float)
                    and not isinstance(value, bool)
                    and math.isfinite(value)
                    and value > 0
                )
            except OverflowError:
                valid = False
            if not valid:
                raise ValueError(f"s0.{name} must be a positive finite number")


@dataclass
class LVDPlusConfig:
    """Label-referenced LVD+ at every eligible supervised next-token position.

    Current-logit Top-K competitors and uniform aggregation are fixed contracts,
    not legacy geometry overrides. Distances use physical hidden-state units,
    never d0 ratios. Gradient-ratio lambda uses RMS measured once at Base before
    training, without a pilot, reference model, d0 cache, or optimizer step.
    The legacy center stays W_j/2 unless recorded S0 is explicitly selected.
    Boundary risk is opt-in; pressure-only keeps the historical response and gain.
    """

    config_version: int = 1
    top_k: int = 512
    position_chunk_size: int = 4
    exponential_rate: float = 0.01
    transition_width: float = 0.01
    exponent_clamp: float = 6.0
    normal_epsilon: float = 1e-12
    moment_mode: LVDMomentMode = LVDMomentMode.UNCENTERED_SECOND_MOMENT
    lambda_mode: LVDLambdaMode = LVDLambdaMode.GRADIENT_RATIO
    fixed_lambda: float | None = None
    resolved_lambda: float | None = None
    target_gradient_ratio: float = 0.0005
    calibration_batches: int = 64
    calibration_accumulation_steps: int = 4
    calibration_seed: int = 20260927
    lambda_warmup_steps: int = 0
    anchor_mode: LVDPlusAnchorMode = LVDPlusAnchorMode.HALF_UNEMBEDDING
    s0_cache_path: str | None = None
    # Filled by cache verification with a manifest/semantic digest, never a path hash.
    s0_cache_sha256: str | None = None
    # Optional v1 extension: gain=1 preserves the original response exactly.
    # q_new = q_old + (gain - 1) * H_tau(u), u = max(s - max(B, 0), 0).
    # Keep s <= 0 exactly zero. H_tau(0) = H_tau'(0) = 0 adds no response
    # or slope jump at the exact boundary. Once fully ramped,
    # q_new' = exp(min(mu * u, C)) + gain - 1.
    post_crossing_slope_gain: float = 1.0
    # Append-only opt-in extension; artifact binding is owned by preparation.
    objective_mode: LVDPlusObjectiveMode = LVDPlusObjectiveMode.PRESSURE_ONLY
    risk_weight: float = 3.0
    risk_temperature: float = 1.0
    resolved_outer_scale: float | None = None
    resolved_risk_scale: float | None = None
    scale_cache_path: str | None = None
    scale_cache_sha256: str | None = None
    lambda_cache_path: str | None = None
    lambda_cache_sha256: str | None = None

    @property
    def uses_boundary_risk(self) -> bool:
        return self.objective_mode is LVDPlusObjectiveMode.PRESSURE_PLUS_RISK

    def _validate_anchor_mode(self) -> None:
        if not isinstance(self.anchor_mode, LVDPlusAnchorMode):
            raise ValueError("lvd_plus.anchor_mode must be half_unembedding or recorded_s0")
        if self.anchor_mode is LVDPlusAnchorMode.HALF_UNEMBEDDING:
            for name in ("s0_cache_path", "s0_cache_sha256"):
                if getattr(self, name) is not None:
                    raise ValueError(f"lvd_plus.{name} is forbidden with half_unembedding")
            return
        # Method YAML may leave the path unresolved for QuickRun to derive. A
        # complete recorded-S0 experiment must supply it (ExperimentConfig below).
        if self.s0_cache_path is not None and (
            not isinstance(self.s0_cache_path, str) or not self.s0_cache_path.strip()
        ):
            raise ValueError("lvd_plus.s0_cache_path must be a non-empty string or null")
        if self.s0_cache_sha256 is not None and (
            not isinstance(self.s0_cache_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", self.s0_cache_sha256) is None
        ):
            raise ValueError("lvd_plus.s0_cache_sha256 must be 64 lowercase hex characters or null")

    def validate(self) -> None:
        self._validate_anchor_mode()
        if not isinstance(self.objective_mode, LVDPlusObjectiveMode):
            raise ValueError("lvd_plus.objective_mode must be pressure_only or pressure_plus_risk")
        if type(self.config_version) is not int or self.config_version != 1:
            raise ValueError("lvd_plus.config_version must be the integer 1")
        for name in (
            "top_k",
            "position_chunk_size",
            "calibration_batches",
            "calibration_accumulation_steps",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"lvd_plus.{name} must be a positive integer")
        for name in ("calibration_seed", "lambda_warmup_steps"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"lvd_plus.{name} must be a non-negative integer")
        positive_fields = (
            "exponential_rate",
            "transition_width",
            "exponent_clamp",
            "normal_epsilon",
            "target_gradient_ratio",
            "post_crossing_slope_gain",
            "risk_weight",
            "risk_temperature",
        )
        optional_positive_fields = ("resolved_outer_scale", "resolved_risk_scale")
        for name in (
            *positive_fields,
            *optional_positive_fields,
            "fixed_lambda",
            "resolved_lambda",
        ):
            value = getattr(self, name)
            if value is None and name not in positive_fields:
                continue
            positive = name in positive_fields or name in optional_positive_fields
            try:
                valid = (
                    isinstance(value, int | float)
                    and not isinstance(value, bool)
                    and math.isfinite(value)
                    and (value > 0 if positive else value >= 0)
                )
            except OverflowError:
                valid = False
            if not valid:
                bound = "positive" if positive else "non-negative"
                raise ValueError(f"lvd_plus.{name} must be a {bound} finite number")
        if not 1.0 <= self.post_crossing_slope_gain <= 1000.0:
            raise ValueError("lvd_plus.post_crossing_slope_gain must be in [1, 1000]")
        if self.exponent_clamp > 20:
            raise ValueError("lvd_plus.exponent_clamp must be at most 20")
        if self.moment_mode is not LVDMomentMode.UNCENTERED_SECOND_MOMENT:
            raise ValueError("lvd_plus.moment_mode must be uncentered_second_moment")
        if not isinstance(self.lambda_mode, LVDLambdaMode):
            raise ValueError("lvd_plus.lambda_mode must be gradient_ratio or fixed")
        if self.lambda_mode is LVDLambdaMode.FIXED:
            if self.fixed_lambda is None:
                raise ValueError("lvd_plus.fixed_lambda is required when lambda_mode=fixed")
        elif self.fixed_lambda is not None:
            raise ValueError("lvd_plus.fixed_lambda must be null when lambda_mode=gradient_ratio")
        for name in ("scale_cache_path", "lambda_cache_path"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"lvd_plus.{name} must be a non-empty string or null")
        for name in ("scale_cache_sha256", "lambda_cache_sha256"):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
            ):
                raise ValueError(f"lvd_plus.{name} must be 64 lowercase hex characters or null")
        if self.uses_boundary_risk:
            if self.post_crossing_slope_gain != 1.0:
                raise ValueError(
                    "lvd_plus.post_crossing_slope_gain must be 1 when "
                    "objective_mode=pressure_plus_risk"
                )
            if (self.resolved_outer_scale is None) != (self.resolved_risk_scale is None):
                raise ValueError(
                    "lvd_plus.resolved_outer_scale and lvd_plus.resolved_risk_scale "
                    "must both be set or both be null"
                )
        else:
            for name in (
                *optional_positive_fields,
                "scale_cache_path",
                "scale_cache_sha256",
                "lambda_cache_path",
                "lambda_cache_sha256",
            ):
                if getattr(self, name) is not None:
                    raise ValueError(f"lvd_plus.{name} requires objective_mode=pressure_plus_risk")
            for name, default in (("risk_weight", 3.0), ("risk_temperature", 1.0)):
                if getattr(self, name) != default:
                    raise ValueError(
                        f"non-default lvd_plus.{name} requires objective_mode=pressure_plus_risk"
                    )


@dataclass
class AnchorGeometryConfig:
    """Base output references and versioned trainable-head drift objectives."""

    config_version: int = 1
    objective: str = "centered_v1"
    band_scale_ratio: float = 0.1
    band_temperature: float = 2.0
    band_frequency_strength: float = 0.1
    band_scale_top_k: int = 128
    band_solver_iterations: int = 48
    band_min_calibration_rms: float = 1e-6
    current_directions: int = 128
    base_directions: int = 128
    coverage_directions: int = 256
    current_mass: float = 0.4
    base_mass: float = 0.3
    coverage_mass: float = 0.3
    base_top_k: int = 512
    coverage_cap_multiplier: float = 8.0
    coverage_seed: int = 42
    geometry_delta: float = 0.01
    normal_epsilon: float = 1e-6
    hidden_epsilon: float = 1e-6
    position_chunk_size: int = 4
    reference_samples_per_shard: int = 32
    reference_cache_path: str | None = None
    reference_sha256: str | None = None
    geometry_start_step: int = 50
    calibration_groups: int = 16
    calibration_seed: int = 42
    geometry_gradient_ratio: float = 0.05
    geometry_ramp_steps: int = 20
    resolved_lambda: float | None = None
    calibrated_at_step: int | None = None
    calibration_sha256: str | None = None

    def validate(self) -> None:
        if type(self.config_version) is not int or self.config_version not in (1, 2):
            raise ValueError("anchor_geometry.config_version must be the integer 1 or 2")
        if self.objective != {1: "centered_v1", 2: "base_band_v2"}[self.config_version]:
            raise ValueError("anchor_geometry objective must match config_version")
        for name in (
            "current_directions",
            "base_directions",
            "coverage_directions",
            "base_top_k",
            "position_chunk_size",
            "reference_samples_per_shard",
            "geometry_start_step",
            "calibration_groups",
            "geometry_ramp_steps",
            "band_scale_top_k",
            "band_solver_iterations",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"anchor_geometry.{name} must be a positive integer")
        if self.base_top_k < self.current_directions + self.base_directions:
            raise ValueError("anchor_geometry.base_top_k must cover current + base directions")
        if self.band_scale_top_k > self.base_top_k:
            raise ValueError("anchor_geometry.band_scale_top_k must fit the Base top cache")
        if not 16 <= self.band_solver_iterations <= 96:
            raise ValueError("anchor_geometry.band_solver_iterations must be in [16,96]")
        if (
            isinstance(self.band_frequency_strength, bool)
            or not isinstance(self.band_frequency_strength, int | float)
            or not math.isfinite(self.band_frequency_strength)
            or not 0 <= self.band_frequency_strength < 1
        ):
            raise ValueError("anchor_geometry.band_frequency_strength must be in [0,1)")
        for name in ("coverage_seed", "calibration_seed"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"anchor_geometry.{name} must be a non-negative integer")
        for name in (
            "current_mass",
            "base_mass",
            "coverage_mass",
            "coverage_cap_multiplier",
            "geometry_delta",
            "normal_epsilon",
            "hidden_epsilon",
            "geometry_gradient_ratio",
            "band_scale_ratio",
            "band_temperature",
            "band_min_calibration_rms",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, int | float)
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"anchor_geometry.{name} must be finite and positive")
        if not math.isclose(
            self.current_mass + self.base_mass + self.coverage_mass, 1, rel_tol=0, abs_tol=1e-9
        ):
            raise ValueError("anchor_geometry candidate masses must sum to one")
        if self.coverage_cap_multiplier < 1:
            raise ValueError("anchor_geometry.coverage_cap_multiplier must be at least one")
        if self.reference_cache_path is not None and (
            not isinstance(self.reference_cache_path, str) or not self.reference_cache_path.strip()
        ):
            raise ValueError("anchor_geometry.reference_cache_path must be a non-empty path")
        for name in ("reference_sha256", "calibration_sha256"):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
            ):
                raise ValueError(f"anchor_geometry.{name} must be a lowercase SHA256 or null")
        if (self.resolved_lambda is None) != (self.calibrated_at_step is None):
            raise ValueError("anchor_geometry resolved_lambda and calibrated_at_step must pair")
        if self.resolved_lambda is not None:
            if (
                not isinstance(self.resolved_lambda, int | float)
                or isinstance(self.resolved_lambda, bool)
                or not math.isfinite(self.resolved_lambda)
                or self.resolved_lambda <= 0
            ):
                raise ValueError("anchor_geometry.resolved_lambda must be finite and positive")
            if (
                type(self.calibrated_at_step) is not int
                or self.calibrated_at_step != self.geometry_start_step
            ):
                raise ValueError("anchor_geometry calibration must bind geometry_start_step")


@dataclass
class ReferenceConfig:
    mode: ReferenceMode = ReferenceMode.NONE
    model_id: str | None = None
    revision: str | None = None
    cache_path: str | None = None


@dataclass
class TrainingConfig:
    mixed_precision: str = "fp32"
    optimizer: str = "adamw_torch"
    learning_rate: float = 2e-5
    weight_decay: float = 0.0
    num_train_epochs: float = 1.0
    max_steps: int = -1
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.0
    lr_scheduler_type: str = "linear"
    logging_steps: int = 1
    save_strategy: str = "steps"
    save_steps: int = 100
    save_total_limit: int = 2
    recovery_save_steps: int | None = None
    seed: int = 42
    data_seed: int = 42
    distributed_strategy: str = "single"
    fsdp_sync_each_batch: bool = True
    fsdp_cpu_offload: bool | None = None
    fsdp_keep_low_precision_params: bool = False
    report_to: tuple[str, ...] = ("tensorboard",)
    resume_from_checkpoint: str | None = None


@dataclass
class EvaluationConfig:
    target: DatasetConfig | None = None
    batch_size: int = 1
    run_after_training: bool = False
    eval_checkpoint: str | None = None


@dataclass
class ExperimentConfig:
    model: ModelConfig
    data: DataConfig
    paths: PathsConfig = field(default_factory=PathsConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    method: MethodConfig = field(default_factory=MethodConfig)
    lvd: LVDConfig = field(default_factory=LVDConfig)
    reference: ReferenceConfig = field(default_factory=ReferenceConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    lvd_plus: LVDPlusConfig = field(default_factory=LVDPlusConfig)
    s0: S0Config = field(default_factory=S0Config)
    anchor_geometry: AnchorGeometryConfig = field(default_factory=AnchorGeometryConfig)

    def validate(self) -> None:
        if self.method.name is MethodName.PREPARE_ANCHOR_GEOMETRY_REFERENCE:
            raise ValueError("prepare_anchor_geometry_reference is an action-only selector")
        if self.method.name.is_lvd_plus_preparation:
            raise ValueError(
                f"{self.method.name.value} is an action-only LVD+ preparation selector; "
                "run the corresponding preparation action and normalize method.name to "
                "target_plus_anchor_lvd+ before validating ExperimentConfig"
            )
        record_s0 = self.method.name is MethodName.RECORD_S0
        if not self.model.model_id.strip():
            raise ValueError("model.model_id must not be empty")
        if not self.model.revision.strip() or self.model.revision in {"main", "master", "latest"}:
            raise ValueError("model.revision must be an immutable revision or 'local'")
        if not self.model.local_files_only:
            raise ValueError("model.local_files_only must remain true for training runs")
        if self.model.model_variant not in {"base", "instruct"}:
            raise ValueError("model.model_variant must be 'base' or 'instruct'")

        mode = self.model.training_mode
        if mode is TrainingMode.FULL_FT and self.model.freeze_unembedding:
            raise ValueError("full_ft requires model.freeze_unembedding=false")
        if mode is TrainingMode.FULL_FT and self.model.untie_tied_embeddings is True:
            raise ValueError("full_ft must preserve the model's native tied-weight relationship")
        if mode is TrainingMode.BODY_FT_FROZEN_HEAD and not self.model.freeze_unembedding:
            raise ValueError("body_ft_frozen_head requires model.freeze_unembedding=true")
        if mode is TrainingMode.LORA and not self.model.freeze_unembedding:
            raise ValueError("lora requires model.freeze_unembedding=true in the M1 protocol")
        if mode is TrainingMode.LORA and not self.lora.target_modules:
            raise ValueError("lora.target_modules must contain at least one module name")
        if self.lora.rank <= 0 or self.lora.alpha <= 0:
            raise ValueError("lora rank and alpha must be positive")
        if not 0.0 <= self.lora.dropout < 1.0:
            raise ValueError("lora.dropout must be in [0, 1)")

        if self.data.max_length <= 1:
            raise ValueError("data.max_length must be greater than one")
        if self.data.target_batch_size <= 0 or self.data.anchor_batch_size <= 0:
            raise ValueError("target and anchor batch sizes must be positive")
        if self.data.anchor_batches_per_target < 0:
            raise ValueError("data.anchor_batches_per_target must be non-negative")
        if not isinstance(self.data.anchor_sampling_strategy, AnchorSamplingStrategy):
            raise ValueError("data.anchor_sampling_strategy is invalid")
        if self.data.steps_per_epoch is not None and self.data.steps_per_epoch <= 0:
            raise ValueError("data.steps_per_epoch must be positive when set")
        datasets = [
            self.data.target,
            self.data.validation,
            self.data.anchor,
            self.evaluation.target,
        ]
        for dataset in (value for value in datasets if value is not None):
            if dataset.source not in {"jsonl", "hf"}:
                raise ValueError(f"unsupported dataset source: {dataset.source}")
            if not dataset.path.strip():
                raise ValueError("dataset paths must not be empty")
            if dataset.source == "hf" and (
                not dataset.revision or dataset.revision in {"main", "master", "latest"}
            ):
                raise ValueError(
                    f"Hugging Face dataset {dataset.path} requires an immutable revision"
                )
            if dataset.data_file is not None:
                data_file = PurePosixPath(dataset.data_file)
                if dataset.source != "hf" or data_file.is_absolute() or ".." in data_file.parts:
                    raise ValueError(
                        "dataset.data_file must be a relative file in a Hugging Face dataset"
                    )
            if dataset.selection not in {
                "all",
                "train_90",
                "validation_10",
                "sample_n",
                "infinity_code_sample_n",
            }:
                raise ValueError(
                    "dataset.selection must be all, train_90, validation_10, sample_n, "
                    "or infinity_code_sample_n"
                )
            if not 0.0 < dataset.validation_fraction < 1.0:
                raise ValueError("dataset.validation_fraction must be between zero and one")

        anchor_methods = {
            MethodName.TARGET_PLUS_ANCHOR_CE,
            MethodName.TARGET_PLUS_ANCHOR_KL,
            MethodName.TARGET_PLUS_NULL_LVD,
            MethodName.TARGET_PLUS_ANCHOR_LVD,
            MethodName.TARGET_PLUS_ANCHOR_CE_LVD,
            MethodName.TARGET_PLUS_ANCHOR_LVD_PLUS,
            MethodName.TARGET_PLUS_ANCHOR_CE_GEOMETRY,
            MethodName.FILTERING_ANCHOR,
            MethodName.RECORD_ANCHOR_D0,
            MethodName.TEST_INTERFERENCE,
            MethodName.TEST_CROSSING,
        }
        if self.method.name in anchor_methods and self.data.anchor is None:
            raise ValueError(f"{self.method.name.value} requires data.anchor")
        if self.method.name in anchor_methods and self.data.anchor_batches_per_target <= 0:
            raise ValueError(f"{self.method.name.value} requires a positive anchor batch ratio")
        if self.data.anchor is None:
            self.data.anchor_batches_per_target = 0.0
        if (
            self.method.name is MethodName.TARGET_PLUS_ANCHOR_KL
            and self.reference.mode is not ReferenceMode.ONLINE
        ):
            raise ValueError(
                "target_plus_anchor_kl requires reference.mode=online_reference because "
                "the geometry cache stores reference margins, not the full KL distribution"
            )
        if (self.method.name.uses_lvd_plus or record_s0) and (
            self.reference.mode is not ReferenceMode.NONE
            or self.reference.model_id is not None
            or self.reference.revision is not None
            or self.reference.cache_path is not None
        ):
            raise ValueError(
                f"{self.method.name.value} requires reference.mode=none and no reference "
                "model_id, revision, or cache_path"
            )
        if self.reference.mode is ReferenceMode.CACHED:
            if self.data.anchor is None:
                raise ValueError("cached_reference requires data.anchor")
            if not self.reference.cache_path:
                raise ValueError("cached_reference requires reference.cache_path")
        if self.method.name is MethodName.TARGET_PLUS_NULL_LVD and self.method.lvd_weight != 0.0:
            raise ValueError("the M1 null LVD plugin requires method.lvd_weight=0")
        if self.method.anchor_weight < 0 or self.method.lvd_weight < 0:
            raise ValueError("method loss weights must be non-negative")
        if self.method.kl_temperature <= 0:
            raise ValueError("method.kl_temperature must be positive")
        if self.method.name.uses_dynamic_geometry:
            self.anchor_geometry.validate()
            if mode is not TrainingMode.FULL_FT or self.model.freeze_unembedding:
                raise ValueError("target_plus_anchor_ce_geometry requires full_ft and live head")
            if not math.isfinite(self.method.anchor_weight) or self.method.anchor_weight <= 0:
                raise ValueError("target_plus_anchor_ce_geometry requires positive anchor_weight")
            if self.reference.mode is not ReferenceMode.NONE or any(
                value is not None
                for value in (
                    self.reference.model_id,
                    self.reference.revision,
                    self.reference.cache_path,
                    self.lvd.d0_cache_path,
                    self.lvd.calibration_checkpoint,
                )
            ):
                raise ValueError("dynamic geometry forbids legacy reference/d0/pilot settings")
            if self.anchor_geometry.reference_cache_path is None:
                raise ValueError("dynamic geometry requires anchor_geometry.reference_cache_path")
        if self.method.name.uses_lvd_plus or record_s0:
            if record_s0:
                # Recording consumes only the frozen head, S0 options and cache
                # binding; training geometry and lambda calibration are inactive.
                self.lvd_plus._validate_anchor_mode()
                if self.lvd_plus.anchor_mode is not LVDPlusAnchorMode.RECORDED_S0:
                    raise ValueError("record_s0 requires lvd_plus.anchor_mode=recorded_s0")
            else:
                self.lvd_plus.validate()
                if self.lvd_plus.uses_boundary_risk and (
                    self.lvd_plus.anchor_mode is not LVDPlusAnchorMode.RECORDED_S0
                ):
                    raise ValueError(
                        "lvd_plus.objective_mode=pressure_plus_risk requires "
                        "lvd_plus.anchor_mode=recorded_s0 in ExperimentConfig"
                    )
            if mode not in {TrainingMode.BODY_FT_FROZEN_HEAD, TrainingMode.LORA} or (
                not self.model.freeze_unembedding
            ):
                raise ValueError(f"{self.method.name.value} requires a frozen-head or LoRA mode")
            if self.lvd.d0_cache_path is not None:
                raise ValueError(f"{self.method.name.value} forbids lvd.d0_cache_path")
            if self.lvd.calibration_checkpoint is not None:
                raise ValueError(
                    f"{self.method.name.value} forbids lvd.calibration_checkpoint; "
                    "LVD+ and S0 initialize at Base without a pilot"
                )
            if self.lvd_plus.anchor_mode is LVDPlusAnchorMode.RECORDED_S0:
                self.s0.validate()
                if self.lvd_plus.s0_cache_path is None:
                    raise ValueError("recorded_s0 requires a resolved lvd_plus.s0_cache_path")
                self.lvd_plus.s0_cache_path = str(
                    Path(self.lvd_plus.s0_cache_path).expanduser().resolve()
                )
        else:
            self.lvd.validate()
        if self.method.name.uses_lvd:
            if not self.model.freeze_unembedding:
                raise ValueError("target_plus_anchor_lvd requires a frozen Unembedding")
            if self.reference.mode is not ReferenceMode.CACHED:
                raise ValueError("target_plus_anchor_lvd requires reference.mode=cached_reference")
            if not self.lvd.d0_cache_path:
                raise ValueError("target_plus_anchor_lvd requires lvd.d0_cache_path")
            if (
                self.lvd.lambda_mode is LVDLambdaMode.GRADIENT_RATIO
                and self.lvd.resolved_lambda is None
                and not self.lvd.calibration_checkpoint
                and not self.training.resume_from_checkpoint
                and self.lvd.calibration_seed is None
            ):
                raise ValueError(
                    "gradient-ratio LVD lambda requires lvd.calibration_checkpoint from a "
                    "target_only pilot, or a previously resolved lambda"
                )

        if self.training.mixed_precision not in {"bf16", "fp16", "fp32"}:
            raise ValueError("training.mixed_precision must be bf16, fp16, or fp32")
        if self.training.optimizer not in {"adamw_torch", "adamw_bnb_8bit", "adafactor"}:
            raise ValueError("training.optimizer must be adamw_torch, adamw_bnb_8bit, or adafactor")
        if self.training.learning_rate <= 0:
            raise ValueError("training.learning_rate must be positive")
        if self.training.weight_decay < 0:
            raise ValueError("training.weight_decay must be non-negative")
        if self.training.num_train_epochs <= 0:
            raise ValueError("training.num_train_epochs must be positive")
        if self.training.max_steps == 0 or self.training.max_steps < -1:
            raise ValueError("training.max_steps must be -1 or positive")
        if not 0.0 <= self.training.warmup_ratio < 1.0:
            raise ValueError("training.warmup_ratio must be in [0, 1)")
        if self.training.gradient_accumulation_steps <= 0:
            raise ValueError("training.gradient_accumulation_steps must be positive")
        recovery_steps = self.training.recovery_save_steps
        if recovery_steps is not None and (type(recovery_steps) is not int or recovery_steps <= 0):
            raise ValueError("training.recovery_save_steps must be a positive integer or None")
        if self.training.distributed_strategy not in {"single", "ddp", "fsdp"}:
            raise ValueError("training.distributed_strategy must be single, ddp, or fsdp")
        if (
            self.training.fsdp_cpu_offload is not None
            and type(self.training.fsdp_cpu_offload) is not bool
        ):
            raise ValueError("training.fsdp_cpu_offload must be True, False, or None")
        if (
            self.training.distributed_strategy == "fsdp"
            and self.training.optimizer == "adamw_bnb_8bit"
        ):
            raise ValueError(
                "adamw_bnb_8bit is unsupported with FSDP because bitsandbytes "
                "optimizers do not support FSDP"
            )
        if self.training.fsdp_keep_low_precision_params:
            if self.training.distributed_strategy != "fsdp":
                raise ValueError("fsdp_keep_low_precision_params requires FSDP")
            if self.training.mixed_precision != "bf16":
                raise ValueError("fsdp_keep_low_precision_params requires mixed_precision=bf16")
        if self.evaluation.batch_size <= 0:
            raise ValueError("evaluation.batch_size must be positive")
        if self.training.resume_from_checkpoint and self.evaluation.eval_checkpoint:
            raise ValueError(
                "resume_from_checkpoint and eval_checkpoint are independent modes "
                "and cannot be combined"
            )
        if self.paths.estimated_run_bytes < 0 or self.paths.minimum_free_bytes < 0:
            raise ValueError("path capacity values must be non-negative")
        if self.paths.free_space_multiplier < 1.0:
            raise ValueError("paths.free_space_multiplier must be at least one")

        if self.model.tokenizer_revision is None:
            self.model.tokenizer_revision = self.model.revision
        if self.reference.mode is ReferenceMode.ONLINE:
            if self.reference.model_id is None:
                self.reference.model_id = self.model.model_id
            if self.reference.revision is None:
                self.reference.revision = self.model.revision
