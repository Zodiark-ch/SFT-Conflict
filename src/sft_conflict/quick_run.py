"""Unified click-run presets for Tiny and cached real experiments."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Sequence, Split, WhitespaceSplit
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

from sft_conflict.config import (
    AnchorGeometryConfig,
    AnchorSamplingStrategy,
    DataConfig,
    DatasetConfig,
    EvaluationConfig,
    ExperimentConfig,
    LoraConfig,
    LVDConfig,
    LVDPlusAnchorMode,
    LVDPlusConfig,
    MethodConfig,
    MethodName,
    ModelConfig,
    PathsConfig,
    ReferenceConfig,
    ReferenceMode,
    S0Config,
    TrainingConfig,
    TrainingMode,
    load_anchor_geometry_config,
    load_lvd_config,
    load_lvd_plus_config,
    load_s0_config,
)
from sft_conflict.data.anchor_creation import (
    created_anchor_dataset_dir,
    created_anchor_dataset_name,
    load_created_anchor_manifest,
)
from sft_conflict.data.anchor_filtering import (
    filtered_anchor_dataset_dir,
    filtered_anchor_dataset_name,
)
from sft_conflict.data.glue import GLUE_TASK_SPECS
from sft_conflict.data.label_coverage import (
    LABEL_COVERAGE_POLICY,
    LEGACY_FILTER_POLICY,
    label_coverage_dataset_dir,
    label_coverage_dataset_name,
    load_label_coverage_manifest,
)
from sft_conflict.data.registry import DEFAULT_DATASET_REGISTRY, DatasetRole
from sft_conflict.data.s0_cache import S0_CACHE_CONTRACT_VERSION
from sft_conflict.paths import DEFAULT_ARTIFACT_ROOT


@dataclass(frozen=True)
class ModelPreset:
    model_id: str | None
    revision: str
    estimated_run_bytes: int
    minimum_precision: str


@dataclass(frozen=True)
class DatasetPreset:
    path: str | None
    revision: str | None
    name: str | None
    train_split: str
    eval_split: str
    adapter: str
    train_data_file: str | None = None
    validation_split: str | None = None
    validation_data_file: str | None = None
    eval_data_file: str | None = None


MODEL_PRESETS = {
    "tiny_qwen2": ModelPreset(
        model_id=None,
        revision="local",
        estimated_run_bytes=100 * 1024**2,
        minimum_precision="fp32",
    ),
    "qwen25_0.5b": ModelPreset(
        model_id="Qwen/Qwen2.5-0.5B",
        revision="060db6499f32faf8b98477b0a26969ef7d8b9987",
        estimated_run_bytes=20 * 1024**3,
        minimum_precision="bf16",
    ),
    "qwen25_3b": ModelPreset(
        model_id="Qwen/Qwen2.5-3B",
        revision="3aab1f1954e9cc14eb9509a215f9e5ca08227a9b",
        estimated_run_bytes=30 * 1024**3,
        minimum_precision="bf16",
    ),
    "qwen25_7b": ModelPreset(
        model_id="Qwen/Qwen2.5-7B",
        revision="d149729398750b98c0af14eb82c78cfe92750796",
        estimated_run_bytes=40 * 1024**3,
        minimum_precision="bf16",
    ),
    "mistral_7b": ModelPreset(
        model_id="mistralai/Mistral-7B-v0.3",
        revision="caa1feb0e54d415e2df31207e5f4e273e33509b1",
        estimated_run_bytes=40 * 1024**3,
        minimum_precision="bf16",
    ),
}


def _registry_preset(
    key: str,
    train_role: DatasetRole,
    eval_role: DatasetRole | None = None,
) -> DatasetPreset:
    spec = DEFAULT_DATASET_REGISTRY.dataset(key)
    train_split = spec.role_splits[train_role]
    eval_split = spec.role_splits[eval_role] if eval_role is not None else train_split
    return DatasetPreset(
        path=spec.dataset_id,
        revision=spec.revision,
        name=spec.config_name,
        train_split=train_split,
        eval_split=eval_split,
        adapter=spec.adapter,
        train_data_file=spec.role_data_files.get(train_role),
        validation_split=spec.role_splits.get(DatasetRole.TARGET_VALIDATION),
        validation_data_file=spec.role_data_files.get(DatasetRole.TARGET_VALIDATION),
        eval_data_file=(
            spec.role_data_files.get(eval_role)
            if eval_role is not None
            else spec.role_data_files.get(train_role)
        ),
    )


TARGET_DATASET_PRESETS = {
    "tiny_gsm8k": DatasetPreset(None, None, None, "train", "test", "prompt_response"),
    "tiny_sst2": DatasetPreset(None, None, None, "train", "validation", "prompt_response"),
    "tiny_instruction": DatasetPreset(None, None, None, "train", "validation", "prompt_response"),
    "gsm8k": _registry_preset("gsm8k", DatasetRole.TARGET_TRAIN, DatasetRole.TARGET_EVALUATION),
    "codealpaca": _registry_preset("codealpaca", DatasetRole.TARGET_TRAIN),
    "code_bagel_hermes": _registry_preset("code_bagel_hermes", DatasetRole.TARGET_TRAIN),
    "infinity_instruct_code": _registry_preset("infinity_instruct_code", DatasetRole.TARGET_TRAIN),
    "evol_instruct": _registry_preset("evol_instruct", DatasetRole.TARGET_TRAIN),
    "alpaca_eval": _registry_preset(
        "alpaca_eval", DatasetRole.TARGET_EVALUATION, DatasetRole.TARGET_EVALUATION
    ),
    "logiqa2": _registry_preset("logiqa2", DatasetRole.TARGET_TRAIN, DatasetRole.TARGET_EVALUATION),
    "humaneval": _registry_preset(
        "humaneval", DatasetRole.TARGET_EVALUATION, DatasetRole.TARGET_EVALUATION
    ),
    "codexglue_python": _registry_preset(
        "codexglue_python", DatasetRole.TARGET_EVALUATION, DatasetRole.TARGET_EVALUATION
    ),
    "codexglue_java": _registry_preset(
        "codexglue_java", DatasetRole.TARGET_EVALUATION, DatasetRole.TARGET_EVALUATION
    ),
    **{
        task_name: _registry_preset(
            task_name,
            DatasetRole.TARGET_TRAIN,
            DatasetRole.TARGET_EVALUATION,
        )
        for task_name in GLUE_TASK_SPECS
    },
}

CODE_TARGET_EVALUATION_KEYS = {
    "codealpaca": "humaneval",
    "code_bagel_hermes": "codexglue_python",
    "infinity_instruct_code": "codexglue_python",
}

LOGIC_TARGET_EVALUATION_KEYS = {
    "evol_instruct": "alpaca_eval",
}

TARGET_EVALUATION_KEYS = {
    **CODE_TARGET_EVALUATION_KEYS,
    **LOGIC_TARGET_EVALUATION_KEYS,
}

CODE_TARGET_SAMPLE_BUDGETS = {
    "code_bagel_hermes": 20_000,
    "infinity_instruct_code": 20_000,
}


def target_evaluation_preset(target_dataset: str) -> DatasetPreset:
    evaluation_key = TARGET_EVALUATION_KEYS.get(target_dataset)
    if evaluation_key is None:
        return TARGET_DATASET_PRESETS[target_dataset]
    return TARGET_DATASET_PRESETS[evaluation_key]


ANCHOR_DATASET_PRESETS = {
    "none": None,
    "tiny_general": DatasetPreset(None, None, None, "train", "train", "prompt_response"),
    "smart_flan": _registry_preset("smart_flan", DatasetRole.ANCHOR_TRAIN),
}
FILTERED_ANCHOR_DATASETS = tuple(
    filtered_anchor_dataset_name(model_name) for model_name in MODEL_PRESETS
)
LABEL_COVERAGE_ANCHOR_DATASETS = tuple(
    label_coverage_dataset_name(model_name) for model_name in MODEL_PRESETS
)
CREATED_ANCHOR_DATASETS = tuple(
    created_anchor_dataset_name(model_name) for model_name in MODEL_PRESETS
)

EVALUATION_DATASET_PRESETS = {
    "hellaswag": _registry_preset("hellaswag", DatasetRole.GENERAL_EVALUATION)
}

RUN_MODELS = tuple(MODEL_PRESETS)
RUN_TARGET_DATASETS = tuple(TARGET_DATASET_PRESETS)
RUN_ANCHOR_DATASETS = (
    *ANCHOR_DATASET_PRESETS,
    *FILTERED_ANCHOR_DATASETS,
    *LABEL_COVERAGE_ANCHOR_DATASETS,
    *CREATED_ANCHOR_DATASETS,
)
RUN_SUPERVISION_MODES = ("response_only",)
DEFAULT_LVD_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "method" / "weighted_pressure_lvd.yaml"
)
DEFAULT_LVD_PLUS_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "method" / "lvd_plus.yaml"
)
DEFAULT_S0_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "method" / "s0.yaml"
DEFAULT_ANCHOR_GEOMETRY_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "method" / "anchor_ce_geometry.yaml"
)

_LVD_PLUS_PREPARATION_ACTIONS = {
    MethodName.RECORD_LVD_PLUS_SCALES: "record-lvd-plus-scales",
    MethodName.CALIBRATE_LVD_PLUS_LAMBDA: "calibrate-lvd-plus-lambda",
}

# Backwards-compatible internal aliases for older imports.
QUICK_MODELS = RUN_MODELS
QUICK_TARGET_DATASETS = RUN_TARGET_DATASETS
QUICK_ANCHOR_DATASETS = RUN_ANCHOR_DATASETS
QUICK_SUPERVISION_MODES = RUN_SUPERVISION_MODES


@dataclass(frozen=True)
class QuickRunSettings:
    action: str = "train"
    model: str = "tiny_qwen2"
    target_dataset: str = "tiny_gsm8k"
    anchor_dataset: str = "none"
    method: str = MethodName.TARGET_ONLY.value
    supervision: str = "response_only"
    training_mode: str = TrainingMode.BODY_FT_FROZEN_HEAD.value
    distributed_strategy: str = "single"
    mixed_precision: str = "fp32"
    optimizer: str = "adamw_torch"
    max_steps: int = 2
    num_train_epochs: float = 1.0
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    warmup_ratio: float = 0.0
    max_length: int = 64
    target_max_samples: int | None = 256
    target_validation_max_samples: int | None = None
    anchor_max_samples: int | None = 256
    anchor_metadata_path: str | None = None
    evaluation_max_samples: int | None = 256
    evaluation_batch_size: int = 1
    target_batch_size: int = 2
    anchor_batch_size: int = 2
    anchor_sampling_strategy: str = AnchorSamplingStrategy.TASK_BALANCED.value
    gradient_accumulation_steps: int = 1
    recovery_save_steps: int | None = None
    fsdp_sync_each_batch: bool = True
    fsdp_cpu_offload: bool | None = None
    fsdp_keep_low_precision_params: bool = False
    gradient_checkpointing: bool = False
    run_after_training: bool = True
    training_validation: bool = True
    lora_rank: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    anchor_weight: float = 0.2
    kl_temperature: float = 1.0
    seed: int = 42
    artifact_root: str = DEFAULT_ARTIFACT_ROOT
    run_name: str | None = None
    lvd_config_path: str = str(DEFAULT_LVD_CONFIG_PATH)
    lvd_direction: int | None = None
    lvd_position_strategy: str | None = None
    lvd_positions_per_sample: int | None = None
    anchor_d0_cache_path: str | None = None
    lvd_calibration_checkpoint: str | None = None
    lvd_use_capacity_weights: bool | None = None
    lvd_plus_config_path: str = str(DEFAULT_LVD_PLUS_CONFIG_PATH)
    lvd_plus_direction: int | None = None
    filtering_policy: str = LABEL_COVERAGE_POLICY
    s0_config_path: str = str(DEFAULT_S0_CONFIG_PATH)
    s0_cache_path: str | None = None
    lvd_plus_scale_cache_path: str | None = None
    lvd_plus_lambda_cache_path: str | None = None
    anchor_geometry_config_path: str = str(DEFAULT_ANCHOR_GEOMETRY_CONFIG_PATH)
    anchor_geometry_reference_path: str | None = None
    # Appended last: positional construction of older settings must keep working.
    best_model_metric: str = "eval_loss"

    def validate(self) -> None:
        if (
            self.lvd_use_capacity_weights is not None
            and type(self.lvd_use_capacity_weights) is not bool
        ):
            raise ValueError("lvd_use_capacity_weights must be True, False, or None")
        if self.action not in {
            "train",
            "generate-pilot",
            "audit-model",
            "evaluate",
            "merge-lora",
            "build-reference-cache",
            "record-anchor-d0",
            "native-evaluate",
            "audit-data",
            "benchmark",
            "test-interference",
            "test-crossing",
            "filtering-anchor",
            "create-anchor-dataset",
            "record-s0",
            "prepare-anchor-geometry-reference",
            *_LVD_PLUS_PREPARATION_ACTIONS.values(),
        }:
            raise ValueError(f"unsupported preset action: {self.action}")
        if self.model not in RUN_MODELS:
            raise ValueError(f"model must be one of {RUN_MODELS}")
        method = MethodName(self.method)
        record_s0 = method is MethodName.RECORD_S0
        lvd_plus_preparation = method.is_lvd_plus_preparation
        uses_lvd_plus = method.uses_lvd_plus or lvd_plus_preparation
        prepare_geometry = method is MethodName.PREPARE_ANCHOR_GEOMETRY_REFERENCE
        dynamic_geometry = method.uses_dynamic_geometry or prepare_geometry
        if prepare_geometry != (self.action == "prepare-anchor-geometry-reference"):
            raise ValueError("prepare_anchor_geometry_reference requires its preparation action")
        for action_method, action in _LVD_PLUS_PREPARATION_ACTIONS.items():
            if method is action_method and self.action != action:
                raise ValueError(f"{action_method.value} requires action={action}")
            if self.action == action and method is not action_method:
                raise ValueError(f"action={action} requires method={action_method.value}")
        if record_s0 and self.action != "record-s0":
            raise ValueError("record_s0 requires action=record-s0")
        if self.action == "record-s0" and not record_s0:
            raise ValueError("action=record-s0 requires method=record_s0")
        if not record_s0 and self.target_dataset not in RUN_TARGET_DATASETS:
            raise ValueError(f"target dataset must be one of {RUN_TARGET_DATASETS}")
        if not record_s0 and self.anchor_dataset not in RUN_ANCHOR_DATASETS:
            raise ValueError(f"anchor dataset must be one of {RUN_ANCHOR_DATASETS}")
        if self.supervision not in RUN_SUPERVISION_MODES:
            raise ValueError(f"supervision must be one of {RUN_SUPERVISION_MODES}")
        training_mode = TrainingMode(self.training_mode)
        if dynamic_geometry:
            if training_mode is not TrainingMode.FULL_FT:
                raise ValueError("dynamic geometry requires training_mode=full_ft")
            for name in (
                "anchor_d0_cache_path",
                "lvd_calibration_checkpoint",
                "lvd_direction",
                "lvd_position_strategy",
                "lvd_positions_per_sample",
                "lvd_use_capacity_weights",
                "s0_cache_path",
                "lvd_plus_direction",
                "lvd_plus_scale_cache_path",
                "lvd_plus_lambda_cache_path",
            ):
                if getattr(self, name) is not None:
                    raise ValueError(f"dynamic geometry forbids legacy setting {name}")
            if self.action not in {"train", "prepare-anchor-geometry-reference"}:
                raise ValueError("dynamic geometry uses prepare-reference or train actions")
        if self.anchor_geometry_reference_path is not None and (
            not dynamic_geometry
            or not isinstance(self.anchor_geometry_reference_path, str)
            or not self.anchor_geometry_reference_path.strip()
        ):
            raise ValueError("anchor_geometry_reference_path requires dynamic geometry")
        if uses_lvd_plus:
            if self.action == "generate-pilot":
                raise ValueError(
                    "generate-pilot requires a legacy LVD method; "
                    "LVD+ initializes at Base without a pilot"
                )
            if self.action in {"build-reference-cache", "record-anchor-d0"}:
                raise ValueError("LVD+ does not use a reference/d0 cache")
        if uses_lvd_plus or record_s0:
            for name in (
                "anchor_d0_cache_path",
                "lvd_calibration_checkpoint",
                "lvd_direction",
                "lvd_position_strategy",
                "lvd_positions_per_sample",
                "lvd_use_capacity_weights",
            ):
                if getattr(self, name) is not None:
                    raise ValueError(
                        f"{name} is a legacy LVD override and is not valid for {method.value}"
                    )
        if self.s0_cache_path is not None:
            if not isinstance(self.s0_cache_path, str) or not self.s0_cache_path.strip():
                raise ValueError("s0_cache_path must be a non-empty string or None")
            if not (uses_lvd_plus or record_s0):
                raise ValueError("s0_cache_path requires record_s0 or recorded-S0 LVD+")
        for name in ("lvd_plus_scale_cache_path", "lvd_plus_lambda_cache_path"):
            value = getattr(self, name)
            if value is not None:
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"{name} must be a non-empty string or None")
                if not uses_lvd_plus:
                    raise ValueError(f"{name} requires hybrid LVD+ or an LVD+ preparation action")
        if self.lvd_plus_direction is not None:
            if type(self.lvd_plus_direction) is not int or self.lvd_plus_direction <= 0:
                raise ValueError("lvd_plus_direction must be a positive integer or None")
            if not uses_lvd_plus:
                raise ValueError(
                    "lvd_plus_direction requires method=target_plus_anchor_lvd+ "
                    "or an LVD+ preparation action"
                )
        if self.action == "generate-pilot" and not method.uses_lvd:
            raise ValueError("generate-pilot requires a formal LVD method")
        if self.distributed_strategy not in {"single", "ddp", "fsdp"}:
            raise ValueError("distributed_strategy must be single, ddp, or fsdp")
        if method not in {MethodName.TARGET_ONLY, MethodName.RECORD_S0} and (
            self.anchor_dataset == "none"
        ):
            raise ValueError(f"{method.value} requires anchor_dataset=tiny_general or smart_flan")
        if method is MethodName.FILTERING_ANCHOR:
            if self.action != "filtering-anchor":
                raise ValueError("filtering_anchor requires action=filtering-anchor")
            if self.anchor_dataset != "smart_flan":
                raise ValueError("filtering_anchor requires the original anchor_dataset=smart_flan")
            if not isinstance(self.filtering_policy, str) or self.filtering_policy not in {
                LABEL_COVERAGE_POLICY,
                LEGACY_FILTER_POLICY,
            }:
                raise ValueError("filtering_policy must be label_coverage_v1 or legacy_quality_v1")
        if self.action == "filtering-anchor" and method is not MethodName.FILTERING_ANCHOR:
            raise ValueError("action=filtering-anchor requires method=filtering_anchor")
        if method is MethodName.CREATE_ANCHOR_DATASET:
            if self.action != "create-anchor-dataset":
                raise ValueError("create_anchor_dataset requires action=create-anchor-dataset")
            if self.anchor_dataset != "smart_flan":
                raise ValueError(
                    "create_anchor_dataset reads the fixed SMART FLAN 2022 submixtures; "
                    "keep anchor_dataset=smart_flan as the placeholder"
                )
        if (
            self.action == "create-anchor-dataset"
            and method is not MethodName.CREATE_ANCHOR_DATASET
        ):
            raise ValueError("action=create-anchor-dataset requires method=create_anchor_dataset")
        label_coverage_anchor = self.anchor_dataset in LABEL_COVERAGE_ANCHOR_DATASETS
        created_anchor = self.anchor_dataset in CREATED_ANCHOR_DATASETS
        expected_filtered_anchor = (
            label_coverage_dataset_name(self.model)
            if label_coverage_anchor
            else created_anchor_dataset_name(self.model)
            if created_anchor
            else filtered_anchor_dataset_name(self.model)
        )
        if (
            not record_s0
            and (
                label_coverage_anchor
                or created_anchor
                or self.anchor_dataset in FILTERED_ANCHOR_DATASETS
            )
            and self.anchor_dataset != expected_filtered_anchor
        ):
            raise ValueError(
                f"model={self.model} requires filtered Anchor {expected_filtered_anchor}, "
                f"not {self.anchor_dataset}"
            )
        if (
            self.action in {"train", "generate-pilot"}
            and method is MethodName.TARGET_ONLY
            and self.anchor_dataset != "none"
        ):
            raise ValueError("target_only requires anchor_dataset=none")
        if self.action in {"build-reference-cache", "record-anchor-d0"} and (
            self.anchor_dataset == "none"
        ):
            raise ValueError(f"{self.action} requires an Anchor dataset")
        if self.action == "record-anchor-d0" and method is not MethodName.RECORD_ANCHOR_D0:
            raise ValueError("record-anchor-d0 requires method=record_anchor_d0")
        diagnostic_actions = {
            MethodName.TEST_INTERFERENCE: "test-interference",
            MethodName.TEST_CROSSING: "test-crossing",
        }
        if method in diagnostic_actions and self.action != diagnostic_actions[method]:
            raise ValueError(f"{method.value} requires action={diagnostic_actions[method]}")
        action_methods = {action: name for name, action in diagnostic_actions.items()}
        if self.action in action_methods and method is not action_methods[self.action]:
            raise ValueError(
                f"action={self.action} requires method={action_methods[self.action].value}"
            )
        if (
            method.uses_lvd_geometry or lvd_plus_preparation or record_s0
        ) and training_mode is TrainingMode.FULL_FT:
            raise ValueError(f"{method.value} requires a frozen-head or LoRA training mode")
        native_tasks = {
            "gsm8k",
            "logiqa2",
            "alpaca_eval",
            "humaneval",
            "codexglue_python",
            "codexglue_java",
            *GLUE_TASK_SPECS,
        }
        if self.action == "native-evaluate" and self.target_dataset not in native_tasks:
            raise ValueError("native-evaluate requires a registered native evaluation dataset")
        if (
            self.action in {"train", "generate-pilot", *_LVD_PLUS_PREPARATION_ACTIONS.values()}
            and method is not MethodName.RECORD_ANCHOR_D0
            and self.model in {"qwen25_3b", "qwen25_7b", "mistral_7b"}
            and training_mode is not TrainingMode.LORA
            and self.distributed_strategy != "fsdp"
        ):
            raise ValueError(
                f"{self.model} with {training_mode.value} requires distributed_strategy=fsdp"
            )
        if (
            self.action in {"train", "generate-pilot"}
            and self.model in {"qwen25_7b", "mistral_7b"}
            and method is MethodName.TARGET_PLUS_ANCHOR_KL
        ):
            raise ValueError(
                f"{self.model} online Anchor KL requires a second 7B reference model and is not "
                "supported by this 4x40GB workflow; use Anchor CE or a smaller model"
            )
        if self.optimizer not in {"adamw_torch", "adamw_bnb_8bit", "adafactor"}:
            raise ValueError("unsupported optimizer")
        if self.max_steps == 0 or self.max_steps < -1:
            raise ValueError("max_steps must be -1 or positive")
        if self.num_train_epochs <= 0:
            raise ValueError("num_train_epochs must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if not 0.0 <= self.warmup_ratio < 1.0:
            raise ValueError("warmup_ratio must be in [0, 1)")
        if self.max_length <= 1:
            raise ValueError("max_length must be greater than one")
        if (
            self.target_batch_size <= 0
            or self.anchor_batch_size <= 0
            or self.evaluation_batch_size <= 0
        ):
            raise ValueError("batch sizes must be positive")
        AnchorSamplingStrategy(self.anchor_sampling_strategy)
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.lora_rank <= 0 or self.lora_alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive")
        if not 0.0 <= self.lora_dropout < 1.0:
            raise ValueError("LoRA dropout must be in [0, 1)")
        if lvd_plus_preparation or any(
            value is not None
            for value in (self.lvd_plus_scale_cache_path, self.lvd_plus_lambda_cache_path)
        ):
            plus = load_lvd_plus_config(self.lvd_plus_config_path)
            if not plus.uses_boundary_risk:
                raise ValueError(
                    "LVD+ preparation actions and scale/lambda cache overrides require "
                    "lvd_plus.objective_mode=pressure_plus_risk"
                )


def s0_cache_path_for_settings(
    settings: QuickRunSettings, s0_config: S0Config | None = None
) -> Path:
    """Resolve an override or model/revision/protocol/S0 cache path without creating it.

    With no supplied S0Config, only the S0 YAML is read. The digest is SHA-256
    of sorted, compact JSON containing the cache contract version and
    asdict(S0Config), truncated to 16 hex digits. A changed acceptance protocol
    therefore selects a new immutable location, even with identical solver settings.
    Dataset choices, sequence length, training seed and LVD+ geometry are absent.
    This is a location key, NOT the verified cache digest used for resume binding.
    """
    config = load_s0_config(settings.s0_config_path) if s0_config is None else s0_config
    config.validate()
    if settings.s0_cache_path is not None:
        if not isinstance(settings.s0_cache_path, str) or not settings.s0_cache_path.strip():
            raise ValueError("s0_cache_path must be a non-empty string or None")
        return Path(settings.s0_cache_path).expanduser().resolve()
    artifact_root = Path(settings.artifact_root).expanduser().resolve()
    preset = MODEL_PRESETS[settings.model]
    model_id = preset.model_id or str(
        artifact_root / "tmp" / "click-run" / "assets-v3" / settings.model
    )
    model_slug = re.sub(r"[^a-z0-9]+", "-", model_id.lower()).strip("-")
    config_digest = hashlib.sha256(
        json.dumps(
            {"cache_contract_version": S0_CACHE_CONTRACT_VERSION, "solver_config": asdict(config)},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()[:16]
    return artifact_root / "s0_cache" / f"{model_slug}-{preset.revision[:12]}-{config_digest}"


def _vocabulary() -> dict[str, int]:
    tokens = (
        "<unk>",
        "<pad>",
        "<eos>",
        "Question:",
        "Answer:",
        "Sentence:",
        "Sentiment:",
        "Instruction:",
        "Response:",
        "Translate:",
        "French:",
        "Repeat:",
        "+",
        "-",
        "=",
        ".",
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "good",
        "bad",
        "great",
        "awful",
        "movie",
        "story",
        "positive",
        "negative",
        "red",
        "blue",
        "rouge",
        "bleu",
        "hello",
        "world",
        "yes",
        "no",
    )
    return {token: index for index, token in enumerate(tokens)}


def _ensure_tiny_model(model_path: Path) -> None:
    required_files = ("config.json", "model.safetensors", "tokenizer.json")
    if all((model_path / file_name).is_file() for file_name in required_files):
        return
    model_path.mkdir(parents=True, exist_ok=True)
    vocabulary = _vocabulary()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = Qwen2ForCausalLM(
            Qwen2Config(
                vocab_size=len(vocabulary),
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_position_embeddings=128,
                pad_token_id=vocabulary["<pad>"],
                eos_token_id=vocabulary["<eos>"],
                tie_word_embeddings=False,
            )
        )
    model.save_pretrained(model_path, safe_serialization=True)

    backend = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    backend.pre_tokenizer = Sequence([WhitespaceSplit(), Split(".", behavior="isolated")])
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        pad_token="<pad>",
        eos_token="<eos>",
        model_max_length=128,
    )
    tokenizer.save_pretrained(model_path)


def _target_rows(dataset_name: str) -> list[dict[str, str]]:
    if dataset_name == "tiny_gsm8k":
        examples = (
            ("Question: 1 + 1\nAnswer:", " 2"),
            ("Question: 2 + 2\nAnswer:", " 4"),
            ("Question: 3 + 2\nAnswer:", " 5"),
            ("Question: 4 - 1\nAnswer:", " 3"),
            ("Question: 5 + 1\nAnswer:", " 6"),
            ("Question: 7 - 2\nAnswer:", " 5"),
            ("Question: 3 + 4\nAnswer:", " 7"),
            ("Question: 9 - 1\nAnswer:", " 8"),
        )
    elif dataset_name == "tiny_sst2":
        examples = (
            ("Sentence: good movie\nSentiment:", " positive"),
            ("Sentence: bad movie\nSentiment:", " negative"),
            ("Sentence: great story\nSentiment:", " positive"),
            ("Sentence: awful story\nSentiment:", " negative"),
        ) * 2
    elif dataset_name == "tiny_instruction":
        examples = (
            ("Instruction: Repeat: hello\nResponse:", " hello"),
            ("Instruction: Repeat: world\nResponse:", " world"),
            ("Instruction: Answer: yes\nResponse:", " yes"),
            ("Instruction: Answer: no\nResponse:", " no"),
        ) * 2
    else:
        raise ValueError(f"unsupported quick target dataset: {dataset_name}")
    return [
        {"sample_id": f"{dataset_name}-{index}", "prompt": prompt, "response": response}
        for index, (prompt, response) in enumerate(examples)
    ]


def _anchor_rows() -> list[dict[str, str]]:
    examples = (
        ("Translate: red\nFrench:", " rouge"),
        ("Translate: blue\nFrench:", " bleu"),
        ("Repeat: hello\nResponse:", " hello"),
        ("Repeat: world\nResponse:", " world"),
        ("Question: 1 + 2\nAnswer:", " 3"),
        ("Question: 2 + 3\nAnswer:", " 5"),
        ("Sentence: good story\nSentiment:", " positive"),
        ("Sentence: bad story\nSentiment:", " negative"),
    )
    return [
        {"sample_id": f"tiny-general-{index}", "prompt": prompt, "response": response}
        for index, (prompt, response) in enumerate(examples)
    ]


def _write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    contents = "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows)
    path.write_text(contents, encoding="utf-8")


def _real_dataset_config(
    preset: DatasetPreset,
    *,
    split: str,
    max_samples: int | None,
) -> DatasetConfig:
    if preset.path is None or preset.revision is None:
        raise ValueError("a real dataset preset requires a path and revision")
    return DatasetConfig(
        source="hf",
        path=preset.path,
        name=preset.name,
        data_file=(
            preset.train_data_file if split == preset.train_split else preset.eval_data_file
        ),
        split=split,
        revision=preset.revision,
        adapter=preset.adapter,
        max_samples=max_samples,
    )


def build_quick_run_config(settings: QuickRunSettings) -> ExperimentConfig:
    settings.validate()
    requested_method = MethodName(settings.method)
    method = (
        MethodName.TARGET_PLUS_ANCHOR_LVD_PLUS
        if requested_method.is_lvd_plus_preparation
        else MethodName.TARGET_PLUS_ANCHOR_CE_GEOMETRY
        if requested_method is MethodName.PREPARE_ANCHOR_GEOMETRY_REFERENCE
        else requested_method
    )
    record_s0 = method is MethodName.RECORD_S0
    if settings.action == "generate-pilot" or requested_method.is_lvd_plus_preparation:
        settings = replace(
            settings,
            max_steps=-1,
            run_after_training=False,
            training_validation=False,
            recovery_save_steps=None,
        )
    elif record_s0 or requested_method is MethodName.PREPARE_ANCHOR_GEOMETRY_REFERENCE:
        settings = replace(
            settings,
            run_after_training=False,
            training_validation=False,
            recovery_save_steps=None,
        )
    artifact_root = Path(settings.artifact_root).expanduser().resolve()
    asset_root = artifact_root / "tmp" / "click-run" / "assets-v3"
    model_preset = MODEL_PRESETS[settings.model]
    anchor_preset = ANCHOR_DATASET_PRESETS.get(settings.anchor_dataset)
    anchor_only = method in {
        MethodName.RECORD_ANCHOR_D0,
        MethodName.FILTERING_ANCHOR,
        MethodName.CREATE_ANCHOR_DATASET,
        MethodName.TEST_INTERFERENCE,
        MethodName.TEST_CROSSING,
    } or settings.action in {"record-anchor-d0", "test-interference", "test-crossing"}

    if model_preset.model_id is None:
        model_path = asset_root / settings.model
        _ensure_tiny_model(model_path)
        model_id = str(model_path)
    else:
        model_id = model_preset.model_id

    anchor_path = asset_root / "tiny_general.jsonl"
    validation_config: DatasetConfig | None = None
    if record_s0:
        # Structural placeholders only: recording must never write/load datasets.
        target_config = DatasetConfig(
            source="jsonl",
            path=str(asset_root / "record_s0_target_placeholder.jsonl"),
            adapter="prompt_response",
            max_samples=1,
        )
        evaluation_config = target_config
    elif anchor_only:
        target_path = asset_root / "record_anchor_d0_target_placeholder.jsonl"
        _write_jsonl(target_path, _target_rows("tiny_instruction"))
        target_config = DatasetConfig(
            source="jsonl",
            path=str(target_path),
            adapter="prompt_response",
            max_samples=1,
        )
        evaluation_config = target_config
    elif (target_preset := TARGET_DATASET_PRESETS[settings.target_dataset]).path is None:
        target_path = asset_root / f"{settings.target_dataset}.jsonl"
        _write_jsonl(target_path, _target_rows(settings.target_dataset))
        target_config = DatasetConfig(
            source="jsonl",
            path=str(target_path),
            adapter=target_preset.adapter,
            max_samples=settings.target_max_samples,
        )
        evaluation_config = DatasetConfig(
            source="jsonl",
            path=str(target_path),
            adapter=target_preset.adapter,
            max_samples=settings.evaluation_max_samples,
        )
    else:
        registered_target = DEFAULT_DATASET_REGISTRY.dataset(settings.target_dataset)
        if DatasetRole.TARGET_TRAIN in registered_target.role_splits:
            target_max_samples = settings.target_max_samples
            if target_max_samples is None:
                target_max_samples = CODE_TARGET_SAMPLE_BUDGETS.get(settings.target_dataset)
            target_config = registered_target.config_for(
                DatasetRole.TARGET_TRAIN,
                max_samples=target_max_samples,
            )
            if (
                settings.training_validation
                and DatasetRole.TARGET_VALIDATION in registered_target.role_splits
            ):
                validation_config = registered_target.config_for(
                    DatasetRole.TARGET_VALIDATION,
                    max_samples=settings.target_validation_max_samples,
                )
        elif settings.action == "native-evaluate":
            target_config = registered_target.config_for(
                DatasetRole.TARGET_EVALUATION,
                max_samples=settings.evaluation_max_samples,
            )
        else:
            raise ValueError(
                f"{settings.target_dataset} is evaluation-only and requires action=native-evaluate"
            )
        evaluation_key = TARGET_EVALUATION_KEYS.get(
            settings.target_dataset,
            settings.target_dataset,
        )
        registered_evaluation = DEFAULT_DATASET_REGISTRY.dataset(evaluation_key)
        evaluation_config = registered_evaluation.config_for(
            DatasetRole.TARGET_EVALUATION,
            max_samples=settings.evaluation_max_samples,
        )

    anchor_config: DatasetConfig | None
    if record_s0:
        anchor_config = None
    elif settings.anchor_dataset == "tiny_general":
        _write_jsonl(anchor_path, _anchor_rows())
        anchor_config = DatasetConfig(
            source="jsonl",
            path=str(anchor_path),
            adapter="prompt_response",
            max_samples=settings.anchor_max_samples,
        )
    elif settings.anchor_dataset in LABEL_COVERAGE_ANCHOR_DATASETS:
        filtered_dir = label_coverage_dataset_dir(artifact_root, settings.model)
        # The data module validates SHA, label-coverage policy, model identity,
        # revision, and tokenization. Never treat a legacy manifest as this policy.
        load_label_coverage_manifest(
            filtered_dir,
            model_name=settings.model,
            model_id=model_id,
            model_revision=model_preset.revision,
            max_length=settings.max_length,
        )
        anchor_config = DatasetConfig(
            source="jsonl",
            path=str(filtered_dir / "train.jsonl"),
            adapter="tokenized",
            max_samples=None,
        )
    elif settings.anchor_dataset in CREATED_ANCHOR_DATASETS:
        created_dir = created_anchor_dataset_dir(artifact_root, settings.model, settings.max_length)
        # Verifies policy, model identity, tokenization contract and the signed
        # manifest; the full record replay runs once in ensure_preset_cache.
        try:
            load_created_anchor_manifest(
                created_dir,
                model_name=settings.model,
                model_id=model_id,
                model_revision=model_preset.revision,
                max_length=settings.max_length,
                verify_records=False,
            )
        except (ValueError, OSError) as exc:
            raise FileNotFoundError(
                f"created Anchor dataset is missing or incompatible: {created_dir}; run "
                "METHOD='create_anchor_dataset' with the same MODEL and MAX_LENGTH first"
            ) from exc
        anchor_config = DatasetConfig(
            source="jsonl",
            path=str(created_dir / "train.jsonl"),
            adapter="tokenized",
            max_samples=None,
        )
    elif settings.anchor_dataset in FILTERED_ANCHOR_DATASETS:
        filtered_dir = filtered_anchor_dataset_dir(artifact_root, settings.model)
        dataset_path = filtered_dir / "train.jsonl"
        manifest_path = filtered_dir / "manifest.json"
        if not dataset_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(
                f"filtered Anchor dataset is missing: {filtered_dir}; run "
                "METHOD='filtering_anchor' with ANCHOR_DATASET='smart_flan' first"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_name = filtered_anchor_dataset_name(settings.model)
        if (
            manifest.get("dataset_name") != expected_name
            or manifest.get("model", {}).get("preset") != settings.model
            or manifest.get("model", {}).get("revision") != model_preset.revision
            or manifest.get("tokenization", {}).get("max_length") != settings.max_length
        ):
            raise ValueError(
                f"filtered Anchor manifest is incompatible with model={settings.model}: "
                f"{manifest_path}"
            )
        anchor_config = DatasetConfig(
            source="jsonl",
            path=str(dataset_path),
            adapter="tokenized",
            max_samples=None,
        )
    elif anchor_preset is not None:
        anchor_config = _real_dataset_config(
            anchor_preset,
            split=anchor_preset.train_split,
            max_samples=settings.anchor_max_samples,
        )
    else:
        anchor_config = None
    if anchor_config is not None:
        anchor_config.metadata_path = settings.anchor_metadata_path

    training_mode = TrainingMode(settings.training_mode)
    freeze_unembedding = training_mode is not TrainingMode.FULL_FT
    untie_embeddings = training_mode is TrainingMode.BODY_FT_FROZEN_HEAD
    run_name = settings.run_name or datetime.now(timezone.utc).strftime(
        f"click-{settings.model}-{settings.target_dataset}-{requested_method.value}-"
        f"{settings.distributed_strategy}-%Y%m%dT%H%M%S-%fZ"
    )
    lvd_plus_config = LVDPlusConfig()
    s0_config = S0Config()
    anchor_geometry_config = AnchorGeometryConfig()
    if method.uses_dynamic_geometry:
        lvd_config = LVDConfig()
        anchor_geometry_config = load_anchor_geometry_config(settings.anchor_geometry_config_path)
        reference_path = (
            settings.anchor_geometry_reference_path or anchor_geometry_config.reference_cache_path
        )
        if reference_path is None:
            recipe = {
                "model_id": model_id,
                "revision": model_preset.revision,
                "anchor": asdict(anchor_config),
                "max_length": settings.max_length,
                "base_top_k": anchor_geometry_config.base_top_k,
                "mixed_precision": settings.mixed_precision,
            }
            digest = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()[:20]
            reference_path = str(
                artifact_root
                / "anchor_geometry_reference"
                / f"{settings.model}-{settings.anchor_dataset}-v1-{digest}"
            )
        anchor_geometry_config.reference_cache_path = str(
            Path(reference_path).expanduser().resolve()
        )
    elif record_s0:
        # No legacy LVD or LVD+ training YAML, d0, pilot or lambda calibration.
        lvd_config = LVDConfig()
        s0_config = load_s0_config(settings.s0_config_path)
        lvd_plus_config = LVDPlusConfig(
            anchor_mode=LVDPlusAnchorMode.RECORDED_S0,
            s0_cache_path=str(s0_cache_path_for_settings(settings, s0_config)),
        )
    elif method.uses_lvd_plus:
        # Legacy LVD is inert metadata here; do not even read its YAML or artifacts.
        lvd_config = LVDConfig()
        lvd_plus_config = load_lvd_plus_config(settings.lvd_plus_config_path)
        if settings.lvd_plus_direction is not None:
            lvd_plus_config = replace(lvd_plus_config, top_k=settings.lvd_plus_direction)
        # None retains YAML overrides or stays unresolved. Automatic scale/lambda
        # paths bind the actual distributed world and are resolved by the CLI,
        # never by the single-process launcher/cache-preparation parent.
        for name, value in (
            ("scale_cache_path", settings.lvd_plus_scale_cache_path),
            ("lambda_cache_path", settings.lvd_plus_lambda_cache_path),
        ):
            if value is not None:
                lvd_plus_config = replace(lvd_plus_config, **{name: value})
        if lvd_plus_config.anchor_mode is LVDPlusAnchorMode.RECORDED_S0:
            s0_config = load_s0_config(settings.s0_config_path)
            cache_settings = settings
            if settings.s0_cache_path is None:
                cache_settings = replace(settings, s0_cache_path=lvd_plus_config.s0_cache_path)
            lvd_plus_config = replace(
                lvd_plus_config,
                s0_cache_path=str(s0_cache_path_for_settings(cache_settings, s0_config)),
            )
        elif settings.s0_cache_path is not None:
            raise ValueError(
                "s0_cache_path is forbidden with lvd_plus.anchor_mode=half_unembedding"
            )
        lvd_plus_config.validate()
    else:
        lvd_config = load_lvd_config(settings.lvd_config_path)
        if settings.lvd_direction is not None:
            lvd_config = replace(lvd_config, top_k=settings.lvd_direction)
        if settings.lvd_position_strategy is not None:
            lvd_config = replace(lvd_config, position_strategy=settings.lvd_position_strategy)
        if settings.lvd_positions_per_sample is not None:
            lvd_config = replace(lvd_config, positions_per_sample=settings.lvd_positions_per_sample)
        if settings.lvd_use_capacity_weights is not None:
            lvd_config = replace(lvd_config, use_capacity_weights=settings.lvd_use_capacity_weights)
        lvd_config.validate()
    uses_d0_cache = method in {
        MethodName.RECORD_ANCHOR_D0,
        MethodName.TARGET_PLUS_ANCHOR_LVD,
        MethodName.TARGET_PLUS_ANCHOR_CE_LVD,
        MethodName.TEST_INTERFERENCE,
    }
    resolved_d0_path = (
        settings.anchor_d0_cache_path or lvd_config.d0_cache_path if uses_d0_cache else None
    )
    if uses_d0_cache and resolved_d0_path is None and settings.anchor_dataset != "none":
        sample_count = (
            "all" if settings.anchor_max_samples is None else str(settings.anchor_max_samples)
        )
        model_name = Path(model_id).name
        cache_name = (
            f"d0-v5-all-token-{model_name}-target-{settings.target_dataset}-"
            f"anchor-{settings.anchor_dataset}-"
            f"{lvd_config.position_strategy}-positions-{lvd_config.positions_per_sample}-"
            f"seed-{lvd_config.position_seed}-"
            f"filter-{lvd_config.d0_filter_mode}-n-{sample_count}"
        )
        resolved_d0_path = str(artifact_root / "reference_cache" / cache_name)
    lvd_config = replace(lvd_config, d0_cache_path=resolved_d0_path)
    if settings.lvd_calibration_checkpoint is not None:
        lvd_config = replace(
            lvd_config,
            calibration_checkpoint=settings.lvd_calibration_checkpoint,
        )

    config = ExperimentConfig(
        model=ModelConfig(
            model_id=model_id,
            revision=model_preset.revision,
            model_variant="base",
            training_mode=training_mode,
            freeze_unembedding=freeze_unembedding,
            untie_tied_embeddings=untie_embeddings,
            local_files_only=True,
            gradient_checkpointing=settings.gradient_checkpointing,
        ),
        lora=LoraConfig(
            rank=settings.lora_rank,
            alpha=settings.lora_alpha,
            dropout=settings.lora_dropout,
            target_modules=("q_proj", "v_proj"),
        ),
        data=DataConfig(
            target=target_config,
            validation=validation_config,
            anchor=anchor_config,
            max_length=settings.max_length,
            target_batch_size=settings.target_batch_size,
            anchor_batch_size=settings.anchor_batch_size,
            anchor_batches_per_target=(
                0.0 if method is MethodName.TARGET_ONLY or record_s0 else 1.0
            ),
            anchor_sampling_strategy=AnchorSamplingStrategy(settings.anchor_sampling_strategy),
            steps_per_epoch=settings.max_steps if settings.max_steps > 0 else None,
            seed=settings.seed,
            pad_to_multiple_of=None,
        ),
        method=MethodConfig(
            name=method,
            anchor_weight=settings.anchor_weight,
            kl_temperature=settings.kl_temperature,
            lvd_weight=0.0,
        ),
        reference=ReferenceConfig(
            mode=(
                ReferenceMode.ONLINE
                if method is MethodName.TARGET_PLUS_ANCHOR_KL
                else (ReferenceMode.CACHED if method.uses_lvd else ReferenceMode.NONE)
            ),
            cache_path=(resolved_d0_path if method.uses_lvd else None),
        ),
        lvd=lvd_config,
        training=TrainingConfig(
            mixed_precision=settings.mixed_precision,
            optimizer=settings.optimizer,
            learning_rate=settings.learning_rate,
            weight_decay=settings.weight_decay,
            num_train_epochs=settings.num_train_epochs,
            max_steps=settings.max_steps,
            warmup_ratio=settings.warmup_ratio,
            gradient_accumulation_steps=settings.gradient_accumulation_steps,
            recovery_save_steps=settings.recovery_save_steps,
            fsdp_sync_each_batch=settings.fsdp_sync_each_batch,
            fsdp_cpu_offload=settings.fsdp_cpu_offload,
            fsdp_keep_low_precision_params=settings.fsdp_keep_low_precision_params,
            logging_steps=1,
            save_strategy="epoch" if validation_config is not None else "no",
            seed=settings.seed,
            data_seed=settings.seed,
            distributed_strategy=settings.distributed_strategy,
            report_to=(),
            best_model_metric=settings.best_model_metric,
        ),
        evaluation=EvaluationConfig(
            target=evaluation_config,
            batch_size=settings.evaluation_batch_size,
            run_after_training=settings.run_after_training,
        ),
        paths=PathsConfig(
            artifact_root=str(artifact_root),
            run_name=run_name,
            estimated_run_bytes=model_preset.estimated_run_bytes,
            minimum_free_bytes=0 if model_preset.model_id is None else 2 * 1024**4,
        ),
        lvd_plus=lvd_plus_config,
        s0=s0_config,
        anchor_geometry=anchor_geometry_config,
    )
    config.validate()
    return config
