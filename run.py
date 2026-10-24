# pyright: reportMissingImports=false
"""Edit SETTINGS, then click VS Code's 'Run Python File' button."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

# -------------------------- PRIMARY SETTINGS ------------------------------
# Usually edit only these seven values.
# train | audit-model | native-evaluate | build-reference-cache | audit-data | benchmark
# generate_pilot/filtering_anchor/record_anchor_d0/prepare_anchor_geometry_reference/test_*
# override ACTION. Historical S0/pressure workflows are no longer click actions.
ACTION = "train"

# tiny_qwen2 | qwen25_0.5b | qwen25_3b | qwen25_7b | mistral_7b
MODEL = "qwen25_3b"

# tiny_gsm8k | tiny_sst2 | tiny_instruction | gsm8k
# cola | sst2 | mrpc | stsb | qqp | mnli | qnli | rte
# Code training presets (ACTION="train"):
# codealpaca (paper evaluation: humaneval pass@1/pass@5/pass@10)
# code_bagel_hermes (paper evaluation: codexglue_python token accuracy)
# infinity_instruct_code (paper evaluation: codexglue_python token accuracy)
# Code evaluation presets (ACTION="native-evaluate" only):
# humaneval (evaluates CodeAlpaca-trained checkpoints; pass@1/pass@5/pass@10)
# codexglue_python (evaluates Infinity/Bagel checkpoints; Python token accuracy)
# codexglue_java (optional second Infinity/Bagel evaluation; Java token accuracy)
# Logic & instruction-following training presets (ACTION="train"):
# evol_instruct (WizardLM Evol-Instruct 70k; paper evaluation: alpaca_eval win rate)
# logiqa2 (official English MRC train/dev; paper evaluation: LogiQA2 test 0-shot Acc)
# Logic evaluation presets (ACTION="native-evaluate" only):
# alpaca_eval (evaluates Evol-Instruct checkpoints; writes official model_outputs.json)
# logiqa2 (evaluates LogiQA2 checkpoints; zero-shot four-choice Accuracy)
TARGET_DATASET = "gsm8k"

# none | tiny_general | smart_flan | smart_flan_[model_name]
# smart_flan_mistral_7b_label_coverage_v1
# Dynamic geometry uses the label-only cover; AUTO_PREPARE_CACHE builds it on CPU.
ANCHOR_DATASET = "smart_flan_qwen25_3b_label_coverage_v1"

# generate_pilot | target_only | target_plus_anchor_ce | target_plus_anchor_kl
# target_plus_anchor_lvd | target_plus_anchor_ce_lvd | target_plus_null_lvd | record_anchor_d0
# target_plus_anchor_ce_geometry | prepare_anchor_geometry_reference
# filtering_anchor | test_interference | test_crossing
# Base-conditioned band workflow: set ACTION="train", then select METHOD in this order:
# prepare_anchor_geometry_reference -> target_plus_anchor_ce_geometry.
# Use TRAINING_MODE="full_ft". Preparation records Base outputs on the first GPU;
# training uses Target CE + Anchor CE, then calibrates and ramps base_band_v2.
# Existing matching Base references are reusable; centered-loss checkpoints are not.
# There is no S0, pressure moment, scale recording, d0 or pilot in this workflow.
# Legacy target_plus_anchor_lvd keeps its separate cache/pilot protocol and YAML;
# see docs/lvd-three-segment-20260926.md. Never resume across these methods.
METHOD = "target_plus_anchor_ce"

TRAINING_MODE = "full_ft"  # full_ft | body_ft_frozen_head | lora

# Training resume: use checkpoint-N (model + optimizer/scheduler/RNG), not final/.
# Set None for a fresh experiment or when changing MODEL / TARGET_DATASET / METHOD.
# The previous random-K / four-position checkpoint is NOT compatible with the new protocol.
RESUME_CHECKPOINT = None

# Required by test_interference/test_crossing; otherwise used by evaluation actions.  () 
EVAL_CHECKPOINT = (
    "/data1/chenhang/SFT_Conflict/artifacts/checkpoints/click-mistral_7b-logiqa2-ce/final"
)
LLM_EVALUATION = False
# ---------------------- TASK-DEPENDENT OVERRIDES --------------------------ha
# Keep None for automatic selection. Set a bool or positive integer to override.
GRADIENT_CHECKPOINTING: bool | None = None
FSDP_SYNC_EACH_BATCH: bool | None = None
# None preserves >=6B auto-offload; False requires enough GPU memory for AdamW.
# False verified for current Mistral/CoLA/LVD, FP32, 4x A100 40GB; see performance report.
# Revalidate memory or reset to None when changing model, GPU count, length or batch.
FSDP_CPU_OFFLOAD: bool | None = False
FSDP_KEEP_LOW_PRECISION_PARAMS: bool | None = None
EVALUATION_MAX_SAMPLES: int | None = None
NATIVE_MAX_NEW_TOKENS: int | None = None

# --------------------------- FIXED DEFAULTS -------------------------------
SUPERVISION = "response_only"
MIXED_PRECISION = "fp32"  # fp32 | bf16 | fp16
OPTIMIZER = "adamw_torch"  # adamw_torch | adamw_bnb_8bit | adafactor

# Training: single uses one GPU; ddp/fsdp use every ID listed here.
# Native evaluation uses the first visible GPU, not the training FSDP/DDP strategy.
DISTRIBUTED_STRATEGY = "fsdp"  # single | ddp | fsdp1
GPU_IDS = (1, 2, 3)
CPU_THREADS_PER_PROCESS = 1  # Positive integer; affects CPU offload/optimizer throughput.
FSDP_AUTO_WRAP_POLICY = "transformer_based_wrap"
MAIN_PROCESS_PORT = None  # None chooses a free local port automatically.
CUDA_ALLOCATOR_CONF = "expandable_segments:True"

MAX_STEPS = -1  # Use a positive value only for a bounded smoke run.
# METHOD="generate_pilot"; joint option: "target_plus_anchor_ce_lvd".
PILOT_METHOD = "target_plus_anchor_lvd"
PILOT_STEPS = 10  # Optimizer updates; the pilot retains the full training scheduler.
PILOT_HEALTH_INTERVAL = 5
NUM_TRAIN_EPOCHS = 3.0
# Extra restart checkpoints; keeps epoch evaluation/best-model selection unchanged.
# None disables. 100 bounds lost work to ~40 minutes at the measured 25 s/step.
RECOVERY_SAVE_STEPS: int | None = 100
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06
# MergeLM encoder references use 1e-5 or 5e-5 by task. Decoder search grids:
# CoLA/MRPC/STS-B: (1e-5, 2e-5, 5e-5); other GLUE tasks: (5e-6, 1e-5, 2e-5).
MAX_LENGTH = 512
TARGET_MAX_SAMPLES = None
TARGET_VALIDATION_MAX_SAMPLES = None  # None uses the complete derived 10% split.
ANCHOR_MAX_SAMPLES = 40_000
ANCHOR_METADATA_PATH = None  # JSONL keyed by sample_id or source_index.
# label_coverage_v1: CPU label coverage; legacy_quality_v1: old correctness filter.
FILTERING_POLICY = "label_coverage_v1"
FILTER_THRESHOLD = 0.6  # Used ONLY by legacy_quality_v1, not by label coverage.
FILTERING_BATCH_SIZE = 4  # Legacy correctness inference only; label coverage needs no GPU.
EVALUATION_BATCH_SIZE = 4  # Native eval; caps concurrent HumanEval sequences.
# Native LogiQA2 scores 4 options per question (4x rows of FP32 152k-vocab logits);
# qwen25_7b at batch 4 OOMs on one A100 40GB, so only that case is capped here.
QWEN25_7B_LOGIQA2_EVALUATION_BATCH_SIZE = 2
TARGET_BATCH_SIZE = 1
ANCHOR_BATCH_SIZE = 1
ANCHOR_SAMPLING_STRATEGY = "mixture_proportional"  # task_balanced | mixture_proportional | random
GRADIENT_ACCUMULATION_STEPS = 4
RUN_AFTER_TRAINING = True

LORA_RANK = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
ANCHOR_WEIGHT = 0.2
KL_TEMPERATURE = 1.0

SEED = 42
ARTIFACT_ROOT = "/data1/chenhang/SFT_Conflict/artifacts"
RUN_NAME = None  # None creates a unique run name each time.
AUTO_PREPARE_CACHE = True

# M2 reference-cache settings.
REFERENCE_CACHE_DIR = None
ANCHOR_POSITION_STRATEGY = "sample_k_positions"
ANCHOR_POSITIONS_PER_SAMPLE = 4
REFERENCE_SAMPLES_PER_SHARD = 4
REFERENCE_STORAGE_DTYPE = "float32"
REFERENCE_FILTER_MODE = "base_correct"
MINIMUM_REFERENCE_CONFIDENCE = None
MINIMUM_REFERENCE_MARGIN = None

# M3 weighted-pressure LVD settings. Recording and training load this commented YAML.
LVD_DIRECTION = 512  # Current-logit Top-K competitors, excluding the base reference token.
LVD_DYNAMIC_POSITIONS = True  # True: all eligible response positions; False: sample fixed K.
# When False, ANCHOR_POSITIONS_PER_SAMPLE controls the fixed count.
# Dynamic positions require recording a new d0 cache with these same settings.
LVD_CONFIG_PATH = str(Path(__file__).resolve().parent / "configs/method/weighted_pressure_lvd.yaml")
# YAML gamma0=0.8 gives zero LVD pressure for up to 80% original-margin erosion.
adapted_weight = False  # False: alpha=1/K; True: normalized inverse-d0 capacity weights.
# Changing gamma0/weights requires fresh lambda calibration, not a new Base d0 cache.
# ACTION="audit-model": real hidden states checked against every planar LVD cell boundary.
LVD_AUDIT_MAX_POSITIONS = 8
LVD_AUDIT_VOCABULARY_CHUNK_SIZE = 256
# METHOD="test_interference": bounded matrix-multiply chunks and optional artifact directory.
INTERFERENCE_TOKEN_CHUNK_SIZE = 256
INTERFERENCE_POSITION_CHUNK_SIZE = 1024
INTERFERENCE_OUTPUT_DIR = None
# test_interference requires an explicit all_reference d0 cache; test_crossing ignores this setting.
# Otherwise, None derives the same model/Anchor/position-specific path for recording and training.
# Auto-derived from the NEW position protocol; never reuse old four-position d0.
ANCHOR_D0_CACHE_PATH = None
# None discovers an accepted compatible pilot; set a path for an explicit export.
LVD_CALIBRATION_CHECKPOINT = None

# Dynamic geometry is independent of the legacy LVD/d0 options above.
# Edit the YAML for Base-derived band scales, Huber delta, warmup and calibration.
ANCHOR_GEOMETRY_CONFIG_PATH = str(
    Path(__file__).resolve().parent / "configs/method/anchor_ce_geometry_band.yaml"
)
# None derives the same Base-output cache for preparation and training.
ANCHOR_GEOMETRY_REFERENCE_PATH: str | None = None

# M2 native evaluation and leakage-audit settings.
NATIVE_STOP_TOKENS = ()
NATIVE_EVALUATION_DIR = None
# After native TARGET_DATASET evaluation, run the pinned lm-evaluation-harness suite.
LLM_EVALUATION_TASKS = (
    "boolq",
    "rte",
    "hellaswag",
    "winogrande",
    "arc_easy",
    "arc_challenge",
    "openbookqa",
    "piqa",
    "truthfulqa_mc2",
    "commonsense_qa",
    "copa",
    "race",
    "mmlu",
    "gsm8k",
    "bbh",
    "ifeval",
    "lambada_openai",
)
LLM_EVALUATION_BATCH_SIZE = "auto"
LLM_EVALUATION_NUM_FEWSHOT = None  # None uses each harness task's default.
LLM_EVALUATION_LIMIT = None  # Set a positive number for a non-formal smoke run.
LLM_EVALUATION_CHECK_INTEGRITY = False
ALPACA_EVAL_RUN_JUDGE = False  # True spends API credits; export OPENAI_API_KEY in the shell.
# chatgpt_fn reproduces DARE; weighted_alpaca_eval_gpt4_turbo is current AlpacaEval 2.0.
ALPACA_EVAL_ANNOTATORS_CONFIG = "chatgpt_fn"
ALLOW_CODE_EXECUTION = True  # Required for HumanEval; generated Python is executed locally.
HUMANEVAL_SAMPLES_PER_PROBLEM = 20
HUMANEVAL_TEMPERATURE = 0.2
HUMANEVAL_TOP_P = 0.95
HUMANEVAL_TIMEOUT_SECONDS = 3.0
HUMANEVAL_EXECUTION_WORKERS = 16
NEAR_DUPLICATE_THRESHOLD = 0.85
DATA_AUDIT_OUTPUT = None

# M2 six-condition benchmark. False writes the full plan without running it.
BENCHMARK_EXECUTE = False
BENCHMARK_FORMAL = False
BENCHMARK_SEEDS = (42,)  # Formal runs require at least three unique seeds.
BENCHMARK_GENERAL_TASKS = (
    "boolq",
    "rte",
    "hellaswag",
    "winogrande",
    "arc_challenge",
    "arc_easy",
    "openbookqa",
    "piqa",
    "truthfulqa",
)
BENCHMARK_GENERAL_LIMIT = None  # Only allowed when BENCHMARK_FORMAL=False.
BENCHMARK_NUM_FEWSHOT = 0
BENCHMARK_CHECK_INTEGRITY = False
BENCHMARK_OUTPUT_DIR = None

# Used only by the matching ACTION.
LORA_ADAPTER_PATH = None
MERGED_MODEL_OUTPUT = None
# -------------------------------------------------------------------------


PINNED_ENVIRONMENT = Path("/data1/chenhang/SFT_Conflict/artifacts/envs/train")
PINNED_PYTHON = PINNED_ENVIRONMENT / "bin" / "python"
PROJECT_SOURCE = Path(__file__).resolve().parent / "src"
DISTRIBUTED_CHILD_ENV = "SFT_CONFLICT_DISTRIBUTED_CHILD"
DISTRIBUTED_RUN_NAME_ENV = "SFT_CONFLICT_DISTRIBUTED_RUN_NAME"
MAX_STEPS_OVERRIDE_ENV = "SFT_CONFLICT_MAX_STEPS_OVERRIDE"
RUN_AFTER_TRAINING_OVERRIDE_ENV = "SFT_CONFLICT_RUN_AFTER_TRAINING_OVERRIDE"
TRAINING_VALIDATION_OVERRIDE_ENV = "SFT_CONFLICT_TRAINING_VALIDATION_OVERRIDE"
RESUME_CHECKPOINT_OVERRIDE_ENV = "SFT_CONFLICT_RESUME_CHECKPOINT_OVERRIDE"

_DISTRIBUTED_TRAINING_ACTIONS = {"train", "generate-pilot"}
_GEOMETRY_REFERENCE_ACTION = "prepare-anchor-geometry-reference"
_RETIRED_CLICK_METHODS = {
    "record_s0",
    "record_lvd_plus_scales",
    "calibrate_lvd_plus_lambda",
    "target_plus_anchor_lvd+",
    "target_plus_anchor_lvd_plus",
}


class ResolvedRuntimeSettings(NamedTuple):
    gradient_checkpointing: bool
    fsdp_sync_each_batch: bool
    fsdp_keep_low_precision_params: bool
    evaluation_max_samples: int | None
    native_max_new_tokens: int
    evaluation_batch_size: int


def _effective_action() -> str:
    if METHOD in _RETIRED_CLICK_METHODS or ACTION in {
        "record-s0",
        "record-lvd-plus-scales",
        "calibrate-lvd-plus-lambda",
    }:
        raise ValueError(
            "Historical S0/LVD+ click workflow is retired; use "
            "prepare_anchor_geometry_reference -> target_plus_anchor_ce_geometry with full_ft. "
            "Historical configs remain readable through the CLI."
        )
    return {
        "generate_pilot": "generate-pilot",
        "record_anchor_d0": "record-anchor-d0",
        "prepare_anchor_geometry_reference": _GEOMETRY_REFERENCE_ACTION,
        "filtering_anchor": "filtering-anchor",
        "test_interference": "test-interference",
        "test_crossing": "test-crossing",
    }.get(METHOD, ACTION)


def _effective_method() -> str:
    method = PILOT_METHOD if METHOD == "generate_pilot" else METHOD
    if _effective_action() == "generate-pilot" and method not in (
        "target_plus_anchor_lvd",
        "target_plus_anchor_ce_lvd",
    ):
        setting = "PILOT_METHOD" if METHOD == "generate_pilot" else "METHOD"
        raise ValueError(
            f"generate-pilot requires {setting}='target_plus_anchor_lvd' or "
            f"'target_plus_anchor_ce_lvd'; got {method!r}"
        )
    return method


def _effective_anchor_dataset() -> str:
    if _effective_action() == "filtering-anchor":
        return "smart_flan"
    return ANCHOR_DATASET


def _label_coverage_filtering() -> bool:
    return _effective_action() == "filtering-anchor" and FILTERING_POLICY == "label_coverage_v1"


def _effective_training_mode() -> str:
    action = _effective_action()
    if (
        METHOD in {"prepare_anchor_geometry_reference", "target_plus_anchor_ce_geometry"}
        and TRAINING_MODE != "full_ft"
    ):
        raise ValueError("dynamic geometry requires TRAINING_MODE='full_ft'")
    if action == "generate-pilot" and TRAINING_MODE not in (
        "body_ft_frozen_head",
        "lora",
    ):
        raise ValueError(
            f"{action} requires TRAINING_MODE='body_ft_frozen_head' or 'lora' "
            f"to keep the Unembedding frozen; got {TRAINING_MODE!r}"
        )
    return TRAINING_MODE


def _training_validation_enabled() -> bool:
    if _effective_action() in {"generate-pilot", _GEOMETRY_REFERENCE_ACTION}:
        return False
    value = os.environ.get(TRAINING_VALIDATION_OVERRIDE_ENV)
    if value is None:
        return True
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{TRAINING_VALIDATION_OVERRIDE_ENV} must be a boolean, got {value!r}")


def _resolve_bool_override(name: str, value: bool | None, automatic: bool) -> bool:
    if value is None:
        return automatic
    if type(value) is not bool:
        raise ValueError(f"{name} must be True, False, or None for automatic selection")
    return value


def _resolve_positive_int_override(name: str, value: int | None, automatic: int) -> int:
    if value is None:
        return automatic
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer or None for automatic selection")
    return value


def _resolved_runtime_settings() -> ResolvedRuntimeSettings:
    effective_action = _effective_action()
    training_action = effective_action in {*_DISTRIBUTED_TRAINING_ACTIONS, "benchmark"}
    large_training_model = MODEL in {"qwen25_3b", "qwen25_7b", "mistral_7b"}
    fsdp_training = training_action and DISTRIBUTED_STRATEGY == "fsdp"
    if EVALUATION_MAX_SAMPLES is None:
        evaluation_max_samples = (
            256 if effective_action == "train" and _effective_max_steps() > 0 else None
        )
    else:
        evaluation_max_samples = _resolve_positive_int_override(
            "EVALUATION_MAX_SAMPLES",
            EVALUATION_MAX_SAMPLES,
            256,
        )
    automatic_new_tokens = {
        "gsm8k": 256,
        "tiny_gsm8k": 256,
        "stsb": 8,
        "humaneval": 512,
        "alpaca_eval": 2048,
        "logiqa2": 1,
    }.get(TARGET_DATASET, 1)
    evaluation_batch_size = EVALUATION_BATCH_SIZE
    if (
        effective_action == "native-evaluate"
        and MODEL == "qwen25_7b"
        and TARGET_DATASET == "logiqa2"
    ):
        evaluation_batch_size = min(EVALUATION_BATCH_SIZE, QWEN25_7B_LOGIQA2_EVALUATION_BATCH_SIZE)
    return ResolvedRuntimeSettings(
        gradient_checkpointing=_resolve_bool_override(
            "GRADIENT_CHECKPOINTING",
            GRADIENT_CHECKPOINTING,
            training_action and large_training_model,
        ),
        fsdp_sync_each_batch=_resolve_bool_override(
            "FSDP_SYNC_EACH_BATCH",
            FSDP_SYNC_EACH_BATCH,
            fsdp_training,
        ),
        fsdp_keep_low_precision_params=_resolve_bool_override(
            "FSDP_KEEP_LOW_PRECISION_PARAMS",
            FSDP_KEEP_LOW_PRECISION_PARAMS,
            False,
        ),
        evaluation_max_samples=evaluation_max_samples,
        native_max_new_tokens=_resolve_positive_int_override(
            "NATIVE_MAX_NEW_TOKENS",
            NATIVE_MAX_NEW_TOKENS,
            automatic_new_tokens,
        ),
        evaluation_batch_size=evaluation_batch_size,
    )


def _print_resolved_runtime_settings(runtime: ResolvedRuntimeSettings) -> None:
    evaluation_samples = (
        "full split" if runtime.evaluation_max_samples is None else runtime.evaluation_max_samples
    )
    print(
        "Resolved task-dependent settings: "
        f"gradient_checkpointing={runtime.gradient_checkpointing}, "
        f"fsdp_sync_each_batch={runtime.fsdp_sync_each_batch}, "
        f"fsdp_cpu_offload={FSDP_CPU_OFFLOAD if FSDP_CPU_OFFLOAD is not None else 'auto (>=6B)'}, "
        f"cpu_threads_per_process={CPU_THREADS_PER_PROCESS}, "
        f"fsdp_keep_low_precision_params={runtime.fsdp_keep_low_precision_params}, "
        f"evaluation_max_samples={evaluation_samples}, "
        f"native_max_new_tokens={runtime.native_max_new_tokens}",
        flush=True,
    )
    if runtime.evaluation_batch_size != EVALUATION_BATCH_SIZE:
        print(
            "Native evaluation batch size capped for single-GPU memory: "
            f"EVALUATION_BATCH_SIZE={EVALUATION_BATCH_SIZE} -> "
            f"{runtime.evaluation_batch_size} ({MODEL}, {TARGET_DATASET})",
            flush=True,
        )


def _configure_cuda_allocator() -> None:
    if CUDA_ALLOCATOR_CONF:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = CUDA_ALLOCATOR_CONF


def _configure_hugging_face_environment() -> None:
    cache_root = Path(ARTIFACT_ROOT).expanduser().resolve() / "hf_cache"
    os.environ.update(
        {
            "HF_HOME": str(cache_root),
            "HF_HUB_CACHE": str(cache_root / "hub"),
            "HF_DATASETS_CACHE": str(cache_root / "datasets"),
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONFAULTHANDLER": "1",
        }
    )


def _ensure_training_interpreter() -> None:
    current_environment = Path(sys.prefix).resolve()
    if PINNED_PYTHON.is_file() and current_environment != PINNED_ENVIRONMENT.resolve():
        environment = os.environ.copy()
        environment["SFT_CONFLICT_CLICK_RUN"] = "1"
        os.execve(
            str(PINNED_PYTHON),
            [str(PINNED_PYTHON), str(Path(__file__).resolve())],
            environment,
        )


def _available_gpu_ids() -> tuple[int, ...]:
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    )
    return tuple(int(line.strip()) for line in completed.stdout.splitlines() if line.strip())


def _validated_gpu_ids() -> tuple[int, ...]:
    try:
        gpu_ids = tuple(int(gpu_id) for gpu_id in GPU_IDS)
    except (TypeError, ValueError) as exc:
        raise ValueError("GPU_IDS must be a tuple/list of integer physical GPU IDs") from exc
    if not gpu_ids:
        raise ValueError("GPU_IDS must contain at least one physical GPU ID")
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError(f"GPU_IDS contains duplicates: {gpu_ids}")
    available = _available_gpu_ids()
    unavailable = tuple(gpu_id for gpu_id in gpu_ids if gpu_id not in available)
    if unavailable:
        raise ValueError(
            f"GPU_IDS contains unavailable IDs {unavailable}; "
            f"available physical IDs are {available}"
        )
    training_action = _effective_action() in _DISTRIBUTED_TRAINING_ACTIONS
    if training_action and DISTRIBUTED_STRATEGY == "single" and len(gpu_ids) != 1:
        raise ValueError("DISTRIBUTED_STRATEGY='single' requires exactly one GPU ID")
    if training_action and DISTRIBUTED_STRATEGY in {"ddp", "fsdp"} and len(gpu_ids) < 2:
        raise ValueError(
            f"DISTRIBUTED_STRATEGY='{DISTRIBUTED_STRATEGY}' requires at least two GPU IDs"
        )
    if DISTRIBUTED_STRATEGY not in {"single", "ddp", "fsdp"}:
        raise ValueError("DISTRIBUTED_STRATEGY must be single, ddp, or fsdp")
    return gpu_ids


def _shared_run_name() -> str:
    existing = os.environ.get(DISTRIBUTED_RUN_NAME_ENV)
    if existing:
        return existing
    if RUN_NAME:
        return RUN_NAME
    objective_tag = ""
    if METHOD in {"target_plus_anchor_ce_geometry", "prepare_anchor_geometry_reference"}:
        from sft_conflict.config import load_anchor_geometry_config

        objective = load_anchor_geometry_config(ANCHOR_GEOMETRY_CONFIG_PATH).objective
        if objective != "centered_v1":
            objective_tag = f"-{objective}"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S-%fZ")
    return (
        f"click-{MODEL}-{TARGET_DATASET}-{METHOD}{objective_tag}-{DISTRIBUTED_STRATEGY}-{timestamp}"
    )


def _free_local_port() -> int:
    if MAIN_PROCESS_PORT is not None:
        return int(MAIN_PROCESS_PORT)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(("127.0.0.1", 0))
        return int(server.getsockname()[1])


def _effective_max_steps() -> int:
    if _effective_action() in {"generate-pilot", _GEOMETRY_REFERENCE_ACTION}:
        return -1
    return int(os.environ.get(MAX_STEPS_OVERRIDE_ENV, MAX_STEPS))


def _effective_run_after_training() -> bool:
    if _effective_action() in {"generate-pilot", _GEOMETRY_REFERENCE_ACTION}:
        return False
    value = os.environ.get(RUN_AFTER_TRAINING_OVERRIDE_ENV)
    if value is None:
        return RUN_AFTER_TRAINING
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{RUN_AFTER_TRAINING_OVERRIDE_ENV} must be a boolean, got {value!r}")


def _effective_resume_checkpoint() -> str | None:
    checkpoint = os.environ.get(RESUME_CHECKPOINT_OVERRIDE_ENV) or RESUME_CHECKPOINT
    if _effective_action() in {"generate-pilot", _GEOMETRY_REFERENCE_ACTION}:
        if checkpoint:
            raise ValueError(
                f"{_effective_action()} does not support resume; clear RESUME_CHECKPOINT and "
                f"{RESUME_CHECKPOINT_OVERRIDE_ENV}"
            )
        return None
    return checkpoint


def _run_arguments() -> list[str]:
    dynamic_geometry = _effective_method() in {
        "target_plus_anchor_ce_geometry",
        "prepare_anchor_geometry_reference",
    }
    if not dynamic_geometry and type(adapted_weight) is not bool:
        raise ValueError("adapted_weight must be True or False")
    if dynamic_geometry and any(
        value is not None
        for value in (ANCHOR_D0_CACHE_PATH, LVD_CALIBRATION_CHECKPOINT, REFERENCE_CACHE_DIR)
    ):
        raise ValueError(
            "dynamic geometry does not accept legacy d0, reference-cache, or pilot paths"
        )
    runtime = _resolved_runtime_settings()
    arguments = [
        "run",
        "--action",
        _effective_action(),
        "--model",
        MODEL,
        "--target-dataset",
        TARGET_DATASET,
        "--anchor-dataset",
        _effective_anchor_dataset(),
        "--method",
        _effective_method(),
        "--supervision",
        SUPERVISION,
        "--training-mode",
        _effective_training_mode(),
        "--distributed-strategy",
        DISTRIBUTED_STRATEGY,
        "--mixed-precision",
        MIXED_PRECISION,
        "--optimizer",
        OPTIMIZER,
        "--max-steps",
        str(_effective_max_steps()),
        "--num-train-epochs",
        str(NUM_TRAIN_EPOCHS),
        "--learning-rate",
        str(LEARNING_RATE),
        "--weight-decay",
        str(WEIGHT_DECAY),
        "--warmup-ratio",
        str(WARMUP_RATIO),
        "--max-length",
        str(MAX_LENGTH),
        "--interference-token-chunk-size",
        str(INTERFERENCE_TOKEN_CHUNK_SIZE),
        "--interference-position-chunk-size",
        str(INTERFERENCE_POSITION_CHUNK_SIZE),
        "--target-max-samples",
        str(TARGET_MAX_SAMPLES),
        "--anchor-max-samples",
        str(ANCHOR_MAX_SAMPLES),
        "--evaluation-max-samples",
        str(runtime.evaluation_max_samples),
        "--evaluation-batch-size",
        str(runtime.evaluation_batch_size),
        "--target-batch-size",
        str(TARGET_BATCH_SIZE),
        "--anchor-batch-size",
        str(ANCHOR_BATCH_SIZE),
        "--anchor-sampling-strategy",
        ANCHOR_SAMPLING_STRATEGY,
        "--gradient-accumulation-steps",
        str(GRADIENT_ACCUMULATION_STEPS),
        "--lora-rank",
        str(LORA_RANK),
        "--lora-alpha",
        str(LORA_ALPHA),
        "--lora-dropout",
        str(LORA_DROPOUT),
        "--anchor-weight",
        str(ANCHOR_WEIGHT),
        "--kl-temperature",
        str(KL_TEMPERATURE),
        "--seed",
        str(SEED),
        "--artifact-root",
        ARTIFACT_ROOT,
        "--native-max-new-tokens",
        str(runtime.native_max_new_tokens),
        "--humaneval-samples-per-problem",
        str(HUMANEVAL_SAMPLES_PER_PROBLEM),
        "--humaneval-temperature",
        str(HUMANEVAL_TEMPERATURE),
        "--humaneval-top-p",
        str(HUMANEVAL_TOP_P),
        "--humaneval-timeout-seconds",
        str(HUMANEVAL_TIMEOUT_SECONDS),
        "--humaneval-execution-workers",
        str(HUMANEVAL_EXECUTION_WORKERS),
        "--near-duplicate-threshold",
        str(NEAR_DUPLICATE_THRESHOLD),
        "--filter-threshold",
        str(FILTER_THRESHOLD),
        "--filtering-batch-size",
        str(FILTERING_BATCH_SIZE),
        "--filtering-policy",
        FILTERING_POLICY,
        "--alpaca-eval-annotators-config",
        ALPACA_EVAL_ANNOTATORS_CONFIG,
    ]
    if dynamic_geometry:
        arguments.extend(
            [
                "--anchor-geometry-config",
                ANCHOR_GEOMETRY_CONFIG_PATH,
            ]
        )
        if ANCHOR_GEOMETRY_REFERENCE_PATH is not None:
            if (
                not isinstance(ANCHOR_GEOMETRY_REFERENCE_PATH, str)
                or not ANCHOR_GEOMETRY_REFERENCE_PATH.strip()
            ):
                raise ValueError("ANCHOR_GEOMETRY_REFERENCE_PATH must be a non-empty path or None")
            arguments.extend(["--anchor-geometry-reference-path", ANCHOR_GEOMETRY_REFERENCE_PATH])
    elif not _label_coverage_filtering():
        arguments.extend(
            [
                "--lvd-audit-max-positions",
                str(LVD_AUDIT_MAX_POSITIONS),
                "--lvd-audit-vocabulary-chunk-size",
                str(LVD_AUDIT_VOCABULARY_CHUNK_SIZE),
                "--lvd-config",
                LVD_CONFIG_PATH,
                "--lvd-direction",
                str(LVD_DIRECTION),
                "--lvd-use-capacity-weights" if adapted_weight else "--no-lvd-use-capacity-weights",
                "--lvd-position-strategy",
                "all_selected_tokens" if LVD_DYNAMIC_POSITIONS else "sample_k_positions",
                "--lvd-positions-per-sample",
                str(ANCHOR_POSITIONS_PER_SAMPLE),
                "--anchor-position-strategy",
                ANCHOR_POSITION_STRATEGY,
                "--anchor-positions-per-sample",
                str(ANCHOR_POSITIONS_PER_SAMPLE),
                "--reference-samples-per-shard",
                str(REFERENCE_SAMPLES_PER_SHARD),
                "--reference-storage-dtype",
                REFERENCE_STORAGE_DTYPE,
                "--reference-filter-mode",
                REFERENCE_FILTER_MODE,
            ]
        )
    if _effective_action() == "generate-pilot":
        arguments.extend(
            [
                "--pilot-steps",
                str(_resolve_positive_int_override("PILOT_STEPS", PILOT_STEPS, 10)),
                "--pilot-health-interval",
                str(
                    _resolve_positive_int_override(
                        "PILOT_HEALTH_INTERVAL", PILOT_HEALTH_INTERVAL, 5
                    )
                ),
            ]
        )
    if TARGET_VALIDATION_MAX_SAMPLES is not None:
        arguments.extend(["--target-validation-max-samples", str(TARGET_VALIDATION_MAX_SAMPLES)])
    if RECOVERY_SAVE_STEPS is not None and _effective_action() not in {
        "generate-pilot",
        _GEOMETRY_REFERENCE_ACTION,
    }:
        recovery_steps = _resolve_positive_int_override(
            "RECOVERY_SAVE_STEPS", RECOVERY_SAVE_STEPS, 100
        )
        arguments.extend(["--recovery-save-steps", str(recovery_steps)])
    arguments.append(
        "--gradient-checkpointing"
        if runtime.gradient_checkpointing
        else "--no-gradient-checkpointing"
    )
    arguments.append(
        "--fsdp-sync-each-batch" if runtime.fsdp_sync_each_batch else "--no-fsdp-sync-each-batch"
    )
    if FSDP_CPU_OFFLOAD is not None:
        cpu_offload = _resolve_bool_override("FSDP_CPU_OFFLOAD", FSDP_CPU_OFFLOAD, False)
        arguments.append("--fsdp-cpu-offload" if cpu_offload else "--no-fsdp-cpu-offload")
    arguments.append(
        "--fsdp-keep-low-precision-params"
        if runtime.fsdp_keep_low_precision_params
        else "--no-fsdp-keep-low-precision-params"
    )
    arguments.append(
        "--run-after-training" if _effective_run_after_training() else "--no-run-after-training"
    )
    arguments.append(
        "--training-validation" if _training_validation_enabled() else "--no-training-validation"
    )
    arguments.append("--auto-prepare-cache" if AUTO_PREPARE_CACHE else "--no-auto-prepare-cache")
    arguments.append(
        "--allow-code-execution" if ALLOW_CODE_EXECUTION else "--no-allow-code-execution"
    )
    arguments.append(
        "--alpaca-eval-run-judge" if ALPACA_EVAL_RUN_JUDGE else "--no-alpaca-eval-run-judge"
    )
    effective_run_name = os.environ.get(DISTRIBUTED_RUN_NAME_ENV) or RUN_NAME
    if effective_run_name:
        arguments.extend(["--run-name", effective_run_name])
    for stop_token in NATIVE_STOP_TOKENS:
        arguments.extend(["--native-stop-token", stop_token])
    if REFERENCE_CACHE_DIR:
        arguments.extend(["--reference-cache-dir", REFERENCE_CACHE_DIR])
    if ANCHOR_D0_CACHE_PATH:
        arguments.extend(["--anchor-d0-cache-path", ANCHOR_D0_CACHE_PATH])
    if INTERFERENCE_OUTPUT_DIR:
        arguments.extend(["--interference-output-dir", INTERFERENCE_OUTPUT_DIR])
    if LVD_CALIBRATION_CHECKPOINT:
        arguments.extend(["--lvd-calibration-checkpoint", LVD_CALIBRATION_CHECKPOINT])
    if ANCHOR_METADATA_PATH:
        arguments.extend(["--anchor-metadata-path", ANCHOR_METADATA_PATH])
    if not dynamic_geometry and not _label_coverage_filtering():
        if MINIMUM_REFERENCE_CONFIDENCE is not None:
            arguments.extend(["--minimum-reference-confidence", str(MINIMUM_REFERENCE_CONFIDENCE)])
        if MINIMUM_REFERENCE_MARGIN is not None:
            arguments.extend(["--minimum-reference-margin", str(MINIMUM_REFERENCE_MARGIN)])
    if NATIVE_EVALUATION_DIR:
        arguments.extend(["--native-evaluation-dir", NATIVE_EVALUATION_DIR])
    llm_evaluation_enabled = _effective_action() == "native-evaluate" and LLM_EVALUATION
    arguments.append("--llm-evaluation" if llm_evaluation_enabled else "--no-llm-evaluation")
    if llm_evaluation_enabled:
        if not LLM_EVALUATION_TASKS:
            raise ValueError("LLM_EVALUATION_TASKS must contain at least one task")
        for task in LLM_EVALUATION_TASKS:
            arguments.extend(["--llm-evaluation-task", task])
        arguments.extend(["--llm-evaluation-batch-size", str(LLM_EVALUATION_BATCH_SIZE)])
        if LLM_EVALUATION_NUM_FEWSHOT is not None:
            arguments.extend(["--llm-evaluation-num-fewshot", str(LLM_EVALUATION_NUM_FEWSHOT)])
        if LLM_EVALUATION_LIMIT is not None:
            arguments.extend(["--llm-evaluation-limit", str(LLM_EVALUATION_LIMIT)])
        arguments.append(
            "--llm-evaluation-check-integrity"
            if LLM_EVALUATION_CHECK_INTEGRITY
            else "--no-llm-evaluation-check-integrity"
        )
    if DATA_AUDIT_OUTPUT:
        arguments.extend(["--data-audit-output", DATA_AUDIT_OUTPUT])
    for benchmark_seed in BENCHMARK_SEEDS:
        arguments.extend(["--benchmark-seed", str(benchmark_seed)])
    for task in BENCHMARK_GENERAL_TASKS:
        arguments.extend(["--benchmark-general-task", task])
    arguments.append("--benchmark-formal" if BENCHMARK_FORMAL else "--no-benchmark-formal")
    arguments.append("--benchmark-execute" if BENCHMARK_EXECUTE else "--no-benchmark-execute")
    arguments.extend(["--benchmark-num-fewshot", str(BENCHMARK_NUM_FEWSHOT)])
    arguments.append(
        "--benchmark-check-integrity"
        if BENCHMARK_CHECK_INTEGRITY
        else "--no-benchmark-check-integrity"
    )
    if BENCHMARK_GENERAL_LIMIT is not None:
        arguments.extend(["--benchmark-general-limit", str(BENCHMARK_GENERAL_LIMIT)])
    if BENCHMARK_OUTPUT_DIR:
        arguments.extend(["--benchmark-output-dir", BENCHMARK_OUTPUT_DIR])
    effective_action = _effective_action()
    if effective_action in {"test-interference", "test-crossing"}:
        if not EVAL_CHECKPOINT:
            raise ValueError(f"METHOD={METHOD!r} requires EVAL_CHECKPOINT")
        if effective_action == "test-interference" and not ANCHOR_D0_CACHE_PATH:
            raise ValueError("METHOD='test_interference' requires ANCHOR_D0_CACHE_PATH")
        arguments.extend(["--eval-checkpoint", EVAL_CHECKPOINT])
    resume_checkpoint = _effective_resume_checkpoint()
    if effective_action == "train" and resume_checkpoint:
        arguments.extend(["--resume-from-checkpoint", resume_checkpoint])
    elif effective_action == "evaluate":
        if not EVAL_CHECKPOINT:
            raise ValueError("ACTION=evaluate requires EVAL_CHECKPOINT")
        arguments.extend(["--eval-checkpoint", EVAL_CHECKPOINT])
    elif effective_action == "native-evaluate" and EVAL_CHECKPOINT:
        arguments.extend(["--eval-checkpoint", EVAL_CHECKPOINT])
    elif effective_action == "merge-lora":
        if not LORA_ADAPTER_PATH or not MERGED_MODEL_OUTPUT:
            raise ValueError("ACTION=merge-lora requires LORA_ADAPTER_PATH and MERGED_MODEL_OUTPUT")
        arguments.extend(["--adapter", LORA_ADAPTER_PATH, "--output", MERGED_MODEL_OUTPUT])
    elif effective_action not in {
        "train",
        "generate-pilot",
        "audit-model",
        "build-reference-cache",
        "record-anchor-d0",
        _GEOMETRY_REFERENCE_ACTION,
        "filtering-anchor",
        "native-evaluate",
        "audit-data",
        "benchmark",
        "test-interference",
        "test-crossing",
    }:
        raise ValueError(f"unsupported ACTION: {effective_action}")
    return arguments


def _prepare_cache_once() -> None:
    if str(PROJECT_SOURCE) not in sys.path:
        sys.path.insert(0, str(PROJECT_SOURCE))
    from sft_conflict.cache_preparation import (
        ensure_preset_cache,
        prepare_preset_derived_cache,
    )
    from sft_conflict.cli.main import _parser, preset_settings_from_args

    parsed = _parser().parse_args(_run_arguments())
    settings = preset_settings_from_args(parsed)
    ensure_preset_cache(settings, auto_prepare=AUTO_PREPARE_CACHE)
    if _effective_action() in _DISTRIBUTED_TRAINING_ACTIONS:
        prepare_preset_derived_cache(settings)


def _accelerate_command(gpu_ids: tuple[int, ...], port: int) -> list[str]:
    runtime = _resolved_runtime_settings()
    accelerate = PINNED_ENVIRONMENT / "bin" / "accelerate"
    if not accelerate.is_file():
        raise FileNotFoundError(f"Accelerate executable not found: {accelerate}")
    command = [
        str(accelerate),
        "launch",
        "--num_processes",
        str(len(gpu_ids)),
        "--num_machines",
        "1",
        "--mixed_precision",
        (
            "no"
            if MIXED_PRECISION == "fp32" or runtime.fsdp_keep_low_precision_params
            else MIXED_PRECISION
        ),
        "--dynamo_backend",
        "no",
        "--num_cpu_threads_per_process",
        str(_resolve_positive_int_override("CPU_THREADS_PER_PROCESS", CPU_THREADS_PER_PROCESS, 1)),
        "--main_process_port",
        str(port),
    ]
    if DISTRIBUTED_STRATEGY == "ddp":
        command.append("--multi_gpu")
    elif DISTRIBUTED_STRATEGY == "fsdp":
        command.extend(
            [
                "--use_fsdp",
                "--fsdp_sharding_strategy",
                "1",
                "--fsdp_auto_wrap_policy",
                FSDP_AUTO_WRAP_POLICY,
                "--fsdp_use_orig_params",
                str(
                    _effective_training_mode() != "full_ft"
                    or METHOD == "target_plus_anchor_ce_geometry"
                ).lower(),
                "--fsdp_state_dict_type",
                "SHARDED_STATE_DICT",
            ]
        )
        if runtime.gradient_checkpointing:
            command.extend(["--fsdp_activation_checkpointing", "true"])
    command.append(str(Path(__file__).resolve()))
    return command


def _accelerate_filtering_command(gpu_ids: tuple[int, ...], port: int) -> list[str]:
    accelerate = PINNED_ENVIRONMENT / "bin" / "accelerate"
    if not accelerate.is_file():
        raise FileNotFoundError(f"Accelerate executable not found: {accelerate}")
    return [
        str(accelerate),
        "launch",
        "--num_processes",
        str(len(gpu_ids)),
        "--num_machines",
        "1",
        "--mixed_precision",
        "no",
        "--dynamo_backend",
        "no",
        "--num_cpu_threads_per_process",
        str(_resolve_positive_int_override("CPU_THREADS_PER_PROCESS", CPU_THREADS_PER_PROCESS, 1)),
        "--main_process_port",
        str(port),
        "--multi_gpu",
        str(Path(__file__).resolve()),
    ]


def _reusable_filter_scores_available() -> bool:
    if _effective_action() != "filtering-anchor":
        return False
    if str(PROJECT_SOURCE) not in sys.path:
        sys.path.insert(0, str(PROJECT_SOURCE))
    from sft_conflict.data.anchor_filtering import (
        anchor_filter_score_dir,
        reusable_filter_score_paths,
    )
    from sft_conflict.quick_run import ANCHOR_DATASET_PRESETS, MODEL_PRESETS

    model = MODEL_PRESETS[MODEL]
    anchor = ANCHOR_DATASET_PRESETS.get(ANCHOR_DATASET)
    if model.model_id is None or anchor is None or anchor.path is None:
        return False
    score_dir = anchor_filter_score_dir(ARTIFACT_ROOT, MODEL, MAX_LENGTH)
    return (
        reusable_filter_score_paths(
            score_dir=score_dir,
            model_name=MODEL,
            model_id=model.model_id,
            model_revision=model.revision,
            max_length=MAX_LENGTH,
            source_dataset_id=anchor.path,
            source_dataset_revision=anchor.revision,
        )
        is not None
    )


def _launch_distributed_if_needed(gpu_ids: tuple[int, ...]) -> None:
    if _label_coverage_filtering():
        return
    if os.environ.get(DISTRIBUTED_CHILD_ENV) == "1":
        return
    if DISTRIBUTED_STRATEGY == "single":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])
        return
    if _effective_action() == "filtering-anchor":
        if _reusable_filter_scores_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])
            print(
                "Reusable Anchor correctness scores found; processing the requested "
                f"threshold={FILTER_THRESHOLD} in one process.",
                flush=True,
            )
            return
        if len(gpu_ids) == 1:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])
            return
        shared_run_name = _shared_run_name()
        os.environ[DISTRIBUTED_RUN_NAME_ENV] = shared_run_name
        _prepare_cache_once()
        environment = os.environ.copy()
        environment[DISTRIBUTED_CHILD_ENV] = "1"
        environment[DISTRIBUTED_RUN_NAME_ENV] = shared_run_name
        environment["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu_id) for gpu_id in gpu_ids)
        command = _accelerate_filtering_command(gpu_ids, _free_local_port())
        print(
            "Launching distributed Anchor filtering: "
            f"physical_gpus={gpu_ids}, processes={len(gpu_ids)}, run_name={shared_run_name}",
            flush=True,
        )
        os.execve(command[0], command, environment)
    if _effective_action() not in _DISTRIBUTED_TRAINING_ACTIONS:
        selected = (
            gpu_ids[:1]
            if _effective_action() in {"record-anchor-d0", _GEOMETRY_REFERENCE_ACTION}
            else gpu_ids
        )
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu_id) for gpu_id in selected)
        return

    shared_run_name = _shared_run_name()
    os.environ[DISTRIBUTED_RUN_NAME_ENV] = shared_run_name
    _prepare_cache_once()
    port = _free_local_port()
    environment = os.environ.copy()
    environment[DISTRIBUTED_CHILD_ENV] = "1"
    environment[DISTRIBUTED_RUN_NAME_ENV] = shared_run_name
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu_id) for gpu_id in gpu_ids)
    command = _accelerate_command(gpu_ids, port)
    print(
        "Launching distributed training: "
        f"strategy={DISTRIBUTED_STRATEGY}, physical_gpus={gpu_ids}, "
        f"processes={len(gpu_ids)}, port={port}, run_name={shared_run_name}",
        flush=True,
    )
    os.execve(command[0], command, environment)


def main() -> int:
    _configure_cuda_allocator()
    _configure_hugging_face_environment()
    _ensure_training_interpreter()
    if os.environ.get(DISTRIBUTED_CHILD_ENV) != "1":
        _print_resolved_runtime_settings(_resolved_runtime_settings())
    if not _label_coverage_filtering():
        gpu_ids = _validated_gpu_ids()
        _launch_distributed_if_needed(gpu_ids)
    if str(PROJECT_SOURCE) not in sys.path:
        sys.path.insert(0, str(PROJECT_SOURCE))
    from sft_conflict.cli.main import main as cli_main

    return cli_main(_run_arguments())


if __name__ == "__main__":
    raise SystemExit(main())
