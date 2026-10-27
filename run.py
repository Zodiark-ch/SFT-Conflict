# pyright: reportMissingImports=false
"""Command-line entry point for every experiment step (AEGIS and baselines).

Typical workflow (see README.md for every task and option):

    # 1. Build the label-coverage pervasive dataset from SMART FLAN (CPU only).
    python run.py --method filtering_anchor --model mistral_7b
    # 2. Record the Base model's geometry on that pervasive dataset (first GPU).
    python run.py --method prepare_anchor_geometry_reference --model mistral_7b
    # 3. Train with AEGIS on a target task (4-GPU FSDP by default).
    python run.py --method target_plus_anchor_ce_geometry --model mistral_7b --target rte
    # 4. Evaluate the exported checkpoint on the target task.
    python run.py --action native-evaluate --model mistral_7b --target rte \\
        --checkpoint <artifact-root>/checkpoints/<run-name>/final

Every command-line option overrides one value of DEFAULT SETTINGS below; running
without options uses these defaults unchanged (handy for an IDE "Run" button).
``--set NAME=VALUE`` overrides any other setting in that block, and ``--dry-run``
prints the resolved step and command without launching anything.

When ``<artifact-root>/envs/train`` exists, the script re-executes itself inside that
pinned environment; multi-GPU training launches its own Accelerate/FSDP workers.
"""

from __future__ import annotations

import argparse
import ast
import os
import shlex
import shutil
import socket
import subprocess
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

REPOSITORY_ROOT = Path(__file__).resolve().parent

# ============================ DEFAULT SETTINGS ==============================
# Each value can be overridden from the command line (python run.py --help).

# Plain actions. Pipeline steps and analyses are selected with METHOD instead:
# filtering_anchor, create_anchor_dataset, prepare_anchor_geometry_reference,
# record_anchor_d0, generate_pilot, test_crossing and test_interference set their own
# action. train | native-evaluate | evaluate | audit-model | audit-data | benchmark |
# build-reference-cache | merge-lora
ACTION = "train"

# tiny_qwen2 | qwen25_0.5b | qwen25_3b | qwen25_7b | mistral_7b
MODEL = "mistral_7b"

# Target (SFT) task.
#   Training presets: cola | sst2 | mrpc | stsb | qqp | mnli | qnli | rte (GLUE),
#   gsm8k, codealpaca | code_bagel_hermes | infinity_instruct_code (code),
#   logiqa2 | evol_instruct (logic / instruction following),
#   tiny_gsm8k | tiny_sst2 | tiny_instruction (CPU smoke tests).
#   Native evaluation presets: every GLUE task, gsm8k, logiqa2, humaneval,
#   codexglue_python, codexglue_java and alpaca_eval.
TARGET_DATASET = "rte"

# Pervasive dataset (called "Anchor" in the code).
#   auto: smart_flan_<MODEL>_label_coverage_v1 for methods that use pervasive data,
#         none for target_only and native evaluation.
#   none | tiny_general | smart_flan (raw SMART FLAN mixture) |
#   smart_flan_<model>_label_coverage_v1 (METHOD="filtering_anchor") |
#   flan2022_<model>_task_vocab_cover_v1 (METHOD="create_anchor_dataset") |
#   smart_flan_<model> (legacy correctness filter, FILTERING_POLICY="legacy_quality_v1")
ANCHOR_DATASET = "auto"

# Training objectives:
#   target_plus_anchor_ce_geometry  AEGIS: Target CE + pervasive CE + geometry term
#   target_only | target_plus_anchor_ce | target_plus_anchor_kl  baselines
#   target_plus_anchor_lvd | target_plus_anchor_ce_lvd | target_plus_null_lvd  legacy LVD
# Pipeline steps: filtering_anchor | create_anchor_dataset |
#   prepare_anchor_geometry_reference | record_anchor_d0 | generate_pilot
# Analyses: test_crossing | test_interference
METHOD = "target_plus_anchor_ce_geometry"

TRAINING_MODE = "full_ft"  # full_ft | body_ft_frozen_head | lora (AEGIS requires full_ft)

# Resume training from a checkpoint-N directory (model, optimizer, scheduler, RNG);
# final/ directories hold only the exported model and cannot be resumed.
RESUME_CHECKPOINT = None
# Checkpoint for native-evaluate / evaluate / test_crossing / test_interference
# (usually <artifact-root>/checkpoints/<run-name>/final). None evaluates the Base model.
EVAL_CHECKPOINT = None
# After native evaluation, run the lm-evaluation-harness pervasive suite below.
LLM_EVALUATION = False

# ------------------ TASK-DEPENDENT AUTOMATIC SETTINGS -----------------------
# None selects automatically; True/False or a positive integer overrides.
GRADIENT_CHECKPOINTING: bool | None = None  # auto: on for 3B/7B training
FSDP_SYNC_EACH_BATCH: bool | None = None  # auto: on for FSDP training
# None offloads FSDP parameters and AdamW states to CPU for models with >= 6B parameters.
# False keeps them on GPU: the paper setting (Mistral-7B, FP32, 4x A100 40GB, ~37 GB/GPU).
FSDP_CPU_OFFLOAD: bool | None = False
FSDP_KEEP_LOW_PRECISION_PARAMS: bool | None = None  # auto: off
EVALUATION_MAX_SAMPLES: int | None = None  # auto: full split; 256 for MAX_STEPS>0 smoke runs
NATIVE_MAX_NEW_TOKENS: int | None = None  # auto: gsm8k 256, humaneval 512, alpaca_eval 2048

# --------------------------- FIXED DEFAULTS ---------------------------------
SUPERVISION = "response_only"
MIXED_PRECISION = "fp32"  # fp32 | bf16 | fp16
OPTIMIZER = "adamw_torch"  # adamw_torch | adamw_bnb_8bit | adafactor

# Training: single uses one GPU; ddp/fsdp use every ID listed here.
# Native evaluation, reference preparation and analyses use the first listed GPU.
DISTRIBUTED_STRATEGY = "fsdp"  # single | ddp | fsdp
GPU_IDS = (0, 1, 2, 3)
CPU_THREADS_PER_PROCESS = 1  # Positive integer; affects CPU offload/optimizer throughput.
FSDP_AUTO_WRAP_POLICY = "transformer_based_wrap"
MAIN_PROCESS_PORT = None  # None chooses a free local port automatically.
CUDA_ALLOCATOR_CONF = "expandable_segments:True"

MAX_STEPS = -1  # Use a positive value only for a bounded smoke run.
# METHOD="generate_pilot" (legacy LVD); joint option: "target_plus_anchor_ce_lvd".
PILOT_METHOD = "target_plus_anchor_lvd"
PILOT_STEPS = 10  # Optimizer updates; the pilot retains the full training scheduler.
PILOT_HEALTH_INTERVAL = 5
NUM_TRAIN_EPOCHS = 3.0
# Extra restart checkpoints every N optimizer steps (epoch checkpoints and best-model
# selection are unchanged). None disables them.
RECOVERY_SAVE_STEPS: int | None = 100
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06
MAX_LENGTH = 512
TARGET_MAX_SAMPLES = None  # None: full split (code presets use their 20k paper budgets)
TARGET_VALIDATION_MAX_SAMPLES = None  # None uses the complete derived 10% split.
ANCHOR_MAX_SAMPLES = 40_000  # Caps only the raw smart_flan mixture.
ANCHOR_METADATA_PATH = None  # JSONL keyed by sample_id or source_index.
# label_coverage_v1: CPU label-vocabulary coverage of SMART FLAN (no model inference);
# legacy_quality_v1: Base-correctness filter + set cover (multi-GPU inference).
FILTERING_POLICY = "label_coverage_v1"
FILTER_THRESHOLD = 0.6  # legacy_quality_v1 only.
FILTERING_BATCH_SIZE = 4  # legacy_quality_v1 only.
# METHOD="create_anchor_dataset": task round-robin label-vocabulary cover over the five
# FLAN 2022 submixtures used by SMART. Selection is deterministic for the seed; the
# accuracy report runs the Base model on the first GPU in GPU_IDS after selection.
ANCHOR_CREATION_SEED = 42
ANCHOR_CREATION_WORKERS = 32  # Tokenization processes; 0 tokenizes in-process.
ANCHOR_CREATION_SCORING_BATCH_SIZE = 4
ANCHOR_CREATION_COMPUTE_ACCURACY = True  # Base teacher-forced accuracy on the result.
ANCHOR_CREATION_SUBMIXTURES = ("flan2021", "t0", "niv2", "cot", "dialog")
ANCHOR_CREATION_MAX_SAMPLES_PER_TASK = None  # Smoke-test cap per task; None traverses all.
# True: a task's first turn keeps its first drawn supervised sample even without a new
# token, so every task contributes at least one sample; later turns require a new label
# token. False: the pure new-token rule from the first turn on.
ANCHOR_CREATION_GUARANTEE_TASK_COVERAGE = True
EVALUATION_BATCH_SIZE = 4  # Native eval; caps concurrent HumanEval sequences.
# Native LogiQA2 scores 4 options per question (4x rows of FP32 152k-vocab logits);
# qwen25_7b at batch 4 OOMs on one A100 40GB, so only that case is capped here.
QWEN25_7B_LOGIQA2_EVALUATION_BATCH_SIZE = 2
TARGET_BATCH_SIZE = 1
ANCHOR_BATCH_SIZE = 1
ANCHOR_SAMPLING_STRATEGY = "mixture_proportional"  # task_balanced | mixture_proportional | random
GRADIENT_ACCUMULATION_STEPS = 4
RUN_AFTER_TRAINING = True
# Validation metric that picks the checkpoint loaded at the end of training:
#   native_accuracy: task-native candidate-likelihood accuracy on the training validation
#                    split (fixed-candidate classification tasks: GLUE except STS-B);
#   eval_loss: token-level validation loss;
#   auto: native_accuracy where supported, eval_loss otherwise.
BEST_MODEL_METRIC = "auto"

LORA_RANK = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
ANCHOR_WEIGHT = 0.2
KL_TEMPERATURE = 1.0

SEED = 42
# Models, Hugging Face caches, datasets, checkpoints and results live outside the Git
# repository: $SFT_CONFLICT_ARTIFACT_ROOT, or ../artifacts next to this repository.
ARTIFACT_ROOT = os.environ.get("SFT_CONFLICT_ARTIFACT_ROOT") or str(
    REPOSITORY_ROOT.parent / "artifacts"
)
RUN_NAME = None  # None creates a unique run name each time.
# Download missing models/datasets into <artifact-root>/hf_cache before running.
AUTO_PREPARE_CACHE = True

# Legacy LVD reference-cache settings (target_plus_anchor_lvd / record_anchor_d0).
REFERENCE_CACHE_DIR = None
ANCHOR_POSITION_STRATEGY = "sample_k_positions"
ANCHOR_POSITIONS_PER_SAMPLE = 4
REFERENCE_SAMPLES_PER_SHARD = 4
REFERENCE_STORAGE_DTYPE = "float32"
REFERENCE_FILTER_MODE = "base_correct"
MINIMUM_REFERENCE_CONFIDENCE = None
MINIMUM_REFERENCE_MARGIN = None

# Legacy weighted-pressure LVD settings (configs/method/weighted_pressure_lvd.yaml).
LVD_DIRECTION = 512  # Current-logit Top-K competitors, excluding the base reference token.
LVD_DYNAMIC_POSITIONS = True  # True: all eligible response positions; False: sample fixed K.
LVD_CONFIG_PATH = str(REPOSITORY_ROOT / "configs/method/weighted_pressure_lvd.yaml")
adapted_weight = False  # False: alpha=1/K; True: normalized inverse-d0 capacity weights.
# ACTION="audit-model": real hidden states checked against every planar LVD cell boundary.
LVD_AUDIT_MAX_POSITIONS = 8
LVD_AUDIT_VOCABULARY_CHUNK_SIZE = 256
# METHOD="test_interference": bounded matrix-multiply chunks and optional artifact directory.
INTERFERENCE_TOKEN_CHUNK_SIZE = 256
INTERFERENCE_POSITION_CHUNK_SIZE = 1024
INTERFERENCE_OUTPUT_DIR = None
# test_interference requires an explicit all_reference d0 cache (METHOD="record_anchor_d0");
# test_crossing ignores it. For legacy LVD training, None derives the matching path.
ANCHOR_D0_CACHE_PATH = None
# None discovers an accepted compatible pilot; set a path for an explicit export.
LVD_CALIBRATION_CHECKPOINT = None

# AEGIS (METHOD="target_plus_anchor_ce_geometry") objective, calibration and gradient
# routing; see the comments in the YAML. Independent of the legacy LVD options above.
ANCHOR_GEOMETRY_CONFIG_PATH = str(
    REPOSITORY_ROOT / "configs/method/anchor_ce_geometry_relative.yaml"
)
# Base-geometry reference written by METHOD="prepare_anchor_geometry_reference".
# None derives the same path (from model and pervasive dataset) for preparation and training.
ANCHOR_GEOMETRY_REFERENCE_PATH: str | None = None

# Native evaluation and leakage-audit settings.
NATIVE_STOP_TOKENS = ()
NATIVE_EVALUATION_DIR = None
# Pervasive-capability suite run by lm-evaluation-harness 0.4.9 after native evaluation.
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
# Isolated lm-eval interpreter and lm-evaluation-harness v0.4.9 checkout.
# None: <artifact-root>/envs/lm-eval/bin/python and <artifact-root>/lm-harness.
LM_EVAL_PYTHON: str | None = None
LM_EVAL_HARNESS_PATH: str | None = None
ALPACA_EVAL_RUN_JUDGE = False  # True spends API credits; export OPENAI_API_KEY in the shell.
# chatgpt_fn reproduces DARE; weighted_alpaca_eval_gpt4_turbo is current AlpacaEval 2.0.
ALPACA_EVAL_ANNOTATORS_CONFIG = "chatgpt_fn"
# HumanEval executes model-generated Python (sandboxed, but use a disposable machine).
ALLOW_CODE_EXECUTION = False
HUMANEVAL_SAMPLES_PER_PROBLEM = 20
HUMANEVAL_TEMPERATURE = 0.2
HUMANEVAL_TOP_P = 0.95
HUMANEVAL_TIMEOUT_SECONDS = 3.0
HUMANEVAL_EXECUTION_WORKERS = 16
NEAR_DUPLICATE_THRESHOLD = 0.85
DATA_AUDIT_OUTPUT = None

# ACTION="benchmark": six-condition benchmark. False writes the full plan without running it.
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

# ACTION="merge-lora" only.
LORA_ADAPTER_PATH = None
MERGED_MODEL_OUTPUT = None
# ============================================================================

# Names that --set NAME=VALUE may override (everything in DEFAULT SETTINGS).
_SETTING_NAMES = frozenset(
    name
    for name in tuple(globals())
    if (name.isupper() and name != "REPOSITORY_ROOT") or name == "adapted_weight"
)

TRAINING_ENVIRONMENT_ENV = "SFT_CONFLICT_TRAIN_ENV"


def _default_training_environment(artifact_root: str) -> Path:
    explicit = os.environ.get(TRAINING_ENVIRONMENT_ENV)
    if explicit:
        return Path(explicit).expanduser()
    return Path(artifact_root).expanduser() / "envs" / "train"


# Pinned training environment; used when it exists, otherwise the current interpreter.
PINNED_ENVIRONMENT = _default_training_environment(ARTIFACT_ROOT)
PINNED_PYTHON = PINNED_ENVIRONMENT / "bin" / "python"
PROJECT_SOURCE = REPOSITORY_ROOT / "src"
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
# Fixed-candidate classification targets whose native accuracy can select checkpoints.
_NATIVE_SELECTION_TARGETS = frozenset({"cola", "sst2", "mrpc", "qqp", "mnli", "qnli", "rte"})
# Command-line arguments forwarded to the pinned-environment re-exec and to every
# Accelerate worker, so all processes resolve the same settings.
_FORWARDED_ARGUMENTS: tuple[str, ...] = ()


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
        "create_anchor_dataset": "create-anchor-dataset",
        "test_interference": "test-interference",
        "test_crossing": "test-crossing",
    }.get(METHOD, ACTION)


def _effective_method() -> str:
    if _effective_action() == "native-evaluate":
        # Native evaluation only needs the model, the checkpoint and the target task.
        return "target_only"
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
    action = _effective_action()
    if action in {"filtering-anchor", "create-anchor-dataset"}:
        return "smart_flan"
    if action == "native-evaluate":
        return "none"
    if ANCHOR_DATASET != "auto":
        return ANCHOR_DATASET
    if _effective_method() == "target_only":
        return "none"
    return f"smart_flan_{MODEL}_label_coverage_v1"


def _label_coverage_filtering() -> bool:
    return _effective_action() == "filtering-anchor" and FILTERING_POLICY == "label_coverage_v1"


def _effective_training_mode() -> str:
    action = _effective_action()
    if (
        _effective_method()
        in {"prepare_anchor_geometry_reference", "target_plus_anchor_ce_geometry"}
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


def _effective_best_model_metric() -> str:
    if BEST_MODEL_METRIC == "auto":
        supported = TARGET_DATASET in _NATIVE_SELECTION_TARGETS and _training_validation_enabled()
        return "native_accuracy" if supported else "eval_loss"
    if BEST_MODEL_METRIC not in {"eval_loss", "native_accuracy"}:
        raise ValueError("BEST_MODEL_METRIC must be auto, eval_loss or native_accuracy")
    return BEST_MODEL_METRIC


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
            [str(PINNED_PYTHON), str(Path(__file__).resolve()), *_FORWARDED_ARGUMENTS],
            environment,
        )


def _accelerate_executable() -> Path:
    """Accelerate from the pinned environment, else from the running interpreter."""
    for environment in (PINNED_ENVIRONMENT, Path(sys.prefix)):
        candidate = environment / "bin" / "accelerate"
        if candidate.is_file():
            return candidate
    found = shutil.which("accelerate")
    if found:
        return Path(found)
    raise FileNotFoundError(
        f"Accelerate executable not found in {PINNED_ENVIRONMENT / 'bin'} or the current "
        "environment; install the training requirements (pip install -e .)"
    )


def _lm_eval_python() -> str:
    if LM_EVAL_PYTHON:
        return str(LM_EVAL_PYTHON)
    return str(Path(ARTIFACT_ROOT).expanduser() / "envs" / "lm-eval" / "bin" / "python")


def _lm_eval_harness_path() -> str:
    if LM_EVAL_HARNESS_PATH:
        return str(LM_EVAL_HARNESS_PATH)
    return str(Path(ARTIFACT_ROOT).expanduser() / "lm-harness")


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
        "--lm-eval-python",
        _lm_eval_python(),
        "--lm-eval-harness-path",
        _lm_eval_harness_path(),
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
    if _effective_action() == "create-anchor-dataset":
        if type(ANCHOR_CREATION_SEED) is not int:
            raise ValueError("ANCHOR_CREATION_SEED must be an integer")
        if type(ANCHOR_CREATION_WORKERS) is not int or ANCHOR_CREATION_WORKERS < 0:
            raise ValueError("ANCHOR_CREATION_WORKERS must be a non-negative integer")
        if not ANCHOR_CREATION_SUBMIXTURES:
            raise ValueError("ANCHOR_CREATION_SUBMIXTURES must list at least one submixture")
        arguments.extend(
            [
                "--anchor-creation-seed",
                str(ANCHOR_CREATION_SEED),
                "--anchor-creation-workers",
                str(ANCHOR_CREATION_WORKERS),
                "--anchor-creation-scoring-batch-size",
                str(
                    _resolve_positive_int_override(
                        "ANCHOR_CREATION_SCORING_BATCH_SIZE",
                        ANCHOR_CREATION_SCORING_BATCH_SIZE,
                        4,
                    )
                ),
                "--anchor-creation-compute-accuracy"
                if ANCHOR_CREATION_COMPUTE_ACCURACY
                else "--no-anchor-creation-compute-accuracy",
                "--anchor-creation-guarantee-task-coverage"
                if ANCHOR_CREATION_GUARANTEE_TASK_COVERAGE
                else "--no-anchor-creation-guarantee-task-coverage",
            ]
        )
        for submixture in ANCHOR_CREATION_SUBMIXTURES:
            arguments.extend(["--anchor-creation-submixture", str(submixture)])
        if ANCHOR_CREATION_MAX_SAMPLES_PER_TASK is not None:
            arguments.extend(
                [
                    "--anchor-creation-max-samples-per-task",
                    str(
                        _resolve_positive_int_override(
                            "ANCHOR_CREATION_MAX_SAMPLES_PER_TASK",
                            ANCHOR_CREATION_MAX_SAMPLES_PER_TASK,
                            1,
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
    arguments.extend(["--best-model-metric", _effective_best_model_metric()])
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
            raise ValueError(f"METHOD={METHOD!r} requires EVAL_CHECKPOINT (--checkpoint)")
        if effective_action == "test-interference" and not ANCHOR_D0_CACHE_PATH:
            raise ValueError(
                "METHOD='test_interference' requires ANCHOR_D0_CACHE_PATH (--anchor-d0-cache-path)"
            )
        arguments.extend(["--eval-checkpoint", EVAL_CHECKPOINT])
    resume_checkpoint = _effective_resume_checkpoint()
    if effective_action == "train" and resume_checkpoint:
        arguments.extend(["--resume-from-checkpoint", resume_checkpoint])
    elif effective_action == "evaluate":
        if not EVAL_CHECKPOINT:
            raise ValueError("ACTION=evaluate requires EVAL_CHECKPOINT (--checkpoint)")
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
        "create-anchor-dataset",
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
    accelerate = _accelerate_executable()
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
    command.extend(_FORWARDED_ARGUMENTS)
    return command


def _accelerate_filtering_command(gpu_ids: tuple[int, ...], port: int) -> list[str]:
    accelerate = _accelerate_executable()
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
        *_FORWARDED_ARGUMENTS,
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
    anchor = ANCHOR_DATASET_PRESETS.get(_effective_anchor_dataset())
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
            if _effective_action()
            in {"record-anchor-d0", _GEOMETRY_REFERENCE_ACTION, "create-anchor-dataset"}
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


# ------------------------------ COMMAND LINE --------------------------------
_ACTIONS = (
    "train",
    "native-evaluate",
    "evaluate",
    "audit-model",
    "audit-data",
    "benchmark",
    "build-reference-cache",
    "merge-lora",
)
_METHODS = (
    "target_plus_anchor_ce_geometry",
    "target_only",
    "target_plus_anchor_ce",
    "target_plus_anchor_kl",
    "filtering_anchor",
    "create_anchor_dataset",
    "prepare_anchor_geometry_reference",
    "test_crossing",
    "test_interference",
    "record_anchor_d0",
    "generate_pilot",
    "target_plus_anchor_lvd",
    "target_plus_anchor_ce_lvd",
    "target_plus_null_lvd",
)
_MODELS = ("tiny_qwen2", "qwen25_0.5b", "qwen25_3b", "qwen25_7b", "mistral_7b")


class _Option(NamedTuple):
    setting: str
    flags: tuple[str, ...]
    kind: str
    help: str
    choices: tuple[str, ...] | None = None


_STRATEGIES = ("single", "ddp", "fsdp")
_SUBMIXTURES = ("flan2021", "t0", "niv2", "cot", "dialog")
_SAMPLING = ("mixture_proportional", "task_balanced", "random")
_JUDGES = ("chatgpt_fn", "weighted_alpaca_eval_gpt4_turbo")
# (setting, flags, kind, help[, choices]); kinds are parsed by _OPTION_TYPES below.
# fmt: off
_OPTION_TABLE = (
    ("experiment", (
        ("ACTION", "--action", "str", "plain action; pipeline steps use --method", _ACTIONS),
        ("METHOD", "--method", "str", "training objective or pipeline step", _METHODS),
        ("MODEL", "--model", "str", "base model preset", _MODELS),
        ("TARGET_DATASET", "--target --target-dataset", "str",
         "target (SFT) task, e.g. rte, gsm8k, codealpaca, humaneval"),
        ("ANCHOR_DATASET", "--anchor --anchor-dataset", "str",
         "pervasive (Anchor) dataset; auto = smart_flan_<model>_label_coverage_v1"),
        ("TRAINING_MODE", "--training-mode", "str", "trained parameters (AEGIS: full_ft)",
         ("full_ft", "body_ft_frozen_head", "lora")),
        ("EVAL_CHECKPOINT", "--checkpoint --eval-checkpoint", "optional_path",
         "checkpoint (final/ directory) to evaluate or analyse; omit for the Base model"),
        ("RESUME_CHECKPOINT", "--resume --resume-checkpoint", "optional_path",
         "resume training from a checkpoint-N directory"),
        ("LLM_EVALUATION", "--llm-evaluation", "bool",
         "after native evaluation, run the lm-evaluation-harness pervasive suite"),
        ("SEED", "--seed", "int", "training and data-order seed"),
        ("RUN_NAME", "--run-name", "optional_str", "fixed run name"),
        ("ARTIFACT_ROOT", "--artifact-root", "path",
         "directory for caches, datasets, checkpoints and results"),
    )),
    ("hardware and parallelism", (
        ("GPU_IDS", "--gpus --gpu-ids", "gpus", "physical GPU ids, e.g. --gpus 0 1 2 3"),
        ("DISTRIBUTED_STRATEGY", "--distributed-strategy", "str", "training parallelism",
         _STRATEGIES),
        ("MIXED_PRECISION", "--mixed-precision", "str", "training precision",
         ("fp32", "bf16", "fp16")),
        ("FSDP_CPU_OFFLOAD", "--fsdp-cpu-offload", "bool",
         "offload FSDP parameters and optimizer states to CPU"),
        ("GRADIENT_CHECKPOINTING", "--gradient-checkpointing", "bool",
         "activation checkpointing (None: on for 3B/7B training)"),
        ("CPU_THREADS_PER_PROCESS", "--cpu-threads-per-process", "positive_int",
         "CPU threads per training process"),
    )),
    ("training", (
        ("NUM_TRAIN_EPOCHS", "--epochs --num-train-epochs", "float", "training epochs"),
        ("MAX_STEPS", "--max-steps", "int", "stop after N optimizer steps (-1: all epochs)"),
        ("LEARNING_RATE", "--lr --learning-rate", "float", "peak learning rate"),
        ("WEIGHT_DECAY", "--weight-decay", "float", "AdamW weight decay"),
        ("WARMUP_RATIO", "--warmup-ratio", "float", "linear warmup ratio"),
        ("MAX_LENGTH", "--max-length", "positive_int", "maximum sequence length"),
        ("TARGET_BATCH_SIZE", "--target-batch-size", "positive_int",
         "target sequences per GPU micro-batch"),
        ("ANCHOR_BATCH_SIZE", "--anchor-batch-size", "positive_int",
         "pervasive sequences per GPU micro-batch"),
        ("GRADIENT_ACCUMULATION_STEPS", "--gradient-accumulation-steps", "positive_int",
         "micro-batches per optimizer step"),
        ("ANCHOR_WEIGHT", "--anchor-weight", "float", "weight of the pervasive CE/KL term"),
        ("ANCHOR_SAMPLING_STRATEGY", "--anchor-sampling-strategy", "str",
         "how pervasive samples are drawn", _SAMPLING),
        ("TARGET_MAX_SAMPLES", "--target-max-samples", "optional_int",
         "cap on target training samples (None: preset budget)"),
        ("TARGET_VALIDATION_MAX_SAMPLES", "--target-validation-max-samples", "optional_int",
         "cap on the derived validation split"),
        ("ANCHOR_MAX_SAMPLES", "--anchor-max-samples", "optional_int",
         "cap on the raw smart_flan mixture"),
        ("BEST_MODEL_METRIC", "--best-model-metric", "str",
         "validation metric that selects the exported checkpoint",
         ("auto", "native_accuracy", "eval_loss")),
        ("RECOVERY_SAVE_STEPS", "--recovery-save-steps", "optional_int",
         "extra restart checkpoint every N steps (None: off)"),
        ("RUN_AFTER_TRAINING", "--run-after-training", "bool",
         "evaluate the validation and test splits after training"),
        ("OPTIMIZER", "--optimizer", "str", "optimizer",
         ("adamw_torch", "adamw_bnb_8bit", "adafactor")),
        ("KL_TEMPERATURE", "--kl-temperature", "float", "pervasive KL temperature"),
        ("LORA_RANK", "--lora-rank", "positive_int", "LoRA rank"),
        ("LORA_ALPHA", "--lora-alpha", "positive_int", "LoRA alpha"),
        ("LORA_DROPOUT", "--lora-dropout", "float", "LoRA dropout"),
    )),
    ("AEGIS", (
        ("ANCHOR_GEOMETRY_CONFIG_PATH", "--geometry-config", "path", "AEGIS objective YAML"),
        ("ANCHOR_GEOMETRY_REFERENCE_PATH", "--geometry-reference", "optional_path",
         "Base-geometry reference directory (None: derived from model and dataset)"),
    )),
    ("pervasive datasets", (
        ("FILTERING_POLICY", "--filtering-policy", "str", "filtering_anchor policy",
         ("label_coverage_v1", "legacy_quality_v1")),
        ("FILTER_THRESHOLD", "--filter-threshold", "float", "legacy_quality_v1 threshold"),
        ("FILTERING_BATCH_SIZE", "--filtering-batch-size", "positive_int",
         "legacy_quality_v1 inference batch size"),
        ("ANCHOR_CREATION_SEED", "--creation-seed", "int", "create_anchor_dataset seed"),
        ("ANCHOR_CREATION_WORKERS", "--creation-workers", "int",
         "tokenization processes (0: in-process)"),
        ("ANCHOR_CREATION_SUBMIXTURES", "--creation-submixtures", "list",
         "FLAN 2022 submixtures to traverse", _SUBMIXTURES),
        ("ANCHOR_CREATION_MAX_SAMPLES_PER_TASK", "--creation-max-samples-per-task",
         "optional_int", "smoke-test cap per task (None: traverse all)"),
        ("ANCHOR_CREATION_GUARANTEE_TASK_COVERAGE", "--creation-guarantee-task-coverage",
         "bool", "keep one sample of every task on its first turn"),
        ("ANCHOR_CREATION_COMPUTE_ACCURACY", "--creation-compute-accuracy", "bool",
         "report the Base model accuracy on the created dataset"),
        ("ANCHOR_CREATION_SCORING_BATCH_SIZE", "--creation-scoring-batch-size",
         "positive_int", "batch size of the accuracy report"),
    )),
    ("evaluation", (
        ("EVALUATION_BATCH_SIZE", "--evaluation-batch-size", "positive_int",
         "native evaluation batch size"),
        ("EVALUATION_MAX_SAMPLES", "--evaluation-max-samples", "optional_int",
         "cap on evaluation samples (None: full split)"),
        ("NATIVE_MAX_NEW_TOKENS", "--native-max-new-tokens", "optional_int",
         "generation length (None: task default)"),
        ("NATIVE_EVALUATION_DIR", "--native-evaluation-dir", "optional_path",
         "output directory of native evaluation"),
        ("LLM_EVALUATION_TASKS", "--llm-evaluation-tasks", "list",
         "lm-evaluation-harness tasks of the pervasive suite"),
        ("LLM_EVALUATION_LIMIT", "--llm-evaluation-limit", "optional_float",
         "examples per task for a smoke run (None: all)"),
        ("LLM_EVALUATION_NUM_FEWSHOT", "--llm-evaluation-num-fewshot", "optional_int",
         "few-shot examples (None: task default)"),
        ("LLM_EVALUATION_BATCH_SIZE", "--llm-evaluation-batch-size", "str",
         "lm-evaluation-harness batch size"),
        ("LM_EVAL_PYTHON", "--lm-eval-python", "optional_path",
         "python of the isolated lm-eval environment"),
        ("LM_EVAL_HARNESS_PATH", "--lm-eval-harness-path", "optional_path",
         "lm-evaluation-harness v0.4.9 checkout"),
        ("ALLOW_CODE_EXECUTION", "--allow-code-execution", "bool",
         "execute generated Python for HumanEval (use a disposable machine)"),
        ("HUMANEVAL_SAMPLES_PER_PROBLEM", "--humaneval-samples-per-problem", "positive_int",
         "HumanEval samples per problem"),
        ("ALPACA_EVAL_RUN_JUDGE", "--alpaca-eval-run-judge", "bool",
         "run the paid AlpacaEval judge (needs OPENAI_API_KEY)"),
        ("ALPACA_EVAL_ANNOTATORS_CONFIG", "--alpaca-eval-annotators-config", "str",
         "AlpacaEval judge configuration", _JUDGES),
        ("ANCHOR_D0_CACHE_PATH", "--anchor-d0-cache-path", "optional_path",
         "d0 cache for test_interference (METHOD record_anchor_d0)"),
        ("INTERFERENCE_OUTPUT_DIR", "--interference-output-dir", "optional_path",
         "output directory of test_interference"),
    )),
    ("other", (
        ("AUTO_PREPARE_CACHE", "--auto-prepare-cache", "bool",
         "download missing models and datasets into <artifact-root>/hf_cache"),
    )),
)
# fmt: on
_OPTION_GROUPS = tuple(
    (
        title,
        tuple(
            _Option(setting, tuple(flags.split()), kind, help_text, *rest)
            for setting, flags, kind, help_text, *rest in rows
        ),
    )
    for title, rows in _OPTION_TABLE
)
_OPTIONS = tuple(option for _, options in _OPTION_GROUPS for option in options)


def _none_or(converter):
    def parse(text: str):
        if text.strip().lower() in {"none", "null", "auto"}:
            return None
        return converter(text)

    parse.__name__ = getattr(converter, "__name__", "value")
    return parse


def _positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {text!r}")
    return value


def _absolute_path(text: str) -> str:
    return os.path.abspath(os.path.expanduser(text))


_OPTION_TYPES = {
    "str": str,
    "int": int,
    "positive_int": _positive_int,
    "float": float,
    "path": _absolute_path,
    "optional_str": _none_or(str),
    "optional_int": _none_or(int),
    "optional_float": _none_or(float),
    "optional_path": _none_or(_absolute_path),
}


def _format_default(value) -> str:
    if isinstance(value, tuple):
        return " ".join(str(item) for item in value) if value else "()"
    if isinstance(value, str) and value.startswith(str(REPOSITORY_ROOT)):
        return os.path.relpath(value, REPOSITORY_ROOT)
    return str(value)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description=(
            "Run one experiment step. Every option overrides a DEFAULT SETTING at the top "
            "of run.py; without options the defaults are used unchanged."
        ),
        epilog=(
            "AEGIS workflow: --method filtering_anchor -> prepare_anchor_geometry_reference -> "
            "target_plus_anchor_ce_geometry, then --action native-evaluate --checkpoint "
            ".../final. See README.md."
        ),
        argument_default=argparse.SUPPRESS,
        allow_abbrev=False,
    )
    for title, options in _OPTION_GROUPS:
        group = parser.add_argument_group(title)
        for option in options:
            default = globals()[option.setting]
            help_text = f"{option.help} [default: {_format_default(default)}]".replace("%", "%%")
            keywords = {"dest": option.setting.lower(), "help": help_text}
            if option.choices is not None and option.kind != "list":
                keywords["choices"] = option.choices
            if option.kind == "bool":
                group.add_argument(*option.flags, action=argparse.BooleanOptionalAction, **keywords)
            elif option.kind in {"gpus", "list"}:
                if option.choices is not None:
                    keywords["choices"] = option.choices
                group.add_argument(
                    *option.flags,
                    nargs="+",
                    metavar=option.flags[0][2:].upper().replace("-", "_"),
                    **keywords,
                )
            else:
                if option.choices is None:
                    keywords["metavar"] = option.flags[0][2:].upper().replace("-", "_")
                group.add_argument(*option.flags, type=_OPTION_TYPES[option.kind], **keywords)
    other = parser.add_argument_group("advanced")
    other.add_argument(
        "--set",
        action="append",
        metavar="NAME=VALUE",
        dest="set_values",
        help="override any other DEFAULT SETTING by name; VALUE is a Python literal or a "
        "plain string, e.g. --set LVD_DIRECTION=256",
    )
    other.add_argument(
        "--dry-run",
        action="store_true",
        help="print the resolved step and command, then exit without running",
    )
    return parser


def _parse_set_value(parser: argparse.ArgumentParser, item: str) -> tuple[str, object]:
    name, separator, text = item.partition("=")
    name = name.strip()
    if not separator or not name:
        parser.error(f"--set expects NAME=VALUE, got {item!r}")
    if name not in _SETTING_NAMES:
        parser.error(f"--set: unknown setting {name!r}")
    try:
        value = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        value = text
    return name, value


def _apply_cli_arguments(parser: argparse.ArgumentParser, namespace: argparse.Namespace) -> None:
    global PINNED_ENVIRONMENT, PINNED_PYTHON
    previous_root = ARTIFACT_ROOT
    for option in _OPTIONS:
        destination = option.setting.lower()
        if not hasattr(namespace, destination):
            continue
        value = getattr(namespace, destination)
        if option.kind == "gpus":
            try:
                value = tuple(
                    int(part) for item in value for part in str(item).split(",") if part.strip()
                )
            except ValueError:
                parser.error(f"{option.flags[0]} expects integer GPU ids, got {value!r}")
        elif option.kind == "list":
            value = tuple(value)
        globals()[option.setting] = value
    for item in getattr(namespace, "set_values", None) or ():
        name, value = _parse_set_value(parser, item)
        globals()[name] = value
    if previous_root != ARTIFACT_ROOT and not os.environ.get(TRAINING_ENVIRONMENT_ENV):
        PINNED_ENVIRONMENT = _default_training_environment(ARTIFACT_ROOT)
        PINNED_PYTHON = PINNED_ENVIRONMENT / "bin" / "python"


def _launch_plan() -> str:
    action = _effective_action()
    gpu_ids = tuple(GPU_IDS)
    if _label_coverage_filtering():
        return "one CPU process (label-coverage filtering needs no GPU)"
    if action in _DISTRIBUTED_TRAINING_ACTIONS:
        if DISTRIBUTED_STRATEGY == "single":
            return f"one process on GPU {gpu_ids[0]}"
        return (
            f"Accelerate {DISTRIBUTED_STRATEGY.upper()} with {len(gpu_ids)} processes on "
            f"GPUs {gpu_ids}"
        )
    if action == "filtering-anchor":
        return f"multi-GPU correctness scoring on GPUs {gpu_ids} (legacy policy)"
    if action in {"record-anchor-d0", _GEOMETRY_REFERENCE_ACTION, "create-anchor-dataset"}:
        return f"one process on GPU {gpu_ids[0]}"
    return f"one process; first visible GPU of {gpu_ids}"


def _print_dry_run() -> None:
    print("Dry run: nothing is launched.")
    print(
        "  step: "
        f"action={_effective_action()} method={_effective_method()} model={MODEL} "
        f"target={TARGET_DATASET} anchor={_effective_anchor_dataset()}"
    )
    print(f"  launch: {_launch_plan()}")
    print(f"  artifact root: {ARTIFACT_ROOT}")
    print("  equivalent package command:")
    print("    sft-conflict " + shlex.join(_run_arguments()))


def main(argv: Sequence[str] = ()) -> int:
    global _FORWARDED_ARGUMENTS
    arguments = list(argv)
    parser = _build_parser()
    namespace = parser.parse_args(arguments)
    _apply_cli_arguments(parser, namespace)
    _FORWARDED_ARGUMENTS = tuple(argument for argument in arguments if argument != "--dry-run")
    is_child = os.environ.get(DISTRIBUTED_CHILD_ENV) == "1"
    try:
        _run_arguments()
    except ValueError as exc:
        parser.exit(2, f"run.py: error: {exc}\n")
    if getattr(namespace, "dry_run", False):
        _print_dry_run()
        return 0
    if not arguments and not is_child and os.environ.get("SFT_CONFLICT_CLICK_RUN") != "1":
        print(
            "run.py: no command-line options; using the DEFAULT SETTINGS at the top of run.py "
            "(python run.py --help lists every option).",
            flush=True,
        )
    _configure_cuda_allocator()
    _configure_hugging_face_environment()
    _ensure_training_interpreter()
    if not is_child:
        _print_resolved_runtime_settings(_resolved_runtime_settings())
    if not _label_coverage_filtering():
        gpu_ids = _validated_gpu_ids()
        _launch_distributed_if_needed(gpu_ids)
    if str(PROJECT_SOURCE) not in sys.path:
        sys.path.insert(0, str(PROJECT_SOURCE))
    from sft_conflict.cli.main import main as cli_main

    return cli_main(_run_arguments())


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
