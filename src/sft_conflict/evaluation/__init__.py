"""Evaluation bridges kept separate from the training implementation."""

from .code_native import CodeXGLUETokenCompletionEvaluator, HumanEvalEvaluator
from .lm_eval_bridge import LmEvalRequest, ensure_lm_eval_task_cache, run_lm_eval
from .logic_native import AlpacaEvalGenerationEvaluator, LogiQA2Evaluator
from .native import (
    ClassificationEvaluator,
    EvaluationProtocol,
    EvaluationResult,
    Evaluator,
    GenerationEvaluator,
    GLUEClassificationEvaluator,
    GSM8KEvaluator,
    SST2Evaluator,
    STSBEvaluator,
    build_native_evaluator,
    classification_metrics,
    extract_gsm8k_answer,
    extract_stsb_score,
    regression_metrics,
    write_evaluation_result,
)
from .statistics import paired_correctness_comparison

__all__ = [
    "AlpacaEvalGenerationEvaluator",
    "ClassificationEvaluator",
    "CodeXGLUETokenCompletionEvaluator",
    "EvaluationProtocol",
    "EvaluationResult",
    "Evaluator",
    "GSM8KEvaluator",
    "GenerationEvaluator",
    "GLUEClassificationEvaluator",
    "LmEvalRequest",
    "LogiQA2Evaluator",
    "HumanEvalEvaluator",
    "SST2Evaluator",
    "STSBEvaluator",
    "build_native_evaluator",
    "classification_metrics",
    "extract_gsm8k_answer",
    "extract_stsb_score",
    "ensure_lm_eval_task_cache",
    "paired_correctness_comparison",
    "run_lm_eval",
    "regression_metrics",
    "write_evaluation_result",
]
