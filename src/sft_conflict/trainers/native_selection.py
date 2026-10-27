"""Best-checkpoint selection by the task-native classification metric."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sft_conflict.data import LogicalDataset
from sft_conflict.evaluation.native import ClassificationEvaluator

BEST_MODEL_METRICS = ("eval_loss", "native_accuracy")


@dataclass(frozen=True)
class NativeSelection:
    """Validation-split native classification accuracy scored at every evaluation.

    The logical dataset is the same training-derived validation split that provides
    the evaluation loss, scored with the task's native candidate-likelihood protocol
    (for example " yes"/" no" for RTE); the held-out evaluation split is never used.
    """

    evaluator: ClassificationEvaluator
    dataset: LogicalDataset
    tokenizer: Any
    batch_size: int
    task_name: str

    def __post_init__(self) -> None:
        if not isinstance(self.evaluator, ClassificationEvaluator):
            raise ValueError("native selection requires a classification evaluator")
        if not self.evaluator.supports_selection():
            raise ValueError(
                f"{type(self.evaluator).__name__} does not expose fixed candidates for selection"
            )
        if self.batch_size <= 0:
            raise ValueError("native selection batch_size must be positive")
        if len(self.dataset) == 0:
            raise ValueError("native selection dataset is empty")
