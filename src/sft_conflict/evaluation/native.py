"""Task-native evaluation with versioned protocols and per-sample outputs."""

from __future__ import annotations

import json
import math
import os
import re
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

from sft_conflict.data import LogicalDataset
from sft_conflict.data.glue import GLUE_TASK_SPECS


@dataclass(frozen=True)
class EvaluationProtocol:
    task_name: str
    evaluator_type: str
    metric_name: str
    prompt_template: str
    prompt_template_version: str
    num_fewshot: int = 0
    model_variant: str = "base"
    max_input_length: int = 512
    max_new_tokens: int = 128
    stop_tokens: tuple[str, ...] = ()
    do_sample: bool = False
    temperature: float = 0.0
    top_p: float = 1.0
    samples_per_prompt: int = 1

    def validate(self) -> None:
        if self.evaluator_type not in {"classification", "generation"}:
            raise ValueError("evaluator_type must be classification or generation")
        if self.model_variant != "base":
            raise ValueError("M2 native evaluators currently require model_variant=base")
        if self.num_fewshot < 0:
            raise ValueError("num_fewshot must be non-negative")
        if self.max_input_length <= 1 or self.max_new_tokens <= 0:
            raise ValueError("evaluation token limits must be positive")
        if self.samples_per_prompt <= 0:
            raise ValueError("samples_per_prompt must be positive")
        if self.do_sample and self.temperature <= 0:
            raise ValueError("sampled generation requires a positive temperature")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")


@dataclass(frozen=True)
class EvaluationResult:
    protocol: EvaluationProtocol
    dataset_manifest: dict[str, Any]
    aggregate: dict[str, int | float | str]
    samples: tuple[dict[str, Any], ...]

    def to_dict(self, *, include_samples: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": 1,
            "protocol": asdict(self.protocol),
            "dataset_manifest": self.dataset_manifest,
            "aggregate": self.aggregate,
        }
        if include_samples:
            result["samples"] = list(self.samples)
        return result


class Evaluator(ABC):
    def __init__(self, protocol: EvaluationProtocol) -> None:
        protocol.validate()
        self.protocol = protocol

    @abstractmethod
    def evaluate(
        self,
        model,
        tokenizer,
        dataset: LogicalDataset,
        *,
        batch_size: int,
    ) -> EvaluationResult:
        """Evaluate a complete logical dataset."""


class ClassificationEvaluator(Evaluator, ABC):
    """Marker base class for finite-label conditional-likelihood tasks."""


class GenerationEvaluator(Evaluator, ABC):
    """Marker base class for deterministic free-generation tasks."""


def classification_metrics(
    task_name: str,
    predictions: list[int],
    labels: list[int],
) -> dict[str, float]:
    if not predictions or len(predictions) != len(labels):
        raise ValueError("classification metrics require aligned non-empty values")
    accuracy = sum(
        prediction == label for prediction, label in zip(predictions, labels, strict=True)
    ) / len(labels)
    if task_name == "cola":
        pairs = tuple(zip(predictions, labels, strict=True))
        true_positive = sum(p == 1 and y == 1 for p, y in pairs)
        true_negative = sum(p == 0 and y == 0 for p, y in pairs)
        false_positive = sum(p == 1 and y == 0 for p, y in pairs)
        false_negative = sum(p == 0 and y == 1 for p, y in pairs)
        denominator = math.sqrt(
            (true_positive + false_positive)
            * (true_positive + false_negative)
            * (true_negative + false_positive)
            * (true_negative + false_negative)
        )
        matthews = (
            (true_positive * true_negative - false_positive * false_negative) / denominator
            if denominator
            else 0.0
        )
        return {"matthews_correlation": matthews}
    if task_name in {"mrpc", "qqp"}:
        pairs = tuple(zip(predictions, labels, strict=True))
        true_positive = sum(p == 1 and y == 1 for p, y in pairs)
        false_positive = sum(p == 1 and y == 0 for p, y in pairs)
        false_negative = sum(p == 0 and y == 1 for p, y in pairs)
        denominator = 2 * true_positive + false_positive + false_negative
        f1 = 2 * true_positive / denominator if denominator else 0.0
        return {
            "accuracy": accuracy,
            "f1": f1,
            "averaged_scores": (accuracy + f1) / 2,
        }
    return {"accuracy": accuracy}


def _average_ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(indexed):
        end = start + 1
        while end < len(indexed) and indexed[end][1] == indexed[start][1]:
            end += 1
        average_rank = (start + 1 + end) / 2
        for position in range(start, end):
            ranks[indexed[position][0]] = average_rank
        start = end
    return ranks


def _pearson(left: list[float], right: list[float]) -> float:
    if not left or len(left) != len(right):
        raise ValueError("correlation requires aligned non-empty values")
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right, strict=True)
    )
    denominator = math.sqrt(
        sum((value - left_mean) ** 2 for value in left)
        * sum((value - right_mean) ** 2 for value in right)
    )
    return numerator / denominator if denominator else 0.0


def regression_metrics(predictions: list[float], labels: list[float]) -> dict[str, float]:
    pearson = _pearson(predictions, labels)
    spearman = _pearson(_average_ranks(predictions), _average_ranks(labels))
    return {
        "pearson": pearson,
        "spearmanr": spearman,
        "averaged_scores": (pearson + spearman) / 2,
    }


def _model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration as exc:
        raise ValueError("evaluation model has no parameters") from exc


def _batched(values: list[Any], batch_size: int):
    if batch_size <= 0:
        raise ValueError("evaluation batch_size must be positive")
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def _score_completion_candidates(
    model,
    tokenizer,
    prompts: list[str],
    responses: tuple[str, ...],
    *,
    max_input_length: int,
) -> list[list[float]]:
    texts: list[str] = []
    boundaries: list[int] = []
    for prompt in prompts:
        for response in responses:
            texts.append(prompt + response)
            boundaries.append(len(prompt))
    encoded = tokenizer(
        texts,
        add_special_tokens=True,
        padding=True,
        truncation=True,
        max_length=max_input_length,
        return_attention_mask=True,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    offsets = encoded.pop("offset_mapping")
    input_ids = encoded["input_ids"]
    completion_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for row_index, (row_offsets, boundary) in enumerate(zip(offsets, boundaries, strict=True)):
        for token_index, pair in enumerate(row_offsets.tolist()):
            start, end = pair
            if start < boundary < end and not texts[row_index][start:boundary].isspace():
                raise ValueError("classification response boundary falls inside one token")
            if (start >= boundary and end > start) or (
                start < boundary < end and texts[row_index][start:boundary].isspace()
            ):
                completion_mask[row_index, token_index] = True
        if not bool(completion_mask[row_index, 1:].any()):
            raise ValueError("classification candidate was fully truncated")
    device = _model_device(model)
    model_inputs = {
        key: encoded[key].to(device) for key in ("input_ids", "attention_mask") if key in encoded
    }
    targets = input_ids[:, 1:].to(device)
    selected = completion_mask[:, 1:].to(device)
    with torch.inference_mode():
        logits = model(**model_inputs, use_cache=False).logits[:, :-1, :]
        selected_logits = logits[selected].float()
    del logits
    selected_targets = targets.masked_select(selected)
    selected_token_scores = (
        torch.log_softmax(selected_logits, dim=-1)
        .gather(dim=-1, index=selected_targets.unsqueeze(-1))
        .squeeze(-1)
    )
    selected_rows = (
        torch.arange(len(texts), device=device)
        .unsqueeze(1)
        .expand_as(selected)
        .masked_select(selected)
    )
    scores_tensor = torch.zeros(len(texts), dtype=torch.float32, device=device)
    scores_tensor.scatter_add_(0, selected_rows, selected_token_scores)
    scores = scores_tensor.cpu().tolist()
    width = len(responses)
    return [scores[index : index + width] for index in range(0, len(scores), width)]


def _validated_result(
    protocol: EvaluationProtocol,
    dataset: LogicalDataset,
    metric: float,
    samples: list[dict[str, Any]],
) -> EvaluationResult:
    if not samples or len(samples) != len(dataset):
        raise RuntimeError(f"evaluation produced {len(samples)} results for {len(dataset)} samples")
    if not math.isfinite(metric):
        raise RuntimeError(f"evaluation produced non-finite {protocol.metric_name}: {metric}")
    correct = sum(bool(sample["correct"]) for sample in samples)
    return EvaluationResult(
        protocol=protocol,
        dataset_manifest=dataset.manifest,
        aggregate={
            "task_name": protocol.task_name,
            "metric_name": protocol.metric_name,
            protocol.metric_name: metric,
            "correct": correct,
            "sample_count": len(samples),
        },
        samples=tuple(samples),
    )


class SST2Evaluator(ClassificationEvaluator):
    RESPONSES = (" negative", " positive")

    def __init__(self, *, max_input_length: int = 512) -> None:
        super().__init__(
            EvaluationProtocol(
                task_name="sst2",
                evaluator_type="classification",
                metric_name="accuracy",
                prompt_template="Sentence: {sentence}\\nSentiment:",
                prompt_template_version="sst2-sentiment-v1",
                max_input_length=max_input_length,
                max_new_tokens=1,
            )
        )

    def _score_candidates(self, model, tokenizer, prompts: list[str]) -> list[list[float]]:
        return _score_completion_candidates(
            model,
            tokenizer,
            prompts,
            self.RESPONSES,
            max_input_length=self.protocol.max_input_length,
        )

    def evaluate(
        self,
        model,
        tokenizer,
        dataset: LogicalDataset,
        *,
        batch_size: int,
    ) -> EvaluationResult:
        samples: list[dict[str, Any]] = []
        was_training = model.training
        model.eval()
        try:
            indexed_examples = list(enumerate(dataset))
            chunks = _batched(indexed_examples, batch_size)
            for chunk in tqdm(
                chunks,
                total=math.ceil(len(indexed_examples) / batch_size),
                desc="SST-2 native evaluation",
                unit="batch",
                dynamic_ncols=True,
            ):
                prompts = [example.prompt for _, example in chunk]
                score_rows = self._score_candidates(model, tokenizer, prompts)
                for (_, example), scores in zip(chunk, score_rows, strict=True):
                    gold_label = int(example.source_metadata["gold_label"])
                    prediction = max(range(len(scores)), key=scores.__getitem__)
                    samples.append(
                        {
                            "sample_id": example.sample_id,
                            "prediction": prediction,
                            "gold": gold_label,
                            "correct": prediction == gold_label,
                            "candidate_loglikelihoods": {
                                response.strip(): score
                                for response, score in zip(self.RESPONSES, scores, strict=True)
                            },
                        }
                    )
        finally:
            model.train(was_training)
        accuracy = sum(sample["correct"] for sample in samples) / len(samples)
        return _validated_result(self.protocol, dataset, accuracy, samples)


class GLUEClassificationEvaluator(ClassificationEvaluator):
    def __init__(self, task_name: str, *, max_input_length: int = 512) -> None:
        spec = GLUE_TASK_SPECS[task_name]
        if spec.regression:
            raise ValueError(f"{task_name} is a regression task")
        self.task_spec = spec
        super().__init__(
            EvaluationProtocol(
                task_name=task_name,
                evaluator_type="classification",
                metric_name=spec.metric_names[-1],
                prompt_template="dataset adapter prompt",
                prompt_template_version=f"glue-{task_name}-decoder-v1",
                max_input_length=max_input_length,
                max_new_tokens=1,
            )
        )

    def evaluate(
        self,
        model,
        tokenizer,
        dataset: LogicalDataset,
        *,
        batch_size: int,
    ) -> EvaluationResult:
        samples: list[dict[str, Any]] = []
        responses = tuple(f" {label}" for label in self.task_spec.label_texts)
        was_training = model.training
        model.eval()
        try:
            indexed_examples = list(enumerate(dataset))
            for chunk in tqdm(
                _batched(indexed_examples, batch_size),
                total=math.ceil(len(indexed_examples) / batch_size),
                desc=f"{self.task_spec.key.upper()} native evaluation",
                unit="batch",
                dynamic_ncols=True,
            ):
                score_rows = _score_completion_candidates(
                    model,
                    tokenizer,
                    [example.prompt for _, example in chunk],
                    responses,
                    max_input_length=self.protocol.max_input_length,
                )
                for (_, example), scores in zip(chunk, score_rows, strict=True):
                    gold = int(example.source_metadata["gold_label"])
                    prediction = max(range(len(scores)), key=scores.__getitem__)
                    samples.append(
                        {
                            "sample_id": example.sample_id,
                            "prediction": prediction,
                            "prediction_text": self.task_spec.label_texts[prediction],
                            "gold": gold,
                            "gold_text": self.task_spec.label_texts[gold],
                            "correct": prediction == gold,
                            "candidate_loglikelihoods": {
                                label: score
                                for label, score in zip(
                                    self.task_spec.label_texts, scores, strict=True
                                )
                            },
                            "template_id": example.source_metadata.get("template_id"),
                        }
                    )
        finally:
            model.train(was_training)
        metrics = classification_metrics(
            self.task_spec.key,
            [int(sample["prediction"]) for sample in samples],
            [int(sample["gold"]) for sample in samples],
        )
        return EvaluationResult(
            protocol=self.protocol,
            dataset_manifest=dataset.manifest,
            aggregate={
                "task_name": self.task_spec.key,
                "metric_name": self.protocol.metric_name,
                "correct": sum(bool(sample["correct"]) for sample in samples),
                "sample_count": len(samples),
                **metrics,
            },
            samples=tuple(samples),
        )


_NUMBER_PATTERN = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def extract_gsm8k_answer(text: str) -> str | None:
    final_section = text.rsplit("####", maxsplit=1)[-1]
    matches = _NUMBER_PATTERN.findall(final_section)
    if not matches and "####" in text:
        matches = _NUMBER_PATTERN.findall(text)
    if not matches:
        return None
    raw = matches[-1].replace(",", "")
    try:
        value = Decimal(raw)
    except InvalidOperation:
        return raw
    normalized = format(value.normalize(), "f")
    return "0" if normalized in {"-0", "+0"} else normalized


def extract_stsb_score(text: str) -> float | None:
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if match is None:
        return None
    return min(5.0, max(0.0, float(match.group(0))))


class STSBEvaluator(GenerationEvaluator):
    def __init__(self, *, max_input_length: int = 512, max_new_tokens: int = 8) -> None:
        super().__init__(
            EvaluationProtocol(
                task_name="stsb",
                evaluator_type="generation",
                metric_name="averaged_scores",
                prompt_template="dataset adapter prompt",
                prompt_template_version="glue-stsb-decoder-v1",
                max_input_length=max_input_length,
                max_new_tokens=min(max_new_tokens, 8),
            )
        )

    def evaluate(
        self,
        model,
        tokenizer,
        dataset: LogicalDataset,
        *,
        batch_size: int,
    ) -> EvaluationResult:
        samples: list[dict[str, Any]] = []
        device = _model_device(model)
        was_training = model.training
        original_padding_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        model.eval()
        try:
            indexed_examples = list(enumerate(dataset))
            for chunk in tqdm(
                _batched(indexed_examples, batch_size),
                total=math.ceil(len(indexed_examples) / batch_size),
                desc="STS-B native evaluation",
                unit="batch",
                dynamic_ncols=True,
            ):
                encoded = tokenizer(
                    [example.prompt for _, example in chunk],
                    add_special_tokens=True,
                    padding=True,
                    truncation=True,
                    max_length=self.protocol.max_input_length,
                    return_attention_mask=True,
                    return_tensors="pt",
                )
                prompt_width = encoded["input_ids"].shape[1]
                model_inputs = {
                    key: encoded[key].to(device)
                    for key in ("input_ids", "attention_mask")
                    if key in encoded
                }
                with torch.inference_mode():
                    generated = model.generate(
                        **model_inputs,
                        do_sample=False,
                        max_new_tokens=self.protocol.max_new_tokens,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                texts = tokenizer.batch_decode(
                    generated[:, prompt_width:].cpu(), skip_special_tokens=True
                )
                for (_, example), text in zip(chunk, texts, strict=True):
                    parsed = extract_stsb_score(text)
                    prediction = parsed if parsed is not None else 0.0
                    samples.append(
                        {
                            "sample_id": example.sample_id,
                            "prediction": prediction,
                            "gold": float(example.source_metadata["gold_label"]),
                            "generation": text,
                            "parsed": parsed is not None,
                            "template_id": example.source_metadata.get("template_id"),
                        }
                    )
        finally:
            tokenizer.padding_side = original_padding_side
            model.train(was_training)
        metrics = regression_metrics(
            [float(sample["prediction"]) for sample in samples],
            [float(sample["gold"]) for sample in samples],
        )
        return EvaluationResult(
            protocol=self.protocol,
            dataset_manifest=dataset.manifest,
            aggregate={
                "task_name": "stsb",
                "metric_name": "averaged_scores",
                "sample_count": len(samples),
                "parse_rate": sum(bool(sample["parsed"]) for sample in samples) / len(samples),
                **metrics,
            },
            samples=tuple(samples),
        )


class GSM8KEvaluator(GenerationEvaluator):
    def __init__(
        self,
        *,
        max_input_length: int = 512,
        max_new_tokens: int = 256,
        stop_tokens: tuple[str, ...] = (),
    ) -> None:
        super().__init__(
            EvaluationProtocol(
                task_name="gsm8k",
                evaluator_type="generation",
                metric_name="exact_match",
                prompt_template="Question: {question}\\nAnswer:",
                prompt_template_version="gsm8k-cot-v1",
                max_input_length=max_input_length,
                max_new_tokens=max_new_tokens,
                stop_tokens=stop_tokens,
            )
        )

    def _truncate_at_stop(self, text: str) -> str:
        offsets = [text.find(token) for token in self.protocol.stop_tokens if token in text]
        return text[: min(offsets)] if offsets else text

    def evaluate(
        self,
        model,
        tokenizer,
        dataset: LogicalDataset,
        *,
        batch_size: int,
    ) -> EvaluationResult:
        samples: list[dict[str, Any]] = []
        device = _model_device(model)
        was_training = model.training
        original_padding_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        model.eval()
        try:
            indexed_examples = list(enumerate(dataset))
            chunks = _batched(indexed_examples, batch_size)
            for chunk in tqdm(
                chunks,
                total=math.ceil(len(indexed_examples) / batch_size),
                desc="GSM8K native evaluation",
                unit="batch",
                dynamic_ncols=True,
            ):
                encoded = tokenizer(
                    [example.prompt for _, example in chunk],
                    add_special_tokens=True,
                    padding=True,
                    truncation=True,
                    max_length=self.protocol.max_input_length,
                    return_attention_mask=True,
                    return_tensors="pt",
                )
                prompt_width = encoded["input_ids"].shape[1]
                model_inputs = {
                    key: encoded[key].to(device)
                    for key in ("input_ids", "attention_mask")
                    if key in encoded
                }
                with torch.inference_mode():
                    generated = model.generate(
                        **model_inputs,
                        do_sample=False,
                        max_new_tokens=self.protocol.max_new_tokens,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                generated_ids = generated[:, prompt_width:].cpu()
                texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                for (_, example), generated_text in zip(chunk, texts, strict=True):
                    generated_text = self._truncate_at_stop(generated_text)
                    prediction = extract_gsm8k_answer(generated_text)
                    gold = extract_gsm8k_answer(example.response)
                    if gold is None:
                        raise ValueError(f"GSM8K gold answer is unparsable: {example.sample_id}")
                    samples.append(
                        {
                            "sample_id": example.sample_id,
                            "prediction": prediction,
                            "gold": gold,
                            "correct": prediction == gold,
                            "generation": generated_text,
                        }
                    )
        finally:
            tokenizer.padding_side = original_padding_side
            model.train(was_training)
        exact_match = sum(sample["correct"] for sample in samples) / len(samples)
        return _validated_result(self.protocol, dataset, exact_match, samples)


def build_native_evaluator(
    task_name: str,
    *,
    max_input_length: int = 512,
    max_new_tokens: int = 256,
    stop_tokens: tuple[str, ...] = (),
    allow_code_execution: bool = False,
    humaneval_samples_per_problem: int = 20,
    humaneval_temperature: float = 0.2,
    humaneval_top_p: float = 0.95,
    humaneval_timeout_seconds: float = 3.0,
    humaneval_execution_workers: int = 16,
    generator_name: str = "sft-conflict-model",
) -> Evaluator:
    if task_name == "humaneval":
        from .code_native import HumanEvalEvaluator

        return HumanEvalEvaluator(
            max_input_length=max_input_length,
            max_new_tokens=max_new_tokens,
            samples_per_problem=humaneval_samples_per_problem,
            temperature=humaneval_temperature,
            top_p=humaneval_top_p,
            timeout_seconds=humaneval_timeout_seconds,
            execution_workers=humaneval_execution_workers,
            allow_code_execution=allow_code_execution,
        )
    if task_name in {"codexglue_python", "codexglue_java"}:
        from .code_native import CodeXGLUETokenCompletionEvaluator

        return CodeXGLUETokenCompletionEvaluator(
            task_name,
            max_input_length=max_input_length,
        )
    if task_name == "logiqa2":
        from .logic_native import LogiQA2Evaluator

        return LogiQA2Evaluator(max_input_length=max_input_length)
    if task_name == "alpaca_eval":
        from .logic_native import AlpacaEvalGenerationEvaluator

        return AlpacaEvalGenerationEvaluator(
            max_input_length=max_input_length,
            max_new_tokens=max_new_tokens,
            generator_name=generator_name,
        )
    if task_name == "stsb":
        return STSBEvaluator(
            max_input_length=max_input_length,
            max_new_tokens=max_new_tokens,
        )
    if task_name in GLUE_TASK_SPECS:
        return GLUEClassificationEvaluator(
            task_name,
            max_input_length=max_input_length,
        )
    if task_name == "gsm8k":
        return GSM8KEvaluator(
            max_input_length=max_input_length,
            max_new_tokens=max_new_tokens,
            stop_tokens=stop_tokens,
        )
    choices = ", ".join(
        (
            "gsm8k",
            "logiqa2",
            "alpaca_eval",
            "humaneval",
            "codexglue_python",
            "codexglue_java",
            *GLUE_TASK_SPECS,
        )
    )
    raise ValueError(f"native evaluator must be one of: {choices}")


def write_evaluation_result(result: EvaluationResult, output_dir: str | Path) -> dict[str, str]:
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    aggregate_path = destination / "evaluation.json"
    samples_path = destination / "samples.jsonl"
    aggregate_temporary = aggregate_path.with_suffix(".json.tmp")
    samples_temporary = samples_path.with_suffix(".jsonl.tmp")
    aggregate_temporary.write_text(
        json.dumps(result.to_dict(include_samples=False), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    samples_temporary.write_text(
        "".join(
            json.dumps(sample, ensure_ascii=True, sort_keys=True) + "\n"
            for sample in result.samples
        ),
        encoding="utf-8",
    )
    os.replace(aggregate_temporary, aggregate_path)
    os.replace(samples_temporary, samples_path)
    paths = {"aggregate": str(aggregate_path), "samples": str(samples_path)}
    if result.protocol.task_name == "alpaca_eval":
        model_outputs_path = destination / "model_outputs.json"
        model_outputs_temporary = model_outputs_path.with_suffix(".json.tmp")
        required_fields = ("instruction", "output", "generator", "dataset")
        model_outputs = [
            {field: sample[field] for field in required_fields} for sample in result.samples
        ]
        model_outputs_temporary.write_text(
            json.dumps(model_outputs, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(model_outputs_temporary, model_outputs_path)
        paths["model_outputs"] = str(model_outputs_path)
    return paths
