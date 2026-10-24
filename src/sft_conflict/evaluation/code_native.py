"""Native HumanEval and CodeXGLUE evaluators for decoder-only models."""

from __future__ import annotations

import math
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import torch
from tqdm.auto import tqdm

from sft_conflict.data import LogicalDataset

from .code_metrics import aggregate_pass_at_k, execute_python_program
from .native import (
    ClassificationEvaluator,
    EvaluationProtocol,
    EvaluationResult,
    GenerationEvaluator,
    _batched,
    _model_device,
)

HUMANEVAL_STOP_WORDS = (
    "\nclass",
    "\ndef",
    "\n#",
    "\n@",
    "\nprint",
    "\nif",
    "\n```",
    "<file_sep>",
)


def _truncate_at_stop(text: str, stop_words: tuple[str, ...]) -> str:
    offsets = [text.find(word) for word in stop_words if word in text]
    return text[: min(offsets)] if offsets else text


def _execute_candidate(arguments: tuple[str, bool, float]) -> dict[str, Any]:
    program, allow_code_execution, timeout_seconds = arguments
    return execute_python_program(
        program,
        allow_code_execution=allow_code_execution,
        timeout_seconds=timeout_seconds,
    )


class HumanEvalEvaluator(GenerationEvaluator):
    def __init__(
        self,
        *,
        max_input_length: int = 512,
        max_new_tokens: int = 512,
        samples_per_problem: int = 20,
        temperature: float = 0.2,
        top_p: float = 0.95,
        timeout_seconds: float = 3.0,
        execution_workers: int = 16,
        allow_code_execution: bool = False,
        execution_function: Callable[[str], dict[str, Any]] | None = None,
        seed: int = 0,
    ) -> None:
        if samples_per_problem < 10:
            raise ValueError("HumanEval pass@10 requires at least 10 samples per problem")
        if timeout_seconds <= 0 or execution_workers <= 0:
            raise ValueError("HumanEval execution settings must be positive")
        self.samples_per_problem = samples_per_problem
        self.timeout_seconds = timeout_seconds
        self.execution_workers = execution_workers
        self.allow_code_execution = allow_code_execution
        self.execution_function = execution_function
        self.seed = seed
        super().__init__(
            EvaluationProtocol(
                task_name="humaneval",
                evaluator_type="generation",
                metric_name="pass@1",
                prompt_template="OpenAI HumanEval function signature and docstring",
                prompt_template_version="lori-humaneval-completion-v1",
                max_input_length=max_input_length,
                max_new_tokens=max_new_tokens,
                stop_tokens=HUMANEVAL_STOP_WORDS,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                samples_per_prompt=samples_per_problem,
            )
        )

    def _run_candidates(self, programs: list[str]) -> list[dict[str, Any]]:
        if self.execution_function is not None:
            return [self.execution_function(program) for program in programs]
        arguments = [
            (program, self.allow_code_execution, self.timeout_seconds) for program in programs
        ]
        with ProcessPoolExecutor(max_workers=self.execution_workers) as executor:
            return list(executor.map(_execute_candidate, arguments))

    def evaluate(
        self,
        model,
        tokenizer,
        dataset: LogicalDataset,
        *,
        batch_size: int,
    ) -> EvaluationResult:
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("HumanEval batch_size must be a positive integer")
        if not self.allow_code_execution and self.execution_function is None:
            raise PermissionError(
                "HumanEval requires ALLOW_CODE_EXECUTION=True because generated Python "
                "must be executed against hidden tests."
            )
        device = _model_device(model)
        was_training = model.training
        original_padding_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        all_completions: list[list[str]] = []
        generation_calls = 0
        print(
            "HumanEval sampling: "
            f"problems={len(dataset)}, samples_per_problem={self.samples_per_problem}, "
            f"total_completions={len(dataset) * self.samples_per_problem}, "
            f"max_concurrent_sequences={batch_size}",
            flush=True,
        )
        model.eval()
        try:
            indexed_examples = list(enumerate(dataset))
            chunks = _batched(indexed_examples, batch_size)
            with tqdm(
                chunks,
                total=math.ceil(len(indexed_examples) / batch_size),
                desc="HumanEval generation",
                unit="batch",
                dynamic_ncols=True,
            ) as progress:
                for chunk_index, chunk in enumerate(progress):
                    prompts = [example.prompt for _, example in chunk]
                    encoded = tokenizer(
                        prompts,
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
                    chunk_completions: list[list[str]] = [[] for _ in chunk]
                    returns_per_call = max(1, batch_size // len(chunk))
                    devices = [device.index or 0] if device.type == "cuda" else []
                    with torch.random.fork_rng(devices=devices), torch.inference_mode():
                        # Seed once: successive waves must advance the same random stream.
                        torch.manual_seed(self.seed + chunk_index)
                        for start in range(0, self.samples_per_problem, returns_per_call):
                            returns = min(returns_per_call, self.samples_per_problem - start)
                            expected_sequences = len(chunk) * returns
                            generated = model.generate(
                                **model_inputs,
                                do_sample=True,
                                temperature=self.protocol.temperature,
                                top_p=self.protocol.top_p,
                                top_k=0,
                                num_return_sequences=returns,
                                max_new_tokens=self.protocol.max_new_tokens,
                                pad_token_id=tokenizer.pad_token_id,
                                eos_token_id=tokenizer.eos_token_id,
                            )
                            if (
                                not isinstance(generated, torch.Tensor)
                                or generated.ndim != 2
                                or generated.shape[0] != expected_sequences
                            ):
                                raise RuntimeError(
                                    "HumanEval generation returned an invalid sequence count; "
                                    f"expected {expected_sequences}"
                                )
                            decoded = tokenizer.batch_decode(
                                generated[:, prompt_width:].cpu(),
                                skip_special_tokens=True,
                            )
                            del generated
                            if len(decoded) != expected_sequences:
                                raise RuntimeError(
                                    "HumanEval decoding returned an invalid sequence count; "
                                    f"expected {expected_sequences}"
                                )
                            for offset, completions in enumerate(chunk_completions):
                                completions.extend(
                                    _truncate_at_stop(completion, self.protocol.stop_tokens)
                                    for completion in decoded[
                                        offset * returns : (offset + 1) * returns
                                    ]
                                )
                            generation_calls += 1
                            progress.set_postfix(
                                samples=f"{start + returns}/{self.samples_per_problem}",
                                sequences=expected_sequences,
                            )
                    if any(
                        len(completions) != self.samples_per_problem
                        for completions in chunk_completions
                    ):
                        raise RuntimeError("HumanEval generated an incomplete set of completions")
                    all_completions.extend(chunk_completions)
        finally:
            tokenizer.padding_side = original_padding_side
            model.train(was_training)

        programs: list[str] = []
        owners: list[tuple[int, int]] = []
        for problem_index, (example, completions) in enumerate(
            zip(dataset, all_completions, strict=True)
        ):
            test = str(example.source_metadata["test"])
            entry_point = str(example.source_metadata["entry_point"])
            for completion_index, completion in enumerate(completions):
                candidate = example.prompt + completion
                programs.append(f"{candidate}\n{test}\ncheck({entry_point})")
                owners.append((problem_index, completion_index))
        execution_results = self._run_candidates(programs)
        by_problem: list[list[dict[str, Any]]] = [
            [dict() for _ in range(self.samples_per_problem)] for _ in dataset
        ]
        for owner, result in zip(owners, execution_results, strict=True):
            by_problem[owner[0]][owner[1]] = result

        correct_counts = [sum(bool(result["passed"]) for result in rows) for rows in by_problem]
        metrics = aggregate_pass_at_k(
            correct_counts,
            samples_per_problem=self.samples_per_problem,
        )
        samples = tuple(
            {
                "sample_id": example.sample_id,
                "task_id": example.source_metadata["task_id"],
                "correct_completions": correct_counts[index],
                "completion_count": self.samples_per_problem,
                "completions": [
                    {
                        "text": completion,
                        **by_problem[index][completion_index],
                    }
                    for completion_index, completion in enumerate(all_completions[index])
                ],
            }
            for index, example in enumerate(dataset)
        )
        return EvaluationResult(
            protocol=self.protocol,
            dataset_manifest=dataset.manifest,
            aggregate={
                "task_name": "humaneval",
                "metric_name": "pass@1",
                "sample_count": len(dataset),
                "samples_per_problem": self.samples_per_problem,
                "generation_sequence_limit": batch_size,
                "generation_calls": generation_calls,
                "sampling_batching": "bounded_sequences_v1",
                "sampling_seed": self.seed,
                **metrics,
            },
            samples=samples,
        )


def _serialized_code_tokens(tokens: list[str]) -> tuple[str, list[tuple[int, int]]]:
    parts: list[str] = []
    spans: list[tuple[int, int]] = []
    cursor = 0
    for index, token in enumerate(tokens):
        if index:
            parts.append(" ")
            cursor += 1
        start = cursor
        parts.append(token)
        cursor += len(token)
        spans.append((start, cursor))
    return "".join(parts), spans


def _overlapping_token_index(
    offset: tuple[int, int],
    spans: list[tuple[int, int]],
) -> int | None:
    start, end = offset
    overlaps = [
        (min(end, span_end) - max(start, span_start), index)
        for index, (span_start, span_end) in enumerate(spans)
        if min(end, span_end) > max(start, span_start)
    ]
    return max(overlaps)[1] if overlaps else None


class CodeXGLUETokenCompletionEvaluator(ClassificationEvaluator):
    IGNORED_TOKENS = frozenset({"<s>", "</s>", "<EOL>", "<pad>"})

    def __init__(self, task_name: str, *, max_input_length: int = 512) -> None:
        if task_name not in {"codexglue_python", "codexglue_java"}:
            raise ValueError("CodeXGLUE task must be python or java")
        self.task_name = task_name
        super().__init__(
            EvaluationProtocol(
                task_name=task_name,
                evaluator_type="classification",
                metric_name="token_accuracy",
                prompt_template="space-separated official CodeXGLUE code tokens",
                prompt_template_version="codexglue-token-accuracy-v1",
                max_input_length=max_input_length,
                max_new_tokens=1,
            )
        )

    def _score_row(self, model, tokenizer, code_tokens: list[str]) -> dict[str, int]:
        text, spans = _serialized_code_tokens(code_tokens)
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            return_attention_mask=False,
        )
        input_ids = list(encoded["input_ids"])
        offsets = [tuple(pair) for pair in encoded["offset_mapping"]]
        if len(input_ids) < 2:
            return {
                "correct_tokens": 0,
                "scored_tokens": 0,
                "correct_subtokens": 0,
                "scored_subtokens": 0,
            }
        device = _model_device(model)
        predicted_correct: dict[int, list[bool]] = {}
        start = 0
        while start < len(input_ids) - 1:
            end = min(start + self.protocol.max_input_length, len(input_ids))
            block = torch.tensor([input_ids[start:end]], dtype=torch.long, device=device)
            with torch.inference_mode():
                logits = model(input_ids=block, attention_mask=torch.ones_like(block)).logits
            predictions = logits[:, :-1].argmax(dim=-1).squeeze(0).cpu().tolist()
            for local_target, prediction in enumerate(predictions, start=1):
                global_target = start + local_target
                official_index = _overlapping_token_index(offsets[global_target], spans)
                if official_index is None:
                    continue
                predicted_correct.setdefault(official_index, []).append(
                    int(prediction) == input_ids[global_target]
                )
            if end == len(input_ids):
                break
            start = end - 1

        correct_tokens = 0
        scored_tokens = 0
        correct_subtokens = 0
        scored_subtokens = 0
        for index, token in enumerate(code_tokens):
            if token in self.IGNORED_TOKENS:
                continue
            outcomes = predicted_correct.get(index, [])
            if not outcomes:
                continue
            scored_tokens += 1
            correct_tokens += int(all(outcomes))
            scored_subtokens += len(outcomes)
            correct_subtokens += sum(outcomes)
        return {
            "correct_tokens": correct_tokens,
            "scored_tokens": scored_tokens,
            "correct_subtokens": correct_subtokens,
            "scored_subtokens": scored_subtokens,
        }

    def evaluate(
        self,
        model,
        tokenizer,
        dataset: LogicalDataset,
        *,
        batch_size: int,
    ) -> EvaluationResult:
        del batch_size
        was_training = model.training
        model.eval()
        samples: list[dict[str, Any]] = []
        totals = {
            "correct_tokens": 0,
            "scored_tokens": 0,
            "correct_subtokens": 0,
            "scored_subtokens": 0,
        }
        try:
            for example in tqdm(
                dataset,
                desc=f"{self.task_name} token accuracy",
                unit="file",
                dynamic_ncols=True,
            ):
                metrics = self._score_row(
                    model,
                    tokenizer,
                    [str(token) for token in example.source_metadata["code_tokens"]],
                )
                for key, value in metrics.items():
                    totals[key] += value
                samples.append({"sample_id": example.sample_id, **metrics})
        finally:
            model.train(was_training)
        if totals["scored_tokens"] == 0 or totals["scored_subtokens"] == 0:
            raise RuntimeError("CodeXGLUE evaluation found no scored tokens")
        return EvaluationResult(
            protocol=self.protocol,
            dataset_manifest=dataset.manifest,
            aggregate={
                "task_name": self.task_name,
                "metric_name": "token_accuracy",
                "sample_count": len(dataset),
                "token_accuracy": totals["correct_tokens"] / totals["scored_tokens"],
                "model_subtoken_accuracy": (
                    totals["correct_subtokens"] / totals["scored_subtokens"]
                ),
                **totals,
            },
            samples=tuple(samples),
        )
