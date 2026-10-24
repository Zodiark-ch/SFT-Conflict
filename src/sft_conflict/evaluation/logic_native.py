"""Native LogiQA2 and AlpacaEval protocols for decoder-only models."""

from __future__ import annotations

import math
from typing import Any

import torch
from tqdm.auto import tqdm

from sft_conflict.data import LogicalDataset

from .native import (
    ClassificationEvaluator,
    EvaluationProtocol,
    EvaluationResult,
    GenerationEvaluator,
    _batched,
    _model_device,
    _validated_result,
)


class LogiQA2Evaluator(ClassificationEvaluator):
    RESPONSES = (" A", " B", " C", " D")

    def __init__(self, *, max_input_length: int = 512) -> None:
        super().__init__(
            EvaluationProtocol(
                task_name="logiqa2",
                evaluator_type="classification",
                metric_name="accuracy",
                prompt_template=(
                    "Passage: {text}\\nQuestion: {question}\\n"
                    "A. ...\\nB. ...\\nC. ...\\nD. ...\\nAnswer:"
                ),
                prompt_template_version="logiqa2-zero-shot-mrc-v1",
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
        was_training = model.training
        model.eval()
        try:
            indexed_examples = list(enumerate(dataset))
            for chunk in tqdm(
                _batched(indexed_examples, batch_size),
                total=math.ceil(len(indexed_examples) / batch_size),
                desc="LogiQA2 zero-shot evaluation",
                unit="batch",
                dynamic_ncols=True,
            ):
                score_rows = self._score_candidates(
                    model,
                    tokenizer,
                    [example.prompt for _, example in chunk],
                )
                for (_, example), scores in zip(chunk, score_rows, strict=True):
                    gold = int(example.source_metadata["gold_label"])
                    prediction = max(range(len(scores)), key=scores.__getitem__)
                    samples.append(
                        {
                            "sample_id": example.sample_id,
                            "source_id": example.source_metadata.get("source_id"),
                            "prediction": prediction,
                            "prediction_text": self.RESPONSES[prediction].strip(),
                            "gold": gold,
                            "gold_text": self.RESPONSES[gold].strip(),
                            "correct": prediction == gold,
                            "candidate_loglikelihoods": {
                                response.strip(): score
                                for response, score in zip(self.RESPONSES, scores, strict=True)
                            },
                            "template_id": example.source_metadata.get("template_id"),
                        }
                    )
        finally:
            model.train(was_training)
        accuracy = sum(bool(sample["correct"]) for sample in samples) / len(samples)
        return _validated_result(self.protocol, dataset, accuracy, samples)

    def _score_candidates(self, model, tokenizer, prompts: list[str]) -> list[list[float]]:
        rows: list[list[int]] = []
        candidate_lengths: list[int] = []
        for prompt in prompts:
            prompt_ids = list(
                tokenizer(prompt, add_special_tokens=True, verbose=False)["input_ids"]
            )
            response_rows = [
                list(tokenizer(response, add_special_tokens=False, verbose=False)["input_ids"])
                for response in self.RESPONSES
            ]
            if any(not response_ids for response_ids in response_rows):
                raise ValueError("LogiQA2 candidate produced no tokens")
            prompt_budget = self.protocol.max_input_length - max(map(len, response_rows))
            if prompt_budget <= 0:
                raise ValueError("LogiQA2 max_input_length leaves no prompt capacity")
            if len(prompt_ids) > prompt_budget:
                head_length = prompt_budget // 2
                tail_length = prompt_budget - head_length
                selected_prompt = prompt_ids[:head_length] + prompt_ids[-tail_length:]
            else:
                selected_prompt = prompt_ids
            for response_ids in response_rows:
                rows.append(selected_prompt + response_ids)
                candidate_lengths.append(len(response_ids))

        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            raise ValueError("LogiQA2 evaluation requires a tokenizer pad token")
        width = max(len(row) for row in rows)
        input_ids = torch.full((len(rows), width), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        completion_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for index, (row, candidate_length) in enumerate(zip(rows, candidate_lengths, strict=True)):
            input_ids[index, : len(row)] = torch.tensor(row, dtype=torch.long)
            attention_mask[index, : len(row)] = 1
            completion_mask[index, len(row) - candidate_length : len(row)] = True

        device = _model_device(model)
        with torch.inference_mode():
            logits = (
                model(
                    input_ids=input_ids.to(device),
                    attention_mask=attention_mask.to(device),
                )
                .logits[:, :-1, :]
                .float()
            )
        targets = input_ids[:, 1:].to(device)
        selected = completion_mask[:, 1:].to(device)
        token_scores = (
            torch.log_softmax(logits, dim=-1)
            .gather(dim=-1, index=targets.unsqueeze(-1))
            .squeeze(-1)
        )
        scores = (token_scores * selected).sum(dim=-1).cpu().tolist()
        candidate_count = len(self.RESPONSES)
        return [
            scores[index : index + candidate_count]
            for index in range(0, len(scores), candidate_count)
        ]


class AlpacaEvalGenerationEvaluator(GenerationEvaluator):
    def __init__(
        self,
        *,
        max_input_length: int = 512,
        max_new_tokens: int = 2048,
        generator_name: str = "sft-conflict-model",
    ) -> None:
        self.generator_name = generator_name
        super().__init__(
            EvaluationProtocol(
                task_name="alpaca_eval",
                evaluator_type="generation",
                metric_name="official_judge_pending",
                prompt_template=(
                    "A chat between a curious user and an artificial intelligence assistant. "
                    "The assistant gives helpful, detailed, and polite answers to the user's "
                    "questions. USER: {instruction} ASSISTANT:"
                ),
                prompt_template_version="dare-wizardlm-chat-v1",
                max_input_length=max(1024, max_input_length),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
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
        device = _model_device(model)
        samples: list[dict[str, Any]] = []
        was_training = model.training
        original_padding_side = tokenizer.padding_side
        original_truncation_side = tokenizer.truncation_side
        tokenizer.padding_side = "left"
        tokenizer.truncation_side = "left"
        model.eval()
        try:
            indexed_examples = list(enumerate(dataset))
            for chunk in tqdm(
                _batched(indexed_examples, batch_size),
                total=math.ceil(len(indexed_examples) / batch_size),
                desc="AlpacaEval generation",
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
                outputs = tokenizer.batch_decode(
                    generated[:, prompt_width:].cpu(),
                    skip_special_tokens=True,
                )
                for (_, example), output in zip(chunk, outputs, strict=True):
                    samples.append(
                        {
                            "sample_id": example.sample_id,
                            "instruction": str(example.source_metadata["instruction"]),
                            "output": output.strip(),
                            "generator": self.generator_name,
                            "dataset": str(example.source_metadata["dataset"]),
                        }
                    )
        finally:
            tokenizer.padding_side = original_padding_side
            tokenizer.truncation_side = original_truncation_side
            model.train(was_training)
        if not samples or len(samples) != len(dataset):
            raise RuntimeError(
                f"AlpacaEval generated {len(samples)} outputs for {len(dataset)} instructions"
            )
        return EvaluationResult(
            protocol=self.protocol,
            dataset_manifest=dataset.manifest,
            aggregate={
                "task_name": "alpaca_eval",
                "metric_name": "official_judge_pending",
                "official_judge_pending": "model_outputs.json",
                "sample_count": len(samples),
                "generator": self.generator_name,
                "annotators_config_dare": "chatgpt_fn",
                "annotators_config_recommended": "weighted_alpaca_eval_gpt4_turbo",
            },
            samples=tuple(samples),
        )
