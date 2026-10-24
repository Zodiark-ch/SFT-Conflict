"""Random console previews decoded from model-ready input and label tensors."""

from __future__ import annotations

import secrets
from typing import Any

from torch import Tensor

from sft_conflict.data import CausalLMCollator
from sft_conflict.data.loading import tokenize_prompt_response


def _decoded_tensor_rows(
    batch: dict[str, Any],
    tokenizer: Any,
    *,
    dataset_indices: list[int],
    labels_aligned_to_inputs: bool,
) -> list[dict[str, Any]]:
    input_ids = batch.get("input_ids")
    attention_mask = batch.get("attention_mask")
    labels = batch.get("labels")
    if not all(isinstance(value, Tensor) for value in (input_ids, attention_mask, labels)):
        raise TypeError("tensor preview requires input_ids, attention_mask, and labels tensors")
    if input_ids.shape != attention_mask.shape:
        raise ValueError("tensor preview input_ids and attention_mask must align")
    if labels.ndim != 2 or labels.shape[0] != input_ids.shape[0]:
        raise ValueError("tensor preview labels must share the input batch dimension")
    if labels_aligned_to_inputs and labels.shape != input_ids.shape:
        raise ValueError("aligned tensor preview labels must match input_ids shape")
    if input_ids.shape[0] != len(dataset_indices):
        raise ValueError("tensor preview dataset indices must align with the batch")

    rows: list[dict[str, Any]] = []
    sample_ids = batch.get("sample_ids", [None] * input_ids.shape[0])
    for row_index, dataset_index in enumerate(dataset_indices):
        attended = attention_mask[row_index].bool()
        supervised = labels[row_index].ne(-100)
        input_tensor = input_ids[row_index].masked_select(attended).detach().cpu()
        label_tensor = labels[row_index].masked_select(supervised).detach().cpu()
        if labels_aligned_to_inputs and bool(supervised.any()):
            first_supervised_position = int(supervised.nonzero(as_tuple=False)[0].item())
            positions = Tensor(list(range(input_ids.shape[1]))).long()
            prefix_mask = attended.cpu() & positions.lt(first_supervised_position)
            prediction_prefix_tensor = (
                input_ids[row_index].detach().cpu().masked_select(prefix_mask)
            )
        else:
            prediction_prefix_tensor = input_tensor
        rows.append(
            {
                "dataset_index": dataset_index,
                "sample_id": sample_ids[row_index],
                "input_text": tokenizer.decode(
                    input_tensor.tolist(),
                    skip_special_tokens=True,
                ),
                "prediction_prefix_text": tokenizer.decode(
                    prediction_prefix_tensor.tolist(),
                    skip_special_tokens=True,
                ),
                "label_text": tokenizer.decode(
                    label_tensor.tolist(),
                    skip_special_tokens=True,
                ),
                "input_token_count": int(input_tensor.numel()),
                "prediction_prefix_token_count": int(prediction_prefix_tensor.numel()),
                "label_token_count": int(label_tensor.numel()),
                "labels_aligned_to_inputs": labels_aligned_to_inputs,
            }
        )
    return rows


def build_random_tensor_preview(
    dataset: Any,
    tokenizer: Any,
    *,
    stage: str,
    pad_token_id: int,
    pad_to_multiple_of: int | None,
    count: int = 3,
    random_source: Any | None = None,
) -> dict[str, Any]:
    if count <= 0:
        raise ValueError("tensor preview count must be positive")
    if len(dataset) == 0:
        raise ValueError("cannot preview an empty dataset")
    sampler = random_source or secrets.SystemRandom()
    sample_count = min(count, len(dataset))
    dataset_indices = sampler.sample(range(len(dataset)), sample_count)
    records = [dataset[index] for index in dataset_indices]
    return _build_tensor_preview_from_records(
        records,
        tokenizer,
        stage=stage,
        pad_token_id=pad_token_id,
        pad_to_multiple_of=pad_to_multiple_of,
        dataset_indices=dataset_indices,
    )


def _build_tensor_preview_from_records(
    records: list[dict[str, Any]],
    tokenizer: Any,
    *,
    stage: str,
    pad_token_id: int,
    pad_to_multiple_of: int | None,
    dataset_indices: list[int],
) -> dict[str, Any]:
    batch = CausalLMCollator(
        pad_token_id=pad_token_id,
        pad_to_multiple_of=pad_to_multiple_of,
    )(records)
    return _build_tensor_preview_from_batch(
        batch,
        tokenizer,
        stage=stage,
        source="CausalLMCollator tensors; no DataLoader",
        dataset_indices=dataset_indices,
        labels_aligned_to_inputs=True,
    )


def _build_tensor_preview_from_batch(
    batch: dict[str, Any],
    tokenizer: Any,
    *,
    stage: str,
    source: str,
    dataset_indices: list[int],
    labels_aligned_to_inputs: bool,
) -> dict[str, Any]:
    return {
        "stage": stage,
        "source": source,
        "input_tensor_shape": list(batch["input_ids"].shape),
        "label_tensor_shape": list(batch["labels"].shape),
        "samples": _decoded_tensor_rows(
            batch,
            tokenizer,
            dataset_indices=dataset_indices,
            labels_aligned_to_inputs=labels_aligned_to_inputs,
        ),
    }


def build_random_logical_tensor_preview(
    dataset: Any,
    tokenizer: Any,
    *,
    stage: str,
    max_length: int,
    model_variant: str,
    evaluator_type: str,
    pad_to_multiple_of: int | None,
    count: int = 3,
    random_source: Any | None = None,
) -> dict[str, Any]:
    if count <= 0:
        raise ValueError("tensor preview count must be positive")
    if len(dataset) == 0:
        raise ValueError("cannot preview an empty dataset")
    sampler = random_source or secrets.SystemRandom()
    sample_count = min(count, len(dataset))
    dataset_indices = sampler.sample(range(len(dataset)), sample_count)
    examples = [dataset[index] for index in dataset_indices]
    if evaluator_type == "classification" and all(
        "code_tokens" in example.source_metadata for example in examples
    ):
        texts = [
            " ".join(str(token) for token in example.source_metadata["code_tokens"])
            for example in examples
        ]
        encoded = tokenizer(
            texts,
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_attention_mask=True,
            return_tensors="pt",
        )
        labels = encoded["input_ids"].clone()
        labels.masked_fill_(encoded["attention_mask"].eq(0), -100)
        labels[:, 0] = -100
        return _build_tensor_preview_from_batch(
            {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
                "labels": labels,
                "sample_ids": [example.sample_id for example in examples],
            },
            tokenizer,
            stage=stage,
            source="CodeXGLUE next-token evaluation window tensors; no DataLoader",
            dataset_indices=dataset_indices,
            labels_aligned_to_inputs=True,
        )
    if evaluator_type == "generation":
        input_batch = tokenizer(
            [example.prompt for example in examples],
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_attention_mask=True,
            return_tensors="pt",
        )
        label_batch = tokenizer(
            [example.response for example in examples],
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_attention_mask=True,
            return_tensors="pt",
        )
        labels = label_batch["input_ids"].clone()
        labels.masked_fill_(label_batch["attention_mask"].eq(0), -100)
        batch = {
            "input_ids": input_batch["input_ids"],
            "attention_mask": input_batch["attention_mask"],
            "labels": labels,
            "sample_ids": [example.sample_id for example in examples],
        }
        return _build_tensor_preview_from_batch(
            batch,
            tokenizer,
            stage=stage,
            source="native generation input/label tensors; no DataLoader",
            dataset_indices=dataset_indices,
            labels_aligned_to_inputs=False,
        )
    if evaluator_type != "classification":
        raise ValueError("native evaluator_type must be classification or generation")
    records: list[dict[str, Any]] = []
    for example in examples:
        template_id = example.source_metadata.get("template_id")
        record = tokenize_prompt_response(
            tokenizer,
            example.prompt,
            example.response,
            max_length=max_length,
            model_variant=model_variant,
            preserve_response=template_id == "logiqa2-zero-shot-mrc-v1",
            preserve_prompt_tail=template_id == "logiqa2-zero-shot-mrc-v1",
        )
        record["sample_id"] = example.sample_id
        records.append(record)
    return _build_tensor_preview_from_records(
        records,
        tokenizer,
        stage=stage,
        pad_token_id=tokenizer.pad_token_id,
        pad_to_multiple_of=pad_to_multiple_of,
        dataset_indices=dataset_indices,
    )


def render_tensor_preview(preview: dict[str, Any]) -> str:
    lines = [
        "",
        "=" * 88,
        f"RANDOM {str(preview['stage']).upper()} MODEL TENSOR SAMPLES",
        f"Source: {preview['source']}",
        (
            f"Tensor shapes: input_ids={preview['input_tensor_shape']}, "
            f"labels={preview['label_tensor_shape']}"
        ),
    ]
    for number, sample in enumerate(preview["samples"], start=1):
        lines.extend(
            [
                "-" * 88,
                (
                    f"Sample {number}: dataset_index={sample['dataset_index']} "
                    f"sample_id={sample['sample_id']}"
                ),
            ]
        )
        if sample["labels_aligned_to_inputs"]:
            lines.extend(
                [
                    "FULL INPUT TEXT (teacher-forced input_ids tensor):",
                    "  " + sample["input_text"].replace("\n", "\n  "),
                    "FIRST-LABEL PREDICTION PREFIX (label token excluded):",
                    "  " + sample["prediction_prefix_text"].replace("\n", "\n  "),
                ]
            )
        else:
            lines.extend(
                [
                    "MODEL INPUT TEXT (generation prompt input_ids tensor; label excluded):",
                    "  " + sample["input_text"].replace("\n", "\n  "),
                ]
            )
        lines.extend(
            [
                "LABEL TEXT (decoded from labels tensor after removing -100):",
                "  " + sample["label_text"].replace("\n", "\n  "),
                (
                    f"TOKENS: full_input={sample['input_token_count']}, "
                    f"first_label_prefix={sample['prediction_prefix_token_count']}, "
                    f"label={sample['label_token_count']}"
                ),
            ]
        )
    lines.extend(["=" * 88, ""])
    return "\n".join(lines)
