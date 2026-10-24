"""Versioned adapters for instruction following and logical reasoning."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from .registry import RenderedExample

LOGIQA_LABELS = ("A", "B", "C", "D")


def dare_wizardlm_prompt(instruction: str) -> str:
    return (
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's "
        f"questions. USER: {instruction} ASSISTANT:"
    )


def render_evol_instruct(row: Mapping[str, Any]) -> RenderedExample:
    instruction = str(row["instruction"])
    output = str(row["output"])
    return RenderedExample(
        prompt=f"{instruction}\n\n### Response:",
        response=output,
        metadata={
            "task_name": "evol_instruct",
            "instruction": instruction,
            "template_id": "wizardlm-train-freeform-v1",
        },
    )


def render_alpaca_eval(row: Mapping[str, Any]) -> RenderedExample:
    instruction = str(row["instruction"])
    return RenderedExample(
        prompt=dare_wizardlm_prompt(instruction),
        response=" " + str(row["output"]).lstrip(),
        metadata={
            "task_name": "alpaca_eval",
            "instruction": instruction,
            "dataset": str(row["dataset"]),
            "reference_generator": str(row["generator"]),
            "template_id": "dare-wizardlm-chat-v1",
        },
    )


def _logiqa_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    value = row.get("text")
    if "options" in row and "answer" in row:
        payload = dict(row)
    elif isinstance(value, str):
        payload = json.loads(value)
    elif isinstance(value, Mapping):
        payload = dict(value)
    else:
        payload = dict(row)
    options = payload.get("options")
    if not isinstance(options, Sequence) or isinstance(options, str) or len(options) != 4:
        raise ValueError("LogiQA2 rows require exactly four options")
    answer = int(payload["answer"])
    if answer not in range(4):
        raise ValueError("LogiQA2 answer must be an index from zero to three")
    return payload


def render_logiqa2(row: Mapping[str, Any]) -> RenderedExample:
    payload = _logiqa_payload(row)
    options = [str(option) for option in payload["options"]]
    option_text = "\n".join(
        f"{label}. {option}" for label, option in zip(LOGIQA_LABELS, options, strict=True)
    )
    prompt = f"Passage: {payload['text']}\nQuestion: {payload['question']}\n{option_text}\nAnswer:"
    answer = int(payload["answer"])
    return RenderedExample(
        prompt=prompt,
        response=f" {LOGIQA_LABELS[answer]}",
        metadata={
            "task_name": "logiqa2",
            "source_id": int(payload["id"]),
            "gold_label": answer,
            "label_candidates": list(LOGIQA_LABELS),
            "reasoning_types": payload.get("type", {}),
            "template_id": "logiqa2-zero-shot-mrc-v1",
        },
    )
