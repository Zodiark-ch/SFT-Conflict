"""Versioned training and evaluation adapters for code-generation datasets."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .registry import RenderedExample

CODE_SOURCE_NAMES = frozenset(
    {
        "glaiveai/glaive-code-assistant-v3",
        "replete-ai/code_bagel_hermes-2.5",
        "m-a-p/codefeedback-filtered-instruction",
        "bigcode/self-oss-instruct-sc2-exec-filter-50k",
        "codefuse-ai/codeexercise-python-27k",
        "nickrosh/evol-instruct-code-80k-v1",
        "jinaai/code_exercises",
        "tokenbender/code_instructions_122k_alpaca_style",
        "iamtarun/python_code_instructions_18k_alpaca",
        "nan-do/instructional_code-search-net-python",
        "safurai/code-instruct-700k",
        "ajibawa-2023/python-code-23k-sharegpt",
        "jtatman/python-code-dataset-500k",
        "m-a-p/code-feedback",
        "code alpaca 20k",
        "bagel",
    }
)


def _normalized_source(value: Any) -> str:
    return str(value or "").strip().casefold()


def _flatten_label_strings(value: Any) -> list[str]:
    if isinstance(value, Mapping):
        return [text for child in value.values() for text in _flatten_label_strings(child)]
    if isinstance(value, Sequence) and not isinstance(value, str):
        return [text for child in value for text in _flatten_label_strings(child)]
    return [str(value).casefold()]


def is_infinity_code_row(row: Mapping[str, Any]) -> bool:
    source = _normalized_source(row.get("source"))
    if source in CODE_SOURCE_NAMES:
        return True
    labels = " ".join(_flatten_label_strings(row.get("label", {})))
    return any(marker in labels for marker in ("code generation", "coding", "programming"))


def _alpaca_prompt(instruction: str, input_text: str) -> str:
    if input_text.strip():
        return (
            "Below is an instruction that describes a task, paired with an input that "
            "provides further context. Write a response that appropriately completes the "
            "request.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            "### Response:\n"
        )
    return (
        "Below is an instruction that describes a task. Write a response that appropriately "
        "completes the request.\n\n"
        f"### Instruction:\n{instruction}\n\n"
        "### Response:\n"
    )


def _render_alpaca(row: Mapping[str, Any], *, task_name: str) -> RenderedExample:
    instruction = str(row["instruction"])
    input_text = str(row.get("input") or "")
    output = str(row["output"])
    return RenderedExample(
        prompt=_alpaca_prompt(instruction, input_text),
        response=output,
        metadata={
            "task_name": task_name,
            "template_id": "alpaca-instruction-v1",
            "has_input": bool(input_text.strip()),
        },
    )


def render_codealpaca(row: Mapping[str, Any]) -> RenderedExample:
    return _render_alpaca(row, task_name="codealpaca")


def render_code_bagel_hermes(row: Mapping[str, Any]) -> RenderedExample:
    return _render_alpaca(row, task_name="code_bagel_hermes")


def _conversation_role(turn: Mapping[str, Any]) -> str:
    return str(turn.get("from", turn.get("role", ""))).strip().casefold()


def _conversation_text(turn: Mapping[str, Any]) -> str:
    value = turn.get("value", turn.get("content"))
    if value is None:
        raise KeyError("Infinity-Instruct conversation turns require value or content")
    return str(value)


def render_infinity_instruct_code(row: Mapping[str, Any]) -> RenderedExample:
    conversations = row.get("conversations")
    if not isinstance(conversations, Sequence) or isinstance(conversations, str):
        raise TypeError("Infinity-Instruct rows require a conversations sequence")
    turns = [turn for turn in conversations if isinstance(turn, Mapping)]
    if len(turns) < 2 or _conversation_role(turns[-1]) not in {"gpt", "assistant"}:
        raise ValueError("Infinity-Instruct code rows must end with an assistant response")

    prompt_parts: list[str] = []
    for turn in turns[:-1]:
        role = _conversation_role(turn)
        if role in {"human", "user"}:
            label = "Query"
        elif role in {"gpt", "assistant"}:
            label = "Response"
        elif role == "system":
            label = "System"
        else:
            raise ValueError(f"unsupported Infinity-Instruct conversation role: {role!r}")
        prompt_parts.append(f"{label}:\n{_conversation_text(turn)}")
    prompt_parts.append("Response:\n")

    return RenderedExample(
        prompt="\n\n".join(prompt_parts),
        response=_conversation_text(turns[-1]),
        metadata={
            "task_name": "infinity_instruct_code",
            "template_id": "infinity-query-response-v1",
            "source": row.get("source"),
            "label": row.get("label"),
            "conversation_turn_count": len(turns),
        },
    )


def render_humaneval(row: Mapping[str, Any]) -> RenderedExample:
    return RenderedExample(
        prompt=str(row["prompt"]).strip(),
        response=str(row["canonical_solution"]),
        metadata={
            "task_name": "humaneval",
            "task_id": str(row["task_id"]),
            "test": str(row["test"]),
            "entry_point": str(row["entry_point"]),
            "template_id": "lori-humaneval-completion-v1",
        },
    )


def render_codexglue_token_completion(row: Mapping[str, Any]) -> RenderedExample:
    code_tokens = [str(token) for token in row["code"]]
    if len(code_tokens) < 2:
        raise ValueError("CodeXGLUE token-completion rows require at least two tokens")
    return RenderedExample(
        prompt=" ".join(code_tokens[:-1]),
        response=" " + code_tokens[-1],
        metadata={
            "task_name": "codexglue_token_completion",
            "code_tokens": code_tokens,
            "path": row.get("path"),
            "source_id": int(row["id"]),
            "template_id": "codexglue-pretokenized-v1",
        },
    )
