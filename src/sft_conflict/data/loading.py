"""Offline JSONL/Hugging Face adapters and response-only tokenization."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from datasets import DownloadConfig, load_dataset
from huggingface_hub import try_to_load_from_cache
from tqdm.auto import tqdm

from sft_conflict.config import DatasetConfig

from .records import LogicalDataset, LogicalExample, TokenizedDataset, make_sample_id
from .registry import DEFAULT_DATASET_REGISTRY


class NoSupervisedTokensError(ValueError):
    """Raised when truncation removes every supervised response token."""


def _numeric_variant_token_ids(
    tokenizer: Any,
    texts: list[str],
    *,
    component: str,
) -> tuple[list[int], int, tuple[int, ...]]:
    sequences = [list(tokenizer(text, add_special_tokens=False)["input_ids"]) for text in texts]
    if not sequences or len({len(sequence) for sequence in sequences}) != 1:
        raise ValueError(
            f"STS-B numeric-token MSE requires fixed-length {component} response tokens"
        )
    varying_positions = [
        index
        for index in range(len(sequences[0]))
        if len({sequence[index] for sequence in sequences}) > 1
    ]
    if len(varying_positions) != 1:
        raise ValueError(f"STS-B numeric-token MSE requires exactly one varying {component} token")
    position = varying_positions[0]
    token_ids = tuple(int(sequence[position]) for sequence in sequences)
    if len(set(token_ids)) != len(token_ids):
        raise ValueError(f"STS-B numeric-token MSE requires distinct {component} token IDs")
    return sequences[0], position, token_ids


def _stsb_regression_fields(
    tokenizer: Any,
    labels: list[int],
    *,
    label_text: str,
) -> dict[str, Any]:
    integer_text, separator, fraction_text = label_text.partition(".")
    if separator != "." or len(integer_text) != 1 or len(fraction_text) != 1:
        raise ValueError(f"STS-B labels must use one-decimal notation, got {label_text!r}")
    integer_baseline, integer_offset, integer_token_ids = _numeric_variant_token_ids(
        tokenizer,
        [f" {value}.0" for value in range(6)],
        component="integer",
    )
    fraction_baseline, fraction_offset, fraction_token_ids = _numeric_variant_token_ids(
        tokenizer,
        [f" 0.{value}" for value in range(10)],
        component="fractional",
    )
    if integer_baseline != fraction_baseline or integer_offset == fraction_offset:
        raise ValueError("STS-B numeric-token MSE found an inconsistent numeric response layout")
    response_token_ids = list(tokenizer(f" {label_text}", add_special_tokens=False)["input_ids"])
    expected_response = list(integer_baseline)
    expected_response[integer_offset] = integer_token_ids[int(integer_text)]
    expected_response[fraction_offset] = fraction_token_ids[int(fraction_text)]
    if response_token_ids != expected_response:
        raise ValueError("STS-B numeric-token MSE found an inconsistent rendered numeric response")
    matches = [
        index
        for index in range(max(0, len(labels) - len(response_token_ids) + 1))
        if labels[index : index + len(response_token_ids)] == response_token_ids
    ]
    if not matches:
        raise NoSupervisedTokensError(
            "truncation removed part of the supervised STS-B numeric response"
        )
    if len(matches) != 1:
        raise ValueError("STS-B numeric response token pattern must occur exactly once")
    integer_position = matches[0] + integer_offset
    fraction_position = matches[0] + fraction_offset
    if integer_position == 0:
        raise ValueError("STS-B integer token has no causal prediction position")
    return {
        "regression_label": float(label_text),
        "regression_positions": [integer_position, fraction_position],
        "regression_integer_token_ids": list(integer_token_ids),
        "regression_fraction_token_ids": list(fraction_token_ids),
    }


def stratified_train_validation_indices(
    labels: Sequence[int | float],
    *,
    validation_fraction: float,
    seed: int,
    regression: bool,
) -> tuple[list[int], list[int]]:
    if not labels:
        raise ValueError("cannot split an empty dataset")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")

    groups: dict[str, list[int]] = {}
    for index, label in enumerate(labels):
        if regression:
            score_bin = min(9, max(0, math.floor(float(label) * 2.0)))
            key = f"score-bin-{score_bin}"
        else:
            key = f"label-{int(label)}"
        groups.setdefault(key, []).append(index)

    shuffled: dict[str, list[int]] = {}
    quotas: dict[str, int] = {}
    remainders: dict[str, float] = {}
    for key, indices in sorted(groups.items()):
        values = list(indices)
        random.Random(f"{seed}:{key}").shuffle(values)
        shuffled[key] = values
        raw_quota = len(values) * validation_fraction
        maximum = max(0, len(values) - 1)
        quotas[key] = min(math.floor(raw_quota), maximum)
        remainders[key] = raw_quota - math.floor(raw_quota)

    target_validation_count = min(
        int(len(labels) * validation_fraction + 0.5),
        sum(max(0, len(values) - 1) for values in shuffled.values()),
    )
    remaining = target_validation_count - sum(quotas.values())
    candidates = sorted(
        shuffled,
        key=lambda key: (-remainders[key], key),
    )
    while remaining > 0:
        changed = False
        for key in candidates:
            if quotas[key] < max(0, len(shuffled[key]) - 1):
                quotas[key] += 1
                remaining -= 1
                changed = True
                if remaining == 0:
                    break
        if not changed:
            break

    validation_indices = [
        index for key in sorted(shuffled) for index in shuffled[key][: quotas[key]]
    ]
    validation_set = set(validation_indices)
    train_indices = [index for index in range(len(labels)) if index not in validation_set]
    random.Random(seed + 10_003).shuffle(train_indices)
    random.Random(seed + 20_003).shuffle(validation_indices)
    return train_indices, validation_indices


def _selected_hf_indices(dataset: Any, config: DatasetConfig) -> list[int]:
    if config.selection == "all":
        return list(range(len(dataset)))
    if config.selection == "sample_n":
        if config.max_samples is None:
            raise ValueError("sample_n selection requires max_samples")
        sample_count = min(config.max_samples, len(dataset))
        return random.Random(config.selection_seed).sample(range(len(dataset)), sample_count)
    if config.stratify_column is None:
        raise ValueError(f"dataset selection '{config.selection}' requires stratify_column")
    labels = dataset[config.stratify_column]
    train_indices, validation_indices = stratified_train_validation_indices(
        labels,
        validation_fraction=config.validation_fraction,
        seed=config.selection_seed,
        regression=config.stratify_regression,
    )
    return train_indices if config.selection == "train_90" else validation_indices


def _sample_namespace(config: DatasetConfig, template_version: str) -> str:
    return json.dumps(
        {
            "source": config.source,
            "path": config.path,
            "revision": config.revision,
            "name": config.name,
            "data_file": config.data_file,
            "split": config.split,
            "adapter": config.adapter,
            "template_version": template_version,
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def _load_metadata_sidecar(
    path: str | None,
) -> tuple[dict[str, dict[str, Any]], dict[int, dict[str, Any]], str | None]:
    if path is None:
        return {}, {}, None
    source = Path(path).expanduser().resolve()
    by_sample_id: dict[str, dict[str, Any]] = {}
    by_source_index: dict[int, dict[str, Any]] = {}
    digest = hashlib.sha256()
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            digest.update(line.encode())
            try:
                metadata = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid metadata JSON at {source}:{line_number}: {exc}") from exc
            sample_id = metadata.pop("sample_id", None)
            source_index = metadata.pop("source_index", None)
            if sample_id is None and source_index is None:
                raise ValueError(
                    f"metadata sidecar {source}:{line_number} needs sample_id or source_index"
                )
            if sample_id is not None:
                key = str(sample_id)
                if key in by_sample_id:
                    raise ValueError(f"duplicate metadata sidecar sample_id: {key}")
                by_sample_id[key] = dict(metadata)
            if source_index is not None:
                key = int(source_index)
                if key in by_source_index:
                    raise ValueError(f"duplicate metadata sidecar source_index: {key}")
                by_source_index[key] = dict(metadata)
    return by_sample_id, by_source_index, digest.hexdigest()


def _sidecar_metadata(
    *,
    sample_id: str,
    source_index: int,
    by_sample_id: dict[str, dict[str, Any]],
    by_source_index: dict[int, dict[str, Any]],
    required: bool,
) -> dict[str, Any]:
    metadata = by_sample_id.get(sample_id, by_source_index.get(source_index))
    if metadata is None:
        if required:
            raise ValueError(
                f"metadata sidecar has no row for sample_id={sample_id}, "
                f"source_index={source_index}"
            )
        return {}
    return dict(metadata)


def _iter_rows(
    config: DatasetConfig,
    cache_dir: str | Path | None,
    *,
    source_total_callback: Callable[[int], None] | None = None,
):
    if config.source == "jsonl":
        if config.selection != "all":
            raise ValueError("derived split selection is currently supported only for HF datasets")
        path = Path(config.path).expanduser().resolve()
        if source_total_callback is not None:
            with path.open(encoding="utf-8") as handle:
                source_total_callback(sum(bool(line.strip()) for line in handle))
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if line.strip():
                    try:
                        yield line_number - 1, json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
        return
    elif config.source == "hf":
        if config.data_file is not None:
            if config.selection not in {"all", "sample_n"}:
                raise ValueError("repository-file datasets support all or sample_n selection")
            if cache_dir is None:
                raise ValueError("repository-file datasets require a configured cache directory")
            hub_cache = Path(cache_dir).expanduser().resolve().parent / "hub"
            cached_file = try_to_load_from_cache(
                config.path,
                config.data_file,
                cache_dir=hub_cache,
                revision=config.revision,
                repo_type="dataset",
            )
            if not isinstance(cached_file, str):
                raise FileNotFoundError(
                    f"fixed dataset file is not cached: {config.path}@{config.revision}/"
                    f"{config.data_file}; enable preset cache preparation"
                )
            source = Path(cached_file)
            if source.suffix == ".json":
                values = json.loads(source.read_text(encoding="utf-8"))
                if not isinstance(values, list):
                    raise ValueError(f"repository JSON file must contain an array: {source}")
                rows = [dict(value) for value in values]
            else:
                rows = []
                with source.open(encoding="utf-8") as handle:
                    for line_number, line in enumerate(handle, start=1):
                        if not line.strip():
                            continue
                        try:
                            rows.append(dict(json.loads(line)))
                        except json.JSONDecodeError as exc:
                            raise ValueError(
                                f"invalid repository JSON at {source}:{line_number}: {exc}"
                            ) from exc
            indices = list(range(len(rows)))
            if config.selection == "sample_n":
                if config.max_samples is None:
                    raise ValueError("sample_n selection requires max_samples")
                indices = random.Random(config.selection_seed).sample(
                    indices, min(config.max_samples, len(indices))
                )
            if source_total_callback is not None:
                source_total_callback(len(indices))
            for index in indices:
                yield index, rows[index]
            return
        dataset = load_dataset(
            config.path,
            config.name,
            split=config.split,
            revision=config.revision,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
            download_config=DownloadConfig(local_files_only=True),
        )
        if config.selection == "infinity_code_sample_n":
            if config.max_samples is None:
                raise ValueError("infinity_code_sample_n requires max_samples")
            from .code import is_infinity_code_row

            dataset = dataset.filter(
                is_infinity_code_row,
                desc="Filtering Infinity-Instruct code rows",
            )
            sample_count = min(config.max_samples, len(dataset))
            indices = random.Random(config.selection_seed).sample(range(len(dataset)), sample_count)
        else:
            indices = _selected_hf_indices(dataset, config)
        if source_total_callback is not None:
            source_total_callback(len(indices))
        for index in indices:
            yield index, dict(dataset[index])
        return
    else:
        raise ValueError(f"unsupported dataset source: {config.source}")


def _base_tokenize(
    tokenizer: Any,
    prompt: str,
    response: str,
    max_length: int,
    *,
    allow_response_whitespace_bridge: bool = False,
) -> dict[str, list[int]]:
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("response boundary masking requires a fast tokenizer with offset mappings")
    combined = prompt + response
    reserve_eos = 1 if tokenizer.eos_token_id is not None else 0
    encoded = tokenizer(
        combined,
        add_special_tokens=True,
        truncation=True,
        max_length=max_length - reserve_eos,
        return_attention_mask=True,
        return_offsets_mapping=True,
    )
    input_ids = list(encoded["input_ids"])
    attention_mask = list(encoded["attention_mask"])
    offsets = encoded["offset_mapping"]
    response_start = len(prompt)
    labels: list[int] = []
    for token_id, (start, end) in zip(input_ids, offsets, strict=True):
        if start < response_start < end:
            prompt_fragment = combined[start:response_start]
            response_fragment = combined[response_start:end]
            if allow_response_whitespace_bridge and response_fragment.isspace():
                labels.append(-100)
            elif prompt_fragment.isspace():
                labels.append(token_id)
            else:
                raise ValueError(
                    "prompt/response boundary falls inside one token; use a token-aligned separator"
                )
        else:
            labels.append(token_id if start >= response_start and end > start else -100)
    if all(label == -100 for label in labels):
        raise NoSupervisedTokensError("truncation removed every supervised response token")
    if tokenizer.eos_token_id is not None and len(input_ids) < max_length:
        input_ids.append(tokenizer.eos_token_id)
        attention_mask.append(1)
        labels.append(tokenizer.eos_token_id)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def _base_tokenize_preserve_response(
    tokenizer: Any,
    prompt: str,
    response: str,
    max_length: int,
    *,
    maximum_prompt_tokens: int = 256,
    preserve_prompt_tail: bool = False,
) -> dict[str, list[int]]:
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("response boundary masking requires a fast tokenizer")
    prompt_encoded = tokenizer(
        prompt,
        add_special_tokens=True,
        truncation=False,
        return_attention_mask=True,
        verbose=False,
    )
    response_encoded = tokenizer(
        response,
        add_special_tokens=False,
        truncation=False,
        return_attention_mask=True,
        verbose=False,
    )
    prompt_input_ids = list(prompt_encoded["input_ids"])
    prompt_attention_mask = list(prompt_encoded["attention_mask"])
    response_input_ids = list(response_encoded["input_ids"])
    response_attention_mask = list(response_encoded["attention_mask"])
    eos_only_response = not response_input_ids and tokenizer.eos_token_id is not None
    if not response_input_ids and not eos_only_response:
        raise NoSupervisedTokensError("separate tokenization produced no response tokens")
    reserve_eos = 1 if tokenizer.eos_token_id is not None else 0
    budget = max_length - reserve_eos
    if budget <= 1:
        raise NoSupervisedTokensError("max_length leaves no room for response tokens")
    if len(prompt_input_ids) + len(response_input_ids) > budget:
        response_reserve = min(len(response_input_ids), max(1, budget // 2))
        prompt_budget = min(maximum_prompt_tokens, max(1, budget - response_reserve))
        if preserve_prompt_tail and len(prompt_input_ids) > prompt_budget:
            head_length = prompt_budget // 2
            tail_length = prompt_budget - head_length
            prompt_input_ids = prompt_input_ids[:head_length] + prompt_input_ids[-tail_length:]
            prompt_attention_mask = (
                prompt_attention_mask[:head_length] + prompt_attention_mask[-tail_length:]
            )
        else:
            prompt_input_ids = prompt_input_ids[:prompt_budget]
            prompt_attention_mask = prompt_attention_mask[:prompt_budget]
    response_budget = budget - len(prompt_input_ids)
    response_input_ids = response_input_ids[:response_budget]
    response_attention_mask = response_attention_mask[:response_budget]
    if not response_input_ids and not eos_only_response:
        raise NoSupervisedTokensError("truncation removed every response token")
    input_ids = prompt_input_ids + response_input_ids
    attention_mask = prompt_attention_mask + response_attention_mask
    labels = [-100] * len(prompt_input_ids) + response_input_ids
    if tokenizer.eos_token_id is not None and len(input_ids) < max_length:
        input_ids.append(tokenizer.eos_token_id)
        attention_mask.append(1)
        labels.append(tokenizer.eos_token_id)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def _instruct_tokenize(
    tokenizer: Any, prompt: str, response: str, max_length: int
) -> dict[str, list[int]]:
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
        truncation=True,
        max_length=max_length,
    )
    assistant_mask = encoded.get("assistant_masks", encoded.get("assistant_tokens_mask"))
    if assistant_mask is None:
        raise ValueError("the official chat template did not return an assistant token mask")
    input_ids = list(encoded["input_ids"])
    attention_mask = list(encoded.get("attention_mask", [1] * len(input_ids)))
    labels = [
        token_id if selected else -100
        for token_id, selected in zip(input_ids, assistant_mask, strict=True)
    ]
    if all(label == -100 for label in labels):
        raise NoSupervisedTokensError("chat templating produced no supervised assistant tokens")
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def tokenize_prompt_response(
    tokenizer: Any,
    prompt: str,
    response: str,
    *,
    max_length: int,
    model_variant: str,
    preserve_response: bool = False,
    preserve_prompt_tail: bool = False,
    allow_response_whitespace_bridge: bool = False,
) -> dict[str, list[int]]:
    if model_variant == "base":
        if preserve_response:
            return _base_tokenize_preserve_response(
                tokenizer,
                prompt,
                response,
                max_length,
                maximum_prompt_tokens=(max_length if preserve_prompt_tail else 256),
                preserve_prompt_tail=preserve_prompt_tail,
            )
        return _base_tokenize(
            tokenizer,
            prompt,
            response,
            max_length,
            allow_response_whitespace_bridge=allow_response_whitespace_bridge,
        )
    if model_variant == "instruct":
        return _instruct_tokenize(tokenizer, prompt, response, max_length)
    raise ValueError("model_variant must be base or instruct")


def load_tokenized_dataset(
    config: DatasetConfig,
    tokenizer: Any,
    *,
    max_length: int,
    model_variant: str,
    cache_dir: str | Path | None = None,
    progress_description: str | None = None,
) -> TokenizedDataset:
    code_adapters = {"codealpaca", "code_bagel_hermes", "infinity_instruct_code"}
    response_preserving_adapters = {
        *code_adapters,
        "evol_instruct",
        "logiqa2",
        "alpaca_eval",
    }
    # Qwen can merge HumanEval's closing docstring with a leading response newline.
    allow_response_whitespace_bridge = config.adapter in {"smart_flan", "humaneval"}
    template_version = (
        "pretokenized-v1"
        if config.adapter == "tokenized"
        else DEFAULT_DATASET_REGISTRY.adapter(config.adapter).template_version
    )
    preprocessing_contract = {
        "adapter": config.adapter,
        "template_version": template_version,
        "max_length": max_length,
        "model_variant": model_variant,
        "truncation_side": getattr(tokenizer, "truncation_side", "right"),
        "padding_side": getattr(tokenizer, "padding_side", "right"),
        "masking": (
            "assistant_mask"
            if model_variant == "instruct"
            else (
                "separate_prompt_response_response_only_v1"
                if config.adapter in response_preserving_adapters
                else (
                    "offset_response_only_bidirectional_whitespace_bridge_v3"
                    if allow_response_whitespace_bridge
                    else "offset_response_only_whitespace_bridge_v2"
                )
            )
        ),
        "eos_rule": "append_when_space_available",
        "filter_rules": ["drop_no_supervised_tokens"],
        "truncation_policy": (
            "code_separate_prompt_response_keep_start_256_v2"
            if config.adapter in code_adapters
            else (
                "instruction_separate_prompt_response_keep_start_256_v1"
                if config.adapter == "evol_instruct"
                else (
                    "multiple_choice_separate_response_keep_prompt_head_tail_v1"
                    if config.adapter == "logiqa2"
                    else (
                        "instruction_reference_keep_prompt_head_tail_v1"
                        if config.adapter == "alpaca_eval"
                        else "tokenizer_default"
                    )
                )
            )
        ),
        "training_objective": (
            "numeric_token_expectation_mse_v1" if config.adapter == "stsb" else "causal_ce"
        ),
        "selection": config.selection,
        "selection_seed": config.selection_seed,
        "validation_fraction": config.validation_fraction,
        "stratify_column": config.stratify_column,
        "stratify_regression": config.stratify_regression,
    }
    identity = {
        "source": config.source,
        "path": config.path,
        "revision": config.revision,
        "name": config.name,
        "data_file": config.data_file,
        "split": config.split,
        "preprocessing": preprocessing_contract,
    }
    namespace = json.dumps(identity, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    sample_namespace = _sample_namespace(config, template_version)
    sidecar_by_id, sidecar_by_index, sidecar_sha256 = _load_metadata_sidecar(config.metadata_path)
    records: list[dict[str, Any]] = []
    scanned_samples = 0
    filtered_no_supervision = 0
    source_digest = hashlib.sha256()
    progress = (
        tqdm(
            total=config.max_samples,
            desc=f"{progress_description}: tokenizing",
            unit="sample",
            dynamic_ncols=True,
        )
        if progress_description is not None
        else None
    )

    def set_source_total(source_total: int) -> None:
        if progress is None:
            return
        progress.total = (
            source_total if config.max_samples is None else min(config.max_samples, source_total)
        )
        progress.refresh()

    def update_progress(*, accepted: bool = False, final: bool = False) -> None:
        if progress is None:
            return
        if not final and (config.max_samples is None or accepted):
            progress.update(1)
        if final or scanned_samples % 256 == 0:
            progress.set_postfix(
                accepted=len(records),
                filtered=filtered_no_supervision,
                scanned=scanned_samples,
                refresh=final,
            )

    try:
        for source_index, row in _iter_rows(
            config,
            cache_dir,
            source_total_callback=set_source_total if progress is not None else None,
        ):
            scanned_samples += 1
            source_digest.update(
                json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
            )
            sample_id = str(
                row.get("sample_id") or make_sample_id(sample_namespace, source_index, row)
            )
            try:
                adapter_metadata: dict[str, Any] = {}
                if config.adapter == "tokenized":
                    tokenized = {
                        "input_ids": list(row["input_ids"]),
                        "attention_mask": list(
                            row.get("attention_mask", [1] * len(row["input_ids"]))
                        ),
                        "labels": list(row["labels"]),
                    }
                else:
                    rendered = DEFAULT_DATASET_REGISTRY.render(config.adapter, row)
                    prompt, response = rendered.prompt, rendered.response
                    adapter_metadata = rendered.metadata
                    tokenized = tokenize_prompt_response(
                        tokenizer,
                        prompt,
                        response,
                        max_length=max_length,
                        model_variant=model_variant,
                        preserve_response=config.adapter in response_preserving_adapters,
                        preserve_prompt_tail=config.adapter in {"logiqa2", "alpaca_eval"},
                        allow_response_whitespace_bridge=allow_response_whitespace_bridge,
                    )
                    if adapter_metadata.get("task_type") == "regression":
                        tokenized.update(
                            _stsb_regression_fields(
                                tokenizer,
                                tokenized["labels"],
                                label_text=str(adapter_metadata["label_text"]),
                            )
                        )
                sidecar_metadata = _sidecar_metadata(
                    sample_id=sample_id,
                    source_index=source_index,
                    by_sample_id=sidecar_by_id,
                    by_source_index=sidecar_by_index,
                    required=config.metadata_path is not None,
                )
                if sidecar_metadata:
                    adapter_metadata.update(sidecar_metadata)
                    adapter_metadata["task_metadata_available"] = True
            except NoSupervisedTokensError:
                filtered_no_supervision += 1
                update_progress()
                continue
            tokenized["sample_id"] = sample_id
            if config.adapter != "tokenized":
                normalized_text = " ".join(f"{prompt}\n{response}".casefold().split())
                adapter_metadata["normalized_content_sha256"] = hashlib.sha256(
                    normalized_text.encode()
                ).hexdigest()
            tokenized["source_metadata"] = {
                "dataset_name": config.path,
                "dataset_revision": config.revision,
                "split": config.split,
                "template_version": template_version,
                "source_index": source_index,
                **adapter_metadata,
                **{
                    key: row[key]
                    for key in ("task", "task_name", "task_source", "source", "template_type")
                    if key in row
                },
            }
            if "selected_anchor_positions" in row:
                tokenized["selected_anchor_positions"] = list(row["selected_anchor_positions"])
            records.append(tokenized)
            update_progress(accepted=True)
            if config.max_samples is not None and len(records) >= config.max_samples:
                break
    finally:
        if progress is not None:
            update_progress(final=True)
            progress.close()
    if not records and filtered_no_supervision:
        raise NoSupervisedTokensError(
            "truncation removed every supervised response token from all scanned samples"
        )
    task_counts = Counter(
        str(record["source_metadata"].get("task_name", "unknown")) for record in records
    )
    template_counts = Counter(
        str(record["source_metadata"].get("template_type", template_version)) for record in records
    )
    registered = DEFAULT_DATASET_REGISTRY.matching_dataset(
        dataset_id=config.path,
        revision=config.revision,
        config_name=config.name,
    )
    return TokenizedDataset(
        records,
        namespace=namespace,
        preprocessing_stats={
            "scanned_samples": scanned_samples,
            "accepted_samples": len(records),
            "filtered_no_supervision": filtered_no_supervision,
        },
        progress_description=progress_description,
        manifest_metadata={
            "source": {
                "kind": config.source,
                "dataset_id": config.path,
                "configuration": config.name,
                "data_file": config.data_file,
                "revision": config.revision,
                "split": config.split,
                "selection": config.selection,
                "selection_seed": config.selection_seed,
                "validation_fraction": config.validation_fraction,
                "scanned_content_sha256": source_digest.hexdigest(),
                "metadata_sidecar": config.metadata_path,
                "metadata_sidecar_sha256": sidecar_sha256,
            },
            "preprocessing": preprocessing_contract,
            "task_counts": dict(sorted(task_counts.items())),
            "template_counts": dict(sorted(template_counts.items())),
            "lineage": dict(registered.lineage) if registered is not None else {},
            "checks": {
                "sample_ids_unique": len({record["sample_id"] for record in records})
                == len(records),
                "accepted_plus_filtered_equals_scanned": (
                    len(records) + filtered_no_supervision == scanned_samples
                ),
            },
        },
    )


def load_logical_dataset(
    config: DatasetConfig,
    *,
    cache_dir: str | Path | None = None,
) -> LogicalDataset:
    if config.adapter in {"tokenized", "lm_eval_only"}:
        raise ValueError(f"adapter '{config.adapter}' does not expose logical prompt/response rows")
    adapter = DEFAULT_DATASET_REGISTRY.adapter(config.adapter)
    namespace = _sample_namespace(config, adapter.template_version)
    sidecar_by_id, sidecar_by_index, sidecar_sha256 = _load_metadata_sidecar(config.metadata_path)
    examples: list[LogicalExample] = []
    source_digest = hashlib.sha256()
    for source_index, row in _iter_rows(config, cache_dir):
        source_digest.update(
            json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
        )
        rendered = adapter.render(row)
        sample_id = str(row.get("sample_id") or make_sample_id(namespace, source_index, row))
        sidecar_metadata = _sidecar_metadata(
            sample_id=sample_id,
            source_index=source_index,
            by_sample_id=sidecar_by_id,
            by_source_index=sidecar_by_index,
            required=config.metadata_path is not None,
        )
        metadata = {
            "source_index": source_index,
            "template_version": adapter.template_version,
            **rendered.metadata,
            **{
                key: row[key]
                for key in ("task", "task_name", "task_source", "source", "template_type")
                if key in row
            },
            **sidecar_metadata,
        }
        if sidecar_metadata:
            metadata["task_metadata_available"] = True
        examples.append(
            LogicalExample(
                sample_id=sample_id,
                dataset_name=config.path,
                dataset_revision=config.revision,
                task_name=str(metadata.get("task_name", config.adapter)),
                split=config.split,
                prompt=rendered.prompt,
                response=rendered.response,
                loss_mask="response_only",
                source_metadata=metadata,
            )
        )
        if config.max_samples is not None and len(examples) >= config.max_samples:
            break
    return LogicalDataset(
        examples,
        namespace=namespace,
        manifest_metadata={
            "source": {
                "kind": config.source,
                "dataset_id": config.path,
                "configuration": config.name,
                "data_file": config.data_file,
                "revision": config.revision,
                "split": config.split,
                "selection": config.selection,
                "selection_seed": config.selection_seed,
                "validation_fraction": config.validation_fraction,
                "scanned_content_sha256": source_digest.hexdigest(),
                "metadata_sidecar": config.metadata_path,
                "metadata_sidecar_sha256": sidecar_sha256,
            },
            "template_version": adapter.template_version,
            "loss_mask": "response_only",
        },
    )
