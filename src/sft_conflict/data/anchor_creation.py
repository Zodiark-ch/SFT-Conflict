"""Task round-robin vocabulary cover over the FLAN 2022 submixtures used by SMART.

The published ``smart_flan`` mixture is a 400k-row *subset* of five much larger
submixtures (flan2021, t0, niv2, cot, dialog; 17.6M rows, 1,841 tasks).  This
module selects an Anchor dataset directly from those sources with two goals:
cover as many tasks as possible and cover as many label token types as possible.

Selection protocol (``task_vocab_cover_v1``):

1. Collect the task list of every submixture and shuffle it (seeded).
2. Visit tasks in that order.  For the current task draw one sample at random
   without replacement.  If its shifted supervised label tokens contain a token
   that the Anchor set has not covered yet, keep the sample and move to the next
   task.  Otherwise draw another sample from the same task until either one is
   kept or the task has no sample left.  Rejected samples are *not* returned.
3. After every task has been visited once, reshuffle the tasks that still have
   samples and start the next round.  The algorithm stops when every sample of
   every task has been drawn once.
4. Task-coverage guarantee (on by default): the first time a task is visited,
   the first drawn sample that has any supervised label token is kept
   unconditionally, even when it introduces no new token.  Later turns of the
   same task follow rule 2.  With the guarantee off, rule 2 applies from the
   first turn on.

Only unique label token sets are needed to make the decisions, so the sources
are tokenized once in parallel into compact CSR tables; the full token records
are materialized for the kept samples only.  No model inference is involved in
the selection; the base-model accuracy report is a separate, optional step.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from .registry import DEFAULT_DATASET_REGISTRY

ANCHOR_CREATION_POLICY = "task_vocab_cover_v1"
ANCHOR_CREATION_ROOT = "created_anchor_datasets"
ANCHOR_CREATION_LABEL_CACHE_ROOT = "created_anchor_label_cache"
ANCHOR_CREATION_SCHEMA_VERSION = 2
# The label-set cache format is independent of the publication schema; bumping
# ANCHOR_CREATION_SCHEMA_VERSION must not invalidate hours of tokenization.
LABEL_CACHE_SCHEMA_VERSION = 2
SMART_ADAPTER = "smart_flan"
SMART_TEMPLATE_VERSION = "smart-flan-v3"
LABEL_SCOPE = "unique_shifted_supervised_labels_including_eos"

_SIGNATURE_SCOPE = "canonical_manifest_without_signature_including_artifact_sha256"
_TOKENIZATION_SIGNATURE_SCOPE = "model_identity_and_smart_flan_v3_ce_contract"
_SELECTION_ALGORITHM = "task_round_robin_without_replacement_first_new_label_token"


@dataclass(frozen=True)
class SubmixtureSpec:
    key: str
    dataset_id: str
    revision: str
    split: str = "train"


# Fixed revisions of the five FLAN 2022 submixtures published by the SMART authors
# (kowndinya23/<name>-submix-4096).  The rows carry inputs/targets plus
# task_source, task_name (ClassLabel) and template_type.
SMART_SUBMIXTURES: tuple[SubmixtureSpec, ...] = (
    SubmixtureSpec(
        "flan2021", "kowndinya23/flan2021-submix-4096", "4446b9b053b0bde41504239b21246efa803954a8"
    ),
    SubmixtureSpec("t0", "kowndinya23/t0-submix-4096", "7b2fd3b8dbdf13f824e63b023fd9bca709f7c71b"),
    SubmixtureSpec(
        "niv2", "kowndinya23/niv2-submix-4096", "9578148c9b4160aa6f5fa69d804b81165fd950b4"
    ),
    SubmixtureSpec(
        "cot", "kowndinya23/cot-submix-4096", "b706ef7fed9e543fcefecfa9012fc6d25a817104"
    ),
    SubmixtureSpec(
        "dialog", "kowndinya23/dialog-submix-4096", "321cd36a7e290eb6c8abd859bbf77b4989b84c34"
    ),
)
SUBMIXTURE_KEYS = tuple(spec.key for spec in SMART_SUBMIXTURES)


def submixture_spec(key: str) -> SubmixtureSpec:
    for spec in SMART_SUBMIXTURES:
        if spec.key == key:
            return spec
    raise ValueError(f"unknown SMART submixture '{key}'; available: {', '.join(SUBMIXTURE_KEYS)}")


def created_anchor_dataset_name(model_name: str) -> str:
    if not isinstance(model_name, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]*", model_name
    ):
        raise ValueError("model_name must be a non-empty, path-safe preset name")
    return f"flan2022_{model_name}_{ANCHOR_CREATION_POLICY}"


def created_anchor_dataset_dir(artifact_root: str | Path, model_name: str, max_length: int) -> Path:
    """One publication slot per model preset and tokenization length."""
    if type(max_length) is not int or max_length <= 1:
        raise ValueError("max_length must be an integer greater than one")
    return (
        Path(artifact_root).expanduser().resolve()
        / ANCHOR_CREATION_ROOT
        / created_anchor_dataset_name(model_name)
        / f"max-length-{max_length}"
    )


def anchor_label_cache_dir(artifact_root: str | Path, model_name: str, max_length: int) -> Path:
    return (
        Path(artifact_root).expanduser().resolve()
        / ANCHOR_CREATION_LABEL_CACHE_ROOT
        / created_anchor_dataset_name(model_name)
        / f"max-length-{int(max_length)}"
    )


# --------------------------------------------------------------------------
# Tokenization of SMART rows (batched twin of loading.tokenize_prompt_response)
# --------------------------------------------------------------------------


def render_smart_row(row: Mapping[str, Any]) -> tuple[str, str]:
    rendered = DEFAULT_DATASET_REGISTRY.render(SMART_ADAPTER, row)
    return rendered.prompt, rendered.response


def batch_tokenize_smart_rows(
    tokenizer: Any,
    prompts: Sequence[str],
    responses: Sequence[str],
    *,
    max_length: int,
) -> list[dict[str, list[int]] | None]:
    """Tokenize many prompt/response pairs exactly like the SMART-v3 base contract.

    Returns ``None`` for rows whose truncated sequence keeps no supervised
    response token, and raises ``ValueError`` when a token straddles the
    prompt/response boundary with non-whitespace content on both sides.
    """
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("response boundary masking requires a fast tokenizer with offset mappings")
    if len(prompts) != len(responses):
        raise ValueError("prompts and responses must have equal lengths")
    if not prompts:
        return []
    reserve_eos = 1 if tokenizer.eos_token_id is not None else 0
    combined = [prompt + response for prompt, response in zip(prompts, responses, strict=True)]
    encoded = tokenizer(
        combined,
        add_special_tokens=True,
        truncation=True,
        max_length=max_length - reserve_eos,
        return_attention_mask=True,
        return_offsets_mapping=True,
    )
    results: list[dict[str, list[int]] | None] = []
    for index, text in enumerate(combined):
        input_ids = list(encoded["input_ids"][index])
        attention_mask = list(encoded["attention_mask"][index])
        offsets = encoded["offset_mapping"][index]
        response_start = len(prompts[index])
        labels: list[int] = []
        supervised = False
        for token_id, (start, end) in zip(input_ids, offsets, strict=True):
            if start < response_start < end:
                prompt_fragment = text[start:response_start]
                response_fragment = text[response_start:end]
                if response_fragment.isspace():
                    labels.append(-100)
                elif prompt_fragment.isspace():
                    labels.append(token_id)
                    supervised = True
                else:
                    raise ValueError(
                        "prompt/response boundary falls inside one token; "
                        "use a token-aligned separator"
                    )
            elif start >= response_start and end > start:
                labels.append(token_id)
                supervised = True
            else:
                labels.append(-100)
        if not supervised:
            results.append(None)
            continue
        if tokenizer.eos_token_id is not None and len(input_ids) < max_length:
            input_ids.append(tokenizer.eos_token_id)
            attention_mask.append(1)
            labels.append(tokenizer.eos_token_id)
        results.append({"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels})
    return results


def unique_label_token_ids(labels: Sequence[int]) -> np.ndarray:
    """Sorted unique shifted supervised labels (position zero and -100 excluded)."""
    values = np.asarray(labels[1:], dtype=np.int64)
    values = values[values != -100]
    return np.unique(values).astype(np.int32, copy=False)


# --------------------------------------------------------------------------
# CSR label-set tables (one per submixture) with an on-disk cache
# --------------------------------------------------------------------------


@dataclass
class LabelSetTable:
    """Unique shifted supervised label IDs per source row, in CSR layout."""

    offsets: np.ndarray
    token_ids: np.ndarray
    boundary_excluded: int = 0

    def __post_init__(self) -> None:
        self.offsets = np.ascontiguousarray(self.offsets, dtype=np.int64)
        self.token_ids = np.ascontiguousarray(self.token_ids, dtype=np.int32)
        if self.offsets.ndim != 1 or self.token_ids.ndim != 1 or self.offsets.size == 0:
            raise ValueError("label-set table arrays must be one-dimensional and non-empty")
        if self.offsets[0] != 0 or int(self.offsets[-1]) != self.token_ids.size:
            raise ValueError("label-set table offsets do not match the token array")
        if np.any(np.diff(self.offsets) < 0):
            raise ValueError("label-set table offsets must be non-decreasing")
        if self.token_ids.size and int(self.token_ids.min()) < 0:
            raise ValueError("label-set table token IDs must be non-negative")

    def __len__(self) -> int:
        return int(self.offsets.size - 1)

    def label_ids(self, index: int) -> np.ndarray:
        return self.token_ids[self.offsets[index] : self.offsets[index + 1]]

    def lengths(self) -> np.ndarray:
        return np.diff(self.offsets)

    @property
    def max_token_id(self) -> int:
        return int(self.token_ids.max()) if self.token_ids.size else -1

    def content_sha256(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.offsets.tobytes())
        digest.update(self.token_ids.tobytes())
        return digest.hexdigest()


def _label_sets_from_tokenized(
    tokenized: Iterable[dict[str, list[int]] | None],
) -> tuple[np.ndarray, np.ndarray]:
    lengths: list[int] = []
    pieces: list[np.ndarray] = []
    for record in tokenized:
        if record is None:
            lengths.append(0)
            continue
        ids = unique_label_token_ids(record["labels"])
        lengths.append(int(ids.size))
        if ids.size:
            pieces.append(ids)
    offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
    np.cumsum(np.asarray(lengths, dtype=np.int64), out=offsets[1:])
    token_ids = (
        np.concatenate(pieces).astype(np.int32, copy=False)
        if pieces
        else np.zeros(0, dtype=np.int32)
    )
    return offsets, token_ids


_WORKER_STATE: dict[str, Any] = {}


def _init_label_worker(rows: Any, tokenizer: Any, max_length: int) -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    _WORKER_STATE["rows"] = rows
    _WORKER_STATE["tokenizer"] = tokenizer
    _WORKER_STATE["max_length"] = int(max_length)


def _label_sets_for_range(
    rows: Any,
    tokenizer: Any,
    max_length: int,
    start: int,
    end: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    chunk = rows[start:end]
    inputs = chunk["inputs"] if isinstance(chunk, Mapping) else [row["inputs"] for row in chunk]
    targets = chunk["targets"] if isinstance(chunk, Mapping) else [row["targets"] for row in chunk]
    prompts: list[str] = []
    responses: list[str] = []
    for prompt, response in zip(inputs, targets, strict=True):
        rendered_prompt, rendered_response = render_smart_row(
            {"inputs": prompt, "targets": response}
        )
        prompts.append(rendered_prompt)
        responses.append(rendered_response)
    excluded = 0
    try:
        tokenized = batch_tokenize_smart_rows(tokenizer, prompts, responses, max_length=max_length)
    except ValueError:
        # Isolate the offending rows instead of losing a multi-hour job: a row that
        # fails the SMART-v3 boundary contract is treated as having no label set.
        tokenized = []
        for prompt, response in zip(prompts, responses, strict=True):
            try:
                tokenized.extend(
                    batch_tokenize_smart_rows(
                        tokenizer, [prompt], [response], max_length=max_length
                    )
                )
            except ValueError:
                tokenized.append(None)
                excluded += 1
    offsets, token_ids = _label_sets_from_tokenized(tokenized)
    return offsets, token_ids, excluded


def _label_worker_job(bounds: tuple[int, int]) -> tuple[int, np.ndarray, np.ndarray, int]:
    start, end = bounds
    offsets, token_ids, excluded = _label_sets_for_range(
        _WORKER_STATE["rows"],
        _WORKER_STATE["tokenizer"],
        _WORKER_STATE["max_length"],
        start,
        end,
    )
    return start, offsets, token_ids, excluded


def build_label_set_table(
    rows: Any,
    tokenizer: Any,
    *,
    max_length: int,
    workers: int = 0,
    chunk_rows: int = 1024,
    progress_description: str | None = None,
) -> LabelSetTable:
    """Tokenize every row of ``rows`` (``inputs``/``targets`` columns) into a CSR table.

    ``rows`` must support slicing (``rows[a:b]``) returning either a mapping of
    column lists (Hugging Face ``Dataset``) or a sequence of row mappings.
    ``workers`` > 0 forks that many tokenization processes.
    """
    total = len(rows)
    if total == 0:
        raise ValueError("cannot build a label-set table from an empty source")
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    bounds = [(start, min(start + chunk_rows, total)) for start in range(0, total, chunk_rows)]
    progress = None
    if progress_description is not None:
        from tqdm.auto import tqdm

        progress = tqdm(
            total=total, desc=f"{progress_description}: label sets", unit="row", dynamic_ncols=True
        )
    parts: list[tuple[np.ndarray, np.ndarray]] = []
    excluded_total = 0

    def consume(result: tuple[int, np.ndarray, np.ndarray, int], expected_start: int) -> None:
        nonlocal excluded_total
        start, offsets, token_ids, excluded = result
        if start != expected_start:
            raise RuntimeError("label-set chunks were returned out of order")
        parts.append((offsets, token_ids))
        excluded_total += excluded
        if progress is not None:
            progress.update(int(offsets.size - 1))

    try:
        if workers <= 0:
            for start, end in bounds:
                chunk = _label_sets_for_range(rows, tokenizer, max_length, start, end)
                consume((start, *chunk), start)
        else:
            import multiprocessing

            context = multiprocessing.get_context("fork")
            with context.Pool(
                processes=min(workers, len(bounds)),
                initializer=_init_label_worker,
                initargs=(rows, tokenizer, max_length),
            ) as pool:
                for (start, _), result in zip(
                    bounds, pool.imap(_label_worker_job, bounds, chunksize=1), strict=True
                ):
                    consume(result, start)
    finally:
        if progress is not None:
            progress.close()
    lengths = np.concatenate([np.diff(offsets) for offsets, _ in parts])
    offsets = np.zeros(lengths.size + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    token_ids = np.concatenate([ids for _, ids in parts]) if parts else np.zeros(0, dtype=np.int32)
    if offsets.size - 1 != total:
        raise RuntimeError("label-set table row count does not match the source")
    return LabelSetTable(offsets=offsets, token_ids=token_ids, boundary_excluded=excluded_total)


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def tokenization_contract(model: Mapping[str, str], max_length: int) -> dict[str, Any]:
    contract = {
        "adapter": SMART_ADAPTER,
        "template_version": SMART_TEMPLATE_VERSION,
        "model_variant": "base",
        "max_length": int(max_length),
        "padding_side": "right",
        "truncation_side": "right",
        "masking": "offset_response_only_bidirectional_whitespace_bridge_v3",
        "eos_rule": "append_when_space_available",
        "filter_rules": ["drop_no_supervised_tokens"],
        "truncation_policy": "tokenizer_default",
        "training_objective": "causal_ce",
        "selection": "all",
    }
    return {
        **contract,
        "signature_scope": _TOKENIZATION_SIGNATURE_SCOPE,
        "signature_sha256": _sha256({"model": dict(model), "preprocessing": contract}),
    }


def label_cache_identity(
    spec: SubmixtureSpec,
    *,
    rows: int,
    model: Mapping[str, str],
    max_length: int,
) -> dict[str, Any]:
    return {
        "schema_version": LABEL_CACHE_SCHEMA_VERSION,
        "policy": ANCHOR_CREATION_POLICY,
        "submixture": spec.key,
        "dataset_id": spec.dataset_id,
        "revision": spec.revision,
        "split": spec.split,
        "rows": int(rows),
        "model": dict(model),
        "tokenization": tokenization_contract(model, max_length),
        "label_scope": LABEL_SCOPE,
    }


def save_label_set_table(
    directory: str | Path, spec: SubmixtureSpec, table: LabelSetTable, identity: Mapping[str, Any]
) -> Path:
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    if identity["rows"] != len(table):
        raise ValueError("label-set cache identity rows do not match the table")
    array_path = root / f"{spec.key}.npz"
    manifest_path = root / f"{spec.key}.json"
    temporary = array_path.with_name(f".{array_path.name}.tmp-{uuid4().hex}")
    with temporary.open("wb") as handle:
        np.savez(handle, offsets=table.offsets, token_ids=table.token_ids)
    os.replace(temporary, array_path)
    manifest = {
        **dict(identity),
        "boundary_excluded": int(table.boundary_excluded),
        "content_sha256": table.content_sha256(),
    }
    manifest_path.write_bytes(_json_bytes(manifest) + b"\n")
    return array_path


def load_label_set_table(
    directory: str | Path, spec: SubmixtureSpec, identity: Mapping[str, Any]
) -> LabelSetTable | None:
    root = Path(directory)
    array_path = root / f"{spec.key}.npz"
    manifest_path = root / f"{spec.key}.json"
    if not array_path.is_file() or not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_bytes())
    except (OSError, ValueError):
        return None
    expected = dict(identity)
    if any(manifest.get(key) != value for key, value in expected.items()):
        return None
    try:
        with np.load(array_path) as arrays:
            table = LabelSetTable(
                offsets=arrays["offsets"],
                token_ids=arrays["token_ids"],
                boundary_excluded=int(manifest.get("boundary_excluded", 0)),
            )
    except (OSError, ValueError, KeyError):
        return None
    if len(table) != identity["rows"] or table.content_sha256() != manifest.get("content_sha256"):
        return None
    return table


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TaskPool:
    submixture: str
    task_name: str
    sample_indices: np.ndarray  # row indices into the submixture's label-set table

    def __post_init__(self) -> None:
        indices = np.ascontiguousarray(self.sample_indices, dtype=np.int64)
        if indices.ndim != 1 or indices.size == 0:
            raise ValueError(f"task '{self.task_name}' must own at least one sample")
        if np.unique(indices).size != indices.size:
            raise ValueError(f"task '{self.task_name}' has duplicate sample indices")
        object.__setattr__(self, "sample_indices", indices)


@dataclass(frozen=True)
class SelectedSample:
    submixture: str
    task_name: str
    source_index: int
    round_index: int
    selection_order: int
    new_label_token_count: int
    guaranteed: bool = False


@dataclass
class TaskVocabCoverResult:
    selected: list[SelectedSample]
    covered_token_ids: np.ndarray
    rounds: int
    traversed_samples: int
    unsupervised_samples: int
    task_source_counts: dict[str, int]
    task_selected_counts: dict[str, int]
    round_acceptances: list[int] = field(default_factory=list)
    task_permutations: dict[str, np.ndarray] | None = None
    guarantee_task_coverage: bool = False
    guaranteed_samples: int = 0
    guaranteed_without_new_tokens: int = 0

    @property
    def selected_tasks(self) -> int:
        return sum(1 for count in self.task_selected_counts.values() if count > 0)


def _ragged_gather(starts: np.ndarray, lengths: np.ndarray, total: int) -> np.ndarray:
    prefix = np.cumsum(lengths) - lengths
    return np.repeat(starts - prefix, lengths) + np.arange(total, dtype=np.int64)


def select_task_vocab_cover(
    tasks: Sequence[TaskPool],
    tables: Mapping[str, LabelSetTable],
    *,
    seed: int,
    vocab_size: int | None = None,
    log: Callable[[str], None] | None = None,
    record_permutations: bool = False,
    guarantee_task_coverage: bool = False,
) -> TaskVocabCoverResult:
    """Run the task round-robin first-new-label-token cover described in the module docstring.

    With ``guarantee_task_coverage`` a task's first turn keeps its first drawn
    supervised sample unconditionally; every later turn requires a new token.
    """
    if not tasks:
        raise ValueError("task vocabulary cover requires at least one task")
    names = [task.task_name for task in tasks]
    if len(set(names)) != len(names):
        raise ValueError("task names must be unique across submixtures")
    for task in tasks:
        table = tables[task.submixture]
        if int(task.sample_indices.max()) >= len(table):
            raise ValueError(f"task '{task.task_name}' references rows outside its submixture")
    largest_id = max((table.max_token_id for table in tables.values()), default=-1)
    width = largest_id + 1 if vocab_size is None else max(int(vocab_size), largest_id + 1)
    rng = np.random.default_rng(int(seed))
    permutations = [rng.permutation(task.sample_indices) for task in tasks]
    cursors = np.zeros(len(tasks), dtype=np.int64)
    covered = np.zeros(max(width, 1), dtype=bool)
    selected: list[SelectedSample] = []
    task_selected = [0] * len(tasks)
    visited = [False] * len(tasks)
    guaranteed_samples = 0
    guaranteed_without_new = 0
    traversed = 0
    unsupervised = 0
    round_acceptances: list[int] = []
    active = np.arange(len(tasks), dtype=np.int64)
    round_index = 0
    started = time.monotonic()
    while active.size:
        round_index += 1
        order = rng.permutation(active)
        still_active: list[int] = []
        accepted = 0
        for task_position in order.tolist():
            task = tasks[task_position]
            table = tables[task.submixture]
            permutation = permutations[task_position]
            start = int(cursors[task_position])
            count = int(permutation.size)
            first_turn = guarantee_task_coverage and not visited[task_position]
            visited[task_position] = True
            block = 64
            while start < count:
                end = min(start + block, count)
                rows = permutation[start:end]
                row_starts = table.offsets[rows]
                lengths = table.offsets[rows + 1] - row_starts
                total = int(lengths.sum())
                hit = -1
                if first_turn:
                    # Guarantee: keep the first drawn sample that has any supervised label.
                    nonzero = np.flatnonzero(lengths)
                    hit = int(nonzero[0]) if nonzero.size else -1
                elif total:
                    ids = table.token_ids[_ragged_gather(row_starts, lengths, total)]
                    uncovered = ~covered[ids]
                    if uncovered.any():
                        nonzero = np.flatnonzero(lengths)
                        reduce_starts = (np.cumsum(lengths) - lengths)[nonzero]
                        sample_hit = np.logical_or.reduceat(uncovered, reduce_starts)
                        hit = int(nonzero[int(np.argmax(sample_hit))])
                if hit >= 0:
                    unsupervised += int(np.count_nonzero(lengths[:hit] == 0))
                    traversed += hit + 1
                    label_ids = table.label_ids(int(rows[hit]))
                    new_count = int(np.count_nonzero(~covered[label_ids]))
                    covered[label_ids] = True
                    selected.append(
                        SelectedSample(
                            submixture=task.submixture,
                            task_name=task.task_name,
                            source_index=int(rows[hit]),
                            round_index=round_index,
                            selection_order=len(selected),
                            new_label_token_count=new_count,
                            guaranteed=first_turn,
                        )
                    )
                    if first_turn:
                        guaranteed_samples += 1
                        if new_count == 0:
                            guaranteed_without_new += 1
                    task_selected[task_position] += 1
                    accepted += 1
                    start = start + hit + 1
                    break
                unsupervised += int(np.count_nonzero(lengths == 0))
                traversed += end - start
                start = end
                block = min(block * 4, 1 << 16)
            cursors[task_position] = start
            if start < count:
                still_active.append(task_position)
        active = np.sort(np.asarray(still_active, dtype=np.int64))
        round_acceptances.append(accepted)
        if log is not None and (round_index <= 10 or round_index % 50 == 0 or not active.size):
            log(
                f"round {round_index}: accepted={accepted} selected={len(selected)} "
                f"covered_tokens={int(covered.sum())} active_tasks={active.size} "
                f"traversed={traversed} elapsed={time.monotonic() - started:.1f}s"
            )
    total_samples = sum(int(task.sample_indices.size) for task in tasks)
    if traversed != total_samples:
        raise RuntimeError(
            f"task vocabulary cover traversed {traversed} samples, expected {total_samples}"
        )
    return TaskVocabCoverResult(
        selected=selected,
        covered_token_ids=np.flatnonzero(covered).astype(np.int64),
        rounds=round_index,
        traversed_samples=traversed,
        unsupervised_samples=unsupervised,
        task_source_counts={task.task_name: int(task.sample_indices.size) for task in tasks},
        task_selected_counts={
            task.task_name: task_selected[index] for index, task in enumerate(tasks)
        },
        round_acceptances=round_acceptances,
        task_permutations=(
            {task.task_name: permutations[index] for index, task in enumerate(tasks)}
            if record_permutations
            else None
        ),
        guarantee_task_coverage=bool(guarantee_task_coverage),
        guaranteed_samples=guaranteed_samples,
        guaranteed_without_new_tokens=guaranteed_without_new,
    )


def source_token_sample_counts(
    tables: Mapping[str, LabelSetTable],
    width: int,
    rows_by_submixture: Mapping[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Per-token count of source samples whose label set contains the token.

    ``rows_by_submixture`` restricts each table to the rows that were actually
    pooled for selection (for example under a per-task sample cap); by default
    every row of every table counts.
    """
    counts = np.zeros(max(int(width), 1), dtype=np.int64)
    for key, table in tables.items():
        if rows_by_submixture is None:
            token_ids = table.token_ids
        else:
            rows = np.asarray(rows_by_submixture.get(key, ()), dtype=np.int64)
            if rows.size == 0:
                continue
            row_starts = table.offsets[rows]
            lengths = table.offsets[rows + 1] - row_starts
            total = int(lengths.sum())
            if total == 0:
                continue
            token_ids = table.token_ids[_ragged_gather(row_starts, lengths, total)]
        if token_ids.size:
            if int(token_ids.max()) >= counts.size:
                raise ValueError("source token ID exceeds the requested vocabulary width")
            counts += np.bincount(token_ids, minlength=counts.size)
    return counts


# --------------------------------------------------------------------------
# Publication
# --------------------------------------------------------------------------


_MANIFEST_KEYS = {
    "schema_version",
    "policy",
    "dataset_name",
    "format",
    "model",
    "tokenization",
    "sources",
    "selection",
    "counts",
    "tasks",
    "covered_token_ids",
    "accuracy",
    "files",
    "signature",
}
_COUNT_KEYS = {
    "source_samples",
    "traversed_samples",
    "unsupervised_samples",
    "boundary_excluded_samples",
    "source_tasks",
    "selected_samples",
    "selected_tasks",
    "covered_token_types",
    "source_token_types",
    "tokenizer_vocab_size",
    "source_token_sample_incidences",
    "selected_token_sample_incidences",
    "guaranteed_samples",
    "guaranteed_without_new_tokens",
}


def _object(value: Any, name: str, keys: set[str] | None = None) -> dict[str, Any]:
    if not isinstance(value, dict) or (keys is not None and set(value) != keys):
        raise ValueError(f"invalid created-anchor {name}")
    return value


def _count(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"created-anchor {name} must be a non-negative integer")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"created-anchor {name} must be a non-empty string")
    return value


def _digest_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"invalid created-anchor {name} SHA-256")
    return value


def model_identity(model_name: str, model_id: str, model_revision: str) -> dict[str, str]:
    created_anchor_dataset_name(model_name)
    return {
        "preset": model_name,
        "model_id": _text(model_id, "model_id"),
        "revision": _text(model_revision, "model_revision"),
    }


def _validate_record(record: dict[str, Any], max_length: int) -> None:
    _text(record.get("sample_id"), "sample_id")
    fields = [record.get(key) for key in ("input_ids", "attention_mask", "labels")]
    if any(not isinstance(value, list) for value in fields):
        raise ValueError("created-anchor records require list token fields")
    if len({len(value) for value in fields}) != 1 or not 1 <= len(fields[0]) <= max_length:
        raise ValueError("created-anchor record lengths are invalid")
    for token_id in record["input_ids"]:
        _count(token_id, "input token ID")
    for token_id in record["labels"]:
        if type(token_id) is not int or (token_id < 0 and token_id != -100):
            raise ValueError("invalid created-anchor label token ID")
    if any(type(value) is not int or value not in {0, 1} for value in record["attention_mask"]):
        raise ValueError("invalid created-anchor attention mask")
    metadata = _object(record.get("source_metadata"), "source_metadata")
    for key in ("submixture", "task_name", "dataset_name", "dataset_revision"):
        _text(metadata.get(key), f"source_metadata.{key}")
    _count(metadata.get("source_index"), "source_metadata.source_index")
    _count(metadata.get("selection_round"), "source_metadata.selection_round")
    _count(metadata.get("new_label_token_count"), "source_metadata.new_label_token_count")
    if type(metadata.get("guaranteed")) is not bool:
        raise ValueError("created-anchor source_metadata.guaranteed must be a boolean")
    if record.get("task_name") != metadata["task_name"]:
        raise ValueError("created-anchor record task_name must match its source metadata")


def _safe_file(directory: Path, relative: Any) -> Path:
    if (
        not isinstance(relative, str)
        or not relative
        or any(character in relative for character in ("\\", ":", "\x00"))
        or any(part in {"", ".", ".."} for part in relative.split("/"))
    ):
        raise ValueError("created-anchor artifact paths must be safe relative paths")
    return directory / relative


def load_created_anchor_manifest(
    directory: str | Path,
    *,
    model_name: str,
    model_id: str,
    model_revision: str,
    max_length: int,
    verify_records: bool = True,
) -> dict[str, Any]:
    """Verify schema, model/tokenization compatibility, hashes and the selection replay."""
    model = model_identity(model_name, model_id, model_revision)
    root = Path(directory).expanduser().absolute()
    manifest_path = _safe_file(root, "manifest.json")
    manifest = _object(json.loads(manifest_path.read_bytes()), "manifest schema", _MANIFEST_KEYS)
    if (
        manifest["schema_version"] != ANCHOR_CREATION_SCHEMA_VERSION
        or manifest["policy"] != ANCHOR_CREATION_POLICY
        or manifest["format"] != "tokenized_jsonl"
        or manifest["dataset_name"] != created_anchor_dataset_name(model_name)
        or manifest["model"] != model
        or _json_bytes(manifest["tokenization"])
        != _json_bytes(tokenization_contract(model, max_length))
    ):
        raise ValueError("incompatible created-anchor policy, model or tokenization schema")
    signature = _object(manifest["signature"], "signature", {"algorithm", "scope", "sha256"})
    if (
        signature["algorithm"] != "sha256"
        or signature["scope"] != _SIGNATURE_SCOPE
        or signature["sha256"] != _sha256({k: v for k, v in manifest.items() if k != "signature"})
    ):
        raise ValueError("created-anchor manifest signature mismatch")
    counts = _object(manifest["counts"], "counts", _COUNT_KEYS)
    for name, value in counts.items():
        _count(value, name)
    selection = _object(manifest["selection"], "selection")
    guarantee = selection.get("guarantee_task_coverage")
    if type(guarantee) is not bool:
        raise ValueError("created-anchor selection must record guarantee_task_coverage")
    if not (
        counts["source_samples"] == counts["traversed_samples"]
        and counts["selected_samples"] <= counts["source_samples"]
        and counts["selected_tasks"] <= counts["source_tasks"]
        and counts["covered_token_types"] == counts["source_token_types"]
        and counts["guaranteed_without_new_tokens"]
        <= counts["guaranteed_samples"]
        <= counts["selected_samples"]
        and (guarantee or counts["guaranteed_samples"] == 0)
    ):
        raise ValueError("inconsistent created-anchor counts")
    universe = manifest["covered_token_ids"]
    if (
        not isinstance(universe, list)
        or len(universe) != counts["covered_token_types"]
        or any(left >= right for left, right in zip(universe, universe[1:], strict=False))
    ):
        raise ValueError("created-anchor covered token IDs must be strictly increasing")
    files = _object(manifest["files"], "files")
    dataset_path = _safe_file(root, files["dataset"])
    expected_digest = _digest_text(files["dataset_sha256"], "dataset")
    if not verify_records:
        if not dataset_path.is_file():
            raise ValueError("created-anchor dataset file is missing")
        return manifest
    digest = hashlib.sha256()
    selected_counts: Counter[int] = Counter()
    seen_tasks: set[str] = set()
    guaranteed_rows = 0
    guaranteed_without_new = 0
    rows = 0
    with dataset_path.open("rb") as handle:
        for line in handle:
            digest.update(line)
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError("created-anchor dataset rows must be JSON objects")
            _validate_record(record, max_length)
            metadata = record["source_metadata"]
            token_ids = set(unique_label_token_ids(record["labels"]).tolist())
            new_tokens = {token_id for token_id in token_ids if token_id not in selected_counts}
            if len(new_tokens) != metadata["new_label_token_count"]:
                raise ValueError("created-anchor row new-token count does not replay")
            if metadata["guaranteed"]:
                if not (
                    guarantee
                    and metadata["selection_round"] == 1
                    and metadata["task_name"] not in seen_tasks
                    and token_ids
                ):
                    raise ValueError("created-anchor guaranteed row outside a first task turn")
                guaranteed_rows += 1
                if not new_tokens:
                    guaranteed_without_new += 1
            elif not new_tokens:
                raise ValueError("created-anchor row introduces no new label token")
            seen_tasks.add(metadata["task_name"])
            selected_counts.update(token_ids)
            rows += 1
    if digest.hexdigest() != expected_digest:
        raise ValueError("created-anchor dataset SHA-256 mismatch")
    if rows != counts["selected_samples"] or sorted(selected_counts) != universe:
        raise ValueError("created-anchor dataset rows or token coverage mismatch")
    if (
        guaranteed_rows != counts["guaranteed_samples"]
        or guaranteed_without_new != counts["guaranteed_without_new_tokens"]
    ):
        raise ValueError("created-anchor guaranteed row counts do not replay")
    if sum(selected_counts.values()) != counts["selected_token_sample_incidences"]:
        raise ValueError("created-anchor selected token incidences mismatch")
    token_path = _safe_file(root, files["output_token_dict"])
    if hashlib.sha256(token_path.read_bytes()).hexdigest() != _digest_text(
        files["output_token_dict_sha256"], "output token dictionary"
    ):
        raise ValueError("created-anchor output token dictionary SHA-256 mismatch")
    scores_name = files.get("scores")
    if scores_name is not None:
        scores_path = _safe_file(root, scores_name)
        if hashlib.sha256(scores_path.read_bytes()).hexdigest() != _digest_text(
            files.get("scores_sha256"), "scores"
        ):
            raise ValueError("created-anchor scores SHA-256 mismatch")
    return manifest


def _publish_directory(temporary: Path, destination: Path) -> None:
    backup: Path | None = None
    if destination.exists():
        backup = destination.with_name(f".{destination.name}.backup-{uuid4().hex}")
        os.replace(destination, backup)
    try:
        os.replace(temporary, destination)
    except BaseException:
        if backup is not None:
            os.replace(backup, destination)
        raise
    if backup is not None:
        shutil.rmtree(backup)


def _signed(manifest: dict[str, Any]) -> dict[str, Any]:
    body = {key: value for key, value in manifest.items() if key != "signature"}
    body["signature"] = {"algorithm": "sha256", "scope": _SIGNATURE_SCOPE, "sha256": _sha256(body)}
    return body


def write_created_anchor_artifacts(
    *,
    destination: str | Path,
    records: Sequence[Mapping[str, Any]],
    result: TaskVocabCoverResult,
    tables: Mapping[str, LabelSetTable],
    tokenizer: Any,
    model_name: str,
    model_id: str,
    model_revision: str,
    max_length: int,
    sources: Sequence[Mapping[str, Any]],
    selection: Mapping[str, Any],
    tokenizer_vocab_size: int,
    pooled_rows: Mapping[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Publish records (selection order), the token dictionary and a signed manifest.

    ``pooled_rows`` names, per submixture, the rows that entered the task pools;
    source token statistics are restricted to them so that the published
    invariant "every source label token type is covered" holds exactly.
    """
    output = Path(destination).expanduser().absolute()
    if output.exists() and not output.is_dir():
        raise ValueError("created-anchor publication destination is not a directory")
    if output.is_dir() and any(output.iterdir()):
        existing = output / "manifest.json"
        previous: dict[str, Any] = {}
        if existing.is_file():
            try:
                previous = json.loads(existing.read_bytes())
            except ValueError:
                previous = {}
        if previous.get("policy") != ANCHOR_CREATION_POLICY:
            raise ValueError("created-anchor publication must not replace an unrelated directory")
    model = model_identity(model_name, model_id, model_revision)
    if len(records) != len(result.selected):
        raise ValueError("records and selection result lengths differ")
    for side in ("padding_side", "truncation_side"):
        if getattr(tokenizer, side, "right") != "right":
            raise ValueError(f"created-anchor SMART-v3 requires {side}=right")

    digest = hashlib.sha256()
    selected_counts: Counter[int] = Counter()
    sample_ids: list[str] = []
    lines: list[bytes] = []
    for record, chosen in zip(records, result.selected, strict=True):
        record = dict(record)
        _validate_record(record, max_length)
        metadata = record["source_metadata"]
        if (
            metadata["submixture"] != chosen.submixture
            or metadata["task_name"] != chosen.task_name
            or metadata["source_index"] != chosen.source_index
            or metadata["selection_round"] != chosen.round_index
            or metadata["guaranteed"] != chosen.guaranteed
            or metadata["new_label_token_count"] != chosen.new_label_token_count
        ):
            raise ValueError("created-anchor record metadata does not match the selection")
        token_ids = set(unique_label_token_ids(record["labels"]).tolist())
        new_tokens = {token_id for token_id in token_ids if token_id not in selected_counts}
        if len(new_tokens) != chosen.new_label_token_count or (
            not new_tokens and not chosen.guaranteed
        ):
            raise ValueError("created-anchor record does not reproduce the selection decision")
        if chosen.guaranteed and not (result.guarantee_task_coverage and chosen.round_index == 1):
            raise ValueError("created-anchor guaranteed record outside a first task turn")
        selected_counts.update(token_ids)
        sample_ids.append(record["sample_id"])
        line = _json_bytes(record) + b"\n"
        digest.update(line)
        lines.append(line)
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("duplicate created-anchor sample IDs")
    universe = [int(token_id) for token_id in result.covered_token_ids.tolist()]
    if sorted(selected_counts) != universe:
        raise ValueError("created-anchor records do not cover the selected token universe")
    width = max(tokenizer_vocab_size, (universe[-1] + 1) if universe else 0, 1)
    source_counts = source_token_sample_counts(tables, width, pooled_rows)
    token_rows = {
        str(token_id): {
            "token": tokenizer.convert_ids_to_tokens(token_id),
            "source_sample_count": int(source_counts[token_id]),
            "selected_sample_count": int(selected_counts[token_id]),
        }
        for token_id in universe
    }
    token_bytes = (
        _json_bytes(
            {
                "schema_version": ANCHOR_CREATION_SCHEMA_VERSION,
                "policy": ANCHOR_CREATION_POLICY,
                "count_unit": "sample_incidence",
                "tokens": token_rows,
            }
        )
        + b"\n"
    )
    source_token_types = int(np.count_nonzero(source_counts))
    tasks = {
        task_name: {
            "source_samples": int(result.task_source_counts[task_name]),
            "selected_samples": int(result.task_selected_counts[task_name]),
        }
        for task_name in sorted(result.task_source_counts)
    }
    manifest = {
        "schema_version": ANCHOR_CREATION_SCHEMA_VERSION,
        "policy": ANCHOR_CREATION_POLICY,
        "dataset_name": created_anchor_dataset_name(model_name),
        "format": "tokenized_jsonl",
        "model": model,
        "tokenization": tokenization_contract(model, max_length),
        "sources": [dict(source) for source in sources],
        "selection": {
            "algorithm": _SELECTION_ALGORITHM,
            "task_order": "seeded_shuffle_of_remaining_tasks_each_round",
            "within_task_order": "seeded_random_permutation_without_replacement",
            "termination": "every_sample_of_every_task_drawn_once",
            "label_scope": LABEL_SCOPE,
            "causal_shift": 1,
            "ignore_index": -100,
            "guarantee_task_coverage": bool(result.guarantee_task_coverage),
            "first_turn_rule": (
                "accept_first_supervised_sample_unconditionally"
                if result.guarantee_task_coverage
                else "same_new_label_token_rule"
            ),
            "rounds": int(result.rounds),
            **dict(selection),
        },
        "counts": {
            "source_samples": int(result.traversed_samples),
            "traversed_samples": int(result.traversed_samples),
            "unsupervised_samples": int(result.unsupervised_samples),
            "boundary_excluded_samples": int(
                sum(table.boundary_excluded for table in tables.values())
            ),
            "source_tasks": len(result.task_source_counts),
            "selected_samples": len(records),
            "selected_tasks": int(result.selected_tasks),
            "covered_token_types": len(universe),
            "source_token_types": source_token_types,
            "tokenizer_vocab_size": int(tokenizer_vocab_size),
            "source_token_sample_incidences": int(source_counts.sum()),
            "selected_token_sample_incidences": int(sum(selected_counts.values())),
            "guaranteed_samples": int(result.guaranteed_samples),
            "guaranteed_without_new_tokens": int(result.guaranteed_without_new_tokens),
        },
        "tasks": tasks,
        "covered_token_ids": universe,
        "accuracy": None,
        "files": {
            "dataset": "train.jsonl",
            "dataset_sha256": digest.hexdigest(),
            "output_token_dict": "output-token-dict.json",
            "output_token_dict_sha256": hashlib.sha256(token_bytes).hexdigest(),
        },
    }
    manifest = _signed(manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        with (temporary / "train.jsonl").open("wb") as handle:
            for line in lines:
                handle.write(line)
        (temporary / "output-token-dict.json").write_bytes(token_bytes)
        (temporary / "manifest.json").write_bytes(_json_bytes(manifest) + b"\n")
        load_created_anchor_manifest(
            temporary,
            model_name=model_name,
            model_id=model_id,
            model_revision=model_revision,
            max_length=max_length,
        )
        _publish_directory(temporary, output)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    return manifest


def attach_accuracy_report(
    directory: str | Path,
    *,
    model_name: str,
    model_id: str,
    model_revision: str,
    max_length: int,
    accuracy: Mapping[str, Any],
    scores: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Add per-sample base-model correctness scores to an existing publication."""
    root = Path(directory).expanduser().absolute()
    manifest = load_created_anchor_manifest(
        root,
        model_name=model_name,
        model_id=model_id,
        model_revision=model_revision,
        max_length=max_length,
        verify_records=False,
    )
    if len(scores) != manifest["counts"]["selected_samples"]:
        raise ValueError("accuracy scores must cover every selected sample")
    score_bytes = b"".join(_json_bytes(dict(score)) + b"\n" for score in scores)
    temporary_scores = root / f".scores.jsonl.tmp-{uuid4().hex}"
    temporary_scores.write_bytes(score_bytes)
    os.replace(temporary_scores, root / "scores.jsonl")
    updated = {key: value for key, value in manifest.items() if key != "signature"}
    updated["accuracy"] = dict(accuracy)
    updated["files"] = {
        **manifest["files"],
        "scores": "scores.jsonl",
        "scores_sha256": hashlib.sha256(score_bytes).hexdigest(),
    }
    updated = _signed(updated)
    temporary_manifest = root / f".manifest.json.tmp-{uuid4().hex}"
    temporary_manifest.write_bytes(_json_bytes(updated) + b"\n")
    os.replace(temporary_manifest, root / "manifest.json")
    return updated


def summarize_scores(scores: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not scores:
        raise ValueError("cannot summarize an empty score list")
    correct = np.asarray([int(score["correct_tokens"]) for score in scores], dtype=np.int64)
    total = np.asarray([int(score["total_tokens"]) for score in scores], dtype=np.int64)
    if np.any(total <= 0) or np.any(correct < 0) or np.any(correct > total):
        raise ValueError("invalid correctness scores")
    per_sample = correct / total
    return {
        "samples": int(total.size),
        "mean_sample_accuracy": float(per_sample.mean()),
        "token_accuracy": float(correct.sum() / total.sum()),
        "fully_correct_samples": int(np.count_nonzero(correct == total)),
        "correct_tokens": int(correct.sum()),
        "total_tokens": int(total.sum()),
    }
