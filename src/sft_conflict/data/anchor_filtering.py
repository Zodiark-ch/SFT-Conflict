"""Model-correct Anchor filtering with deterministic token set cover."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from heapq import heapify, heappop, heappush
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from .collator import CausalLMCollator, model_tensor_inputs

FILTERED_ANCHOR_ROOT = "filtered_anchor_datasets"
FILTER_SCORE_ROOT = "anchor_filter_scores"


def filtered_anchor_dataset_name(model_name: str) -> str:
    return f"smart_flan_{model_name}"


def filtered_anchor_dataset_dir(artifact_root: str | Path, model_name: str) -> Path:
    return (
        Path(artifact_root).expanduser().resolve()
        / FILTERED_ANCHOR_ROOT
        / filtered_anchor_dataset_name(model_name)
    )


def anchor_filter_score_dir(
    artifact_root: str | Path,
    model_name: str,
    max_length: int,
) -> Path:
    return (
        Path(artifact_root).expanduser().resolve()
        / FILTER_SCORE_ROOT
        / filtered_anchor_dataset_name(model_name)
        / f"max-length-{max_length}"
    )


def _publish_directory(temporary_dir: Path, output_dir: Path) -> None:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if not output_dir.exists():
        os.replace(temporary_dir, output_dir)
        return

    backup_dir = output_dir.with_name(output_dir.name + f".backup-{os.getpid()}")
    if backup_dir.exists():
        raise FileExistsError(f"stale replacement backup exists: {backup_dir}")
    os.replace(output_dir, backup_dir)
    try:
        os.replace(temporary_dir, output_dir)
    except BaseException:
        os.replace(backup_dir, output_dir)
        raise
    shutil.rmtree(backup_dir)


@dataclass(frozen=True)
class AnchorFilterScore:
    source_index: int
    correct_tokens: int
    total_tokens: int

    @property
    def accuracy(self) -> float:
        return self.correct_tokens / self.total_tokens


@dataclass(frozen=True)
class TokenCoverSelection:
    selected_indices: tuple[int, ...]
    universe_token_ids: tuple[int, ...]
    eligible_sample_counts: dict[int, int]
    selected_sample_counts: dict[int, int]


def supervised_output_token_ids(record: Mapping[str, Any]) -> frozenset[int]:
    return frozenset(int(token_id) for token_id in record["labels"][1:] if token_id != -100)


def response_token_accuracy(
    hidden_states: Tensor,
    labels: Tensor,
    output_head: nn.Module,
) -> tuple[Tensor, Tensor]:
    if hidden_states.ndim != 3 or labels.ndim != 2:
        raise ValueError(
            "hidden_states must be [batch, sequence, hidden] and labels [batch, sequence]"
        )
    if hidden_states.shape[:2] != labels.shape:
        raise ValueError(
            "hidden_states and labels must have matching batch and sequence dimensions"
        )

    shifted_labels = labels[:, 1:]
    supervised = shifted_labels.ne(-100)
    coordinates = supervised.nonzero(as_tuple=False)
    batch_size = labels.shape[0]
    if coordinates.numel() == 0:
        zeros = torch.zeros(batch_size, dtype=torch.long, device=labels.device)
        return zeros, zeros

    selected_hidden = hidden_states[:, :-1][supervised]
    predictions = output_head(selected_hidden).argmax(dim=-1)
    targets = shifted_labels[supervised]
    rows = coordinates[:, 0]
    total_counts = torch.bincount(rows, minlength=batch_size)
    correct_counts = torch.bincount(rows[predictions.eq(targets)], minlength=batch_size)
    return correct_counts, total_counts


def greedy_token_cover(
    sample_token_ids: Mapping[int, Iterable[int]],
) -> TokenCoverSelection:
    token_sets = {
        int(sample_index): frozenset(int(token_id) for token_id in token_ids)
        for sample_index, token_ids in sample_token_ids.items()
        if token_ids
    }
    if not token_sets:
        raise ValueError("token cover requires at least one non-empty eligible sample")

    eligible_counts = Counter(
        token_id for token_ids in token_sets.values() for token_id in token_ids
    )
    uncovered = set(eligible_counts)
    heap = [(-len(token_ids), sample_index) for sample_index, token_ids in token_sets.items()]
    heapify(heap)
    selected: list[int] = []
    while uncovered:
        if not heap:
            raise RuntimeError(f"token cover left {len(uncovered)} tokens uncovered")
        negative_cached_gain, sample_index = heappop(heap)
        gain = token_sets[sample_index] & uncovered
        if not gain:
            continue
        if len(gain) != -negative_cached_gain:
            heappush(heap, (-len(gain), sample_index))
            continue
        selected.append(sample_index)
        uncovered.difference_update(gain)

    selected_counts = Counter(
        token_id for sample_index in selected for token_id in token_sets[sample_index]
    )
    retained: list[int] = []
    for sample_index in reversed(selected):
        token_ids = token_sets[sample_index]
        if all(selected_counts[token_id] > 1 for token_id in token_ids):
            selected_counts.subtract(token_ids)
        else:
            retained.append(sample_index)
    retained.reverse()
    retained_indices = tuple(sorted(retained))
    selected_counts = Counter(
        token_id for sample_index in retained_indices for token_id in token_sets[sample_index]
    )
    if set(selected_counts) != set(eligible_counts) or min(selected_counts.values()) < 1:
        raise RuntimeError("token-cover pruning changed the eligible output-token universe")

    return TokenCoverSelection(
        selected_indices=retained_indices,
        universe_token_ids=tuple(sorted(eligible_counts)),
        eligible_sample_counts=dict(sorted(eligible_counts.items())),
        selected_sample_counts=dict(sorted(selected_counts.items())),
    )


def score_anchor_shard(
    *,
    model: nn.Module,
    dataset: Sequence[Mapping[str, Any]],
    collator: CausalLMCollator,
    device: torch.device,
    batch_size: int,
    rank: int,
    world_size: int,
    output_path: str | Path,
) -> None:
    if batch_size <= 0:
        raise ValueError("filtering batch size must be positive")
    transformer = getattr(model, "model", None)
    output_head = model.get_output_embeddings()
    if not isinstance(transformer, nn.Module) or not isinstance(output_head, nn.Module):
        raise TypeError("Anchor filtering requires a causal LM with a model body and output head")

    ordered_indices = sorted(
        range(rank, len(dataset), world_size),
        key=lambda index: (len(dataset[index]["input_ids"]), index),
    )
    loader = DataLoader(
        Subset(dataset, ordered_indices),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    cursor = 0
    progress = tqdm(
        total=len(ordered_indices),
        desc=f"Anchor correctness rank {rank}",
        unit="sample",
        dynamic_ncols=True,
        disable=rank != 0,
    )
    model.eval()
    with temporary.open("w", encoding="utf-8") as handle, torch.inference_mode():
        for batch in loader:
            current_size = int(batch["labels"].shape[0])
            batch_indices = ordered_indices[cursor : cursor + current_size]
            cursor += current_size
            tensor_inputs = {
                name: value.to(device, non_blocking=True)
                for name, value in model_tensor_inputs(batch).items()
            }
            labels = batch["labels"].to(device, non_blocking=True)
            outputs = transformer(**tensor_inputs, use_cache=False, return_dict=True)
            correct_counts, total_counts = response_token_accuracy(
                outputs.last_hidden_state,
                labels,
                output_head,
            )
            for sample_index, correct, total in zip(
                batch_indices,
                correct_counts.cpu().tolist(),
                total_counts.cpu().tolist(),
                strict=True,
            ):
                if total <= 0:
                    raise ValueError(
                        f"Anchor sample {sample_index} has no shifted supervised tokens"
                    )
                handle.write(
                    json.dumps(
                        {
                            "source_index": sample_index,
                            "correct_tokens": correct,
                            "total_tokens": total,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
            del outputs, correct_counts, total_counts, labels, tensor_inputs
            progress.update(current_size)
    progress.close()
    if cursor != len(ordered_indices):
        raise RuntimeError(f"rank {rank} scored {cursor} samples, expected {len(ordered_indices)}")
    os.replace(temporary, destination)


def read_filter_scores(
    paths: Iterable[str | Path],
    *,
    expected_samples: int,
) -> tuple[AnchorFilterScore, ...]:
    scores: dict[int, AnchorFilterScore] = {}
    for path in paths:
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                value = json.loads(line)
                score = AnchorFilterScore(
                    source_index=int(value["source_index"]),
                    correct_tokens=int(value["correct_tokens"]),
                    total_tokens=int(value["total_tokens"]),
                )
                if score.source_index in scores:
                    raise ValueError(f"duplicate Anchor filter score index: {score.source_index}")
                if score.total_tokens <= 0 or not 0 <= score.correct_tokens <= score.total_tokens:
                    raise ValueError(f"invalid Anchor filter score: {score}")
                scores[score.source_index] = score
    expected_indices = set(range(expected_samples))
    if set(scores) != expected_indices:
        missing = sorted(expected_indices - set(scores))[:10]
        unexpected = sorted(set(scores) - expected_indices)[:10]
        raise ValueError(
            "Anchor filter scores do not cover the source dataset "
            f"(missing={missing}, unexpected={unexpected})"
        )
    return tuple(scores[index] for index in range(expected_samples))


def write_filter_score_cache(
    *,
    destination: str | Path,
    score_paths: Sequence[str | Path],
    model_name: str,
    model_id: str,
    model_revision: str,
    max_length: int,
    source_dataset_id: str,
    source_dataset_revision: str | None,
    source_fingerprint: str,
    source_samples: int,
) -> dict[str, Any]:
    output_dir = Path(destination)
    temporary_dir = output_dir.with_name(output_dir.name + f".tmp-{os.getpid()}")
    if temporary_dir.exists():
        shutil.rmtree(temporary_dir)
    temporary_dir.mkdir(parents=True, exist_ok=False)
    try:
        files: list[str] = []
        for score_path in score_paths:
            source = Path(score_path)
            target = temporary_dir / source.name
            shutil.copy2(source, target)
            files.append(target.name)
        manifest = {
            "schema_version": 1,
            "format": "teacher_forced_anchor_correctness_scores",
            "model": {
                "preset": model_name,
                "model_id": model_id,
                "revision": model_revision,
            },
            "tokenization": {"max_length": max_length},
            "source": {
                "dataset_id": source_dataset_id,
                "revision": source_dataset_revision,
                "fingerprint": source_fingerprint,
                "samples": source_samples,
            },
            "score_files": files,
        }
        (temporary_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _publish_directory(temporary_dir, output_dir)
        return manifest
    except BaseException:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise


def reusable_filter_score_paths(
    *,
    score_dir: str | Path,
    model_name: str,
    model_id: str,
    model_revision: str,
    max_length: int,
    source_dataset_id: str,
    source_dataset_revision: str | None,
) -> tuple[Path, ...] | None:
    directory = Path(score_dir)
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != 1
        or manifest.get("format") != "teacher_forced_anchor_correctness_scores"
        or manifest.get("model", {}).get("preset") != model_name
        or manifest.get("model", {}).get("model_id") != model_id
        or manifest.get("model", {}).get("revision") != model_revision
        or manifest.get("tokenization", {}).get("max_length") != max_length
        or manifest.get("source", {}).get("dataset_id") != source_dataset_id
        or manifest.get("source", {}).get("revision") != source_dataset_revision
    ):
        return None
    paths = tuple(directory / str(name) for name in manifest.get("score_files", ()))
    if not paths or any(not path.is_file() for path in paths):
        return None
    return paths


def write_filtered_anchor_artifacts(
    *,
    destination: str | Path,
    dataset: Sequence[Mapping[str, Any]],
    scores: Sequence[AnchorFilterScore],
    threshold: float,
    selection: TokenCoverSelection,
    tokenizer: Any,
    model_name: str,
    model_id: str,
    model_revision: str,
    max_length: int,
    source_fingerprint: str,
) -> dict[str, Any]:
    output_dir = Path(destination)
    temporary_dir = output_dir.with_name(output_dir.name + f".tmp-{os.getpid()}")
    if temporary_dir.exists():
        shutil.rmtree(temporary_dir)
    temporary_dir.mkdir(parents=True, exist_ok=False)
    try:
        dataset_path = temporary_dir / "train.jsonl"
        digest = hashlib.sha256()
        selected_ids: list[str] = []
        with dataset_path.open("w", encoding="utf-8") as handle:
            for source_index in selection.selected_indices:
                record = dict(dataset[source_index])
                selected_ids.append(str(record["sample_id"]))
                line = json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n"
                handle.write(line)
                digest.update(line.encode())

        token_rows = {
            str(token_id): {
                "token": tokenizer.convert_ids_to_tokens(token_id),
                "eligible_sample_count": selection.eligible_sample_counts[token_id],
                "selected_sample_count": selection.selected_sample_counts[token_id],
            }
            for token_id in selection.universe_token_ids
        }
        token_counts_path = temporary_dir / "output-token-dict.json"
        token_counts_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "count_unit": "sample_incidence",
                    "tokens": token_rows,
                },
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        eligible_scores = [score for score in scores if score.accuracy + 1e-12 >= threshold]
        manifest = {
            "schema_version": 1,
            "dataset_name": filtered_anchor_dataset_name(model_name),
            "format": "tokenized_jsonl",
            "model": {
                "preset": model_name,
                "model_id": model_id,
                "revision": model_revision,
            },
            "tokenization": {"max_length": max_length},
            "filter": {
                "threshold": threshold,
                "accuracy": "teacher_forced_shifted_response_token_argmax",
                "source_samples": len(dataset),
                "eligible_samples": len(eligible_scores),
                "incorrect_samples": len(dataset) - len(eligible_scores),
            },
            "cover": {
                "algorithm": "lazy_greedy_set_cover_then_reverse_redundancy_prune",
                "global_optimum_guaranteed": False,
                "eligible_token_types": len(selection.universe_token_ids),
                "selected_token_types": len(selection.selected_sample_counts),
                "selected_samples": len(selection.selected_indices),
                "eligible_to_selected_ratio": len(eligible_scores)
                / len(selection.selected_indices),
            },
            "source_fingerprint": source_fingerprint,
            "selected_source_indices": list(selection.selected_indices),
            "selected_sample_ids": selected_ids,
            "files": {
                "dataset": dataset_path.name,
                "dataset_sha256": digest.hexdigest(),
                "output_token_dict": token_counts_path.name,
            },
        }
        (temporary_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _publish_directory(temporary_dir, output_dir)
        return manifest
    except BaseException:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise
