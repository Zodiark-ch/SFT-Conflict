"""Click-run workflow for METHOD="create_anchor_dataset".

Builds a model-specific Anchor dataset directly from the five FLAN 2022
submixtures that SMART samples from (not from the published 400k ``smart_flan``
subset), using the task round-robin vocabulary cover in
:mod:`sft_conflict.data.anchor_creation`.  Runs on CPU; the optional base-model
accuracy report uses the first visible GPU.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from datasets import ClassLabel, DownloadConfig, load_dataset

from sft_conflict.cache_preparation import (
    PresetCacheError,
    _cache_paths,
    _dataset_cache_ready,
    _label_coverage_model_id,
    _load_label_coverage_tokenizer,
    _model_cache_ready,
    _prepare_dataset,
    _prepare_model,
    _prepare_tokenizer,
    _tokenizer_cache_ready,
)
from sft_conflict.config import DatasetConfig, MethodName, TrainingMode
from sft_conflict.data.anchor_creation import (
    ANCHOR_CREATION_POLICY,
    SMART_ADAPTER,
    SMART_TEMPLATE_VERSION,
    SUBMIXTURE_KEYS,
    LabelSetTable,
    SubmixtureSpec,
    TaskPool,
    TaskVocabCoverResult,
    anchor_label_cache_dir,
    attach_accuracy_report,
    build_label_set_table,
    created_anchor_dataset_dir,
    label_cache_identity,
    load_created_anchor_manifest,
    load_label_set_table,
    model_identity,
    render_smart_row,
    save_label_set_table,
    select_task_vocab_cover,
    submixture_spec,
    summarize_scores,
    unique_label_token_ids,
    write_created_anchor_artifacts,
)
from sft_conflict.data.anchor_filtering import read_filter_scores, score_anchor_shard
from sft_conflict.data.collator import CausalLMCollator
from sft_conflict.data.loading import _sample_namespace, tokenize_prompt_response
from sft_conflict.data.records import make_sample_id
from sft_conflict.quick_run import MODEL_PRESETS, DatasetPreset, QuickRunSettings

ACTION = "create-anchor-dataset"


@dataclass(frozen=True)
class AnchorCreationOptions:
    seed: int = 42
    workers: int = 8
    scoring_batch_size: int = 4
    compute_accuracy: bool = True
    submixtures: tuple[str, ...] = SUBMIXTURE_KEYS
    max_samples_per_task: int | None = None
    label_chunk_rows: int = 1024
    guarantee_task_coverage: bool = True

    def validate(self) -> None:
        if type(self.seed) is not int:
            raise ValueError("anchor creation seed must be an integer")
        if type(self.guarantee_task_coverage) is not bool:
            raise ValueError("guarantee_task_coverage must be a boolean")
        if type(self.workers) is not int or self.workers < 0:
            raise ValueError("anchor creation workers must be a non-negative integer")
        if type(self.scoring_batch_size) is not int or self.scoring_batch_size <= 0:
            raise ValueError("anchor creation scoring batch size must be positive")
        if not self.submixtures or len(set(self.submixtures)) != len(self.submixtures):
            raise ValueError("anchor creation requires a non-empty list of distinct submixtures")
        for key in self.submixtures:
            submixture_spec(key)
        if self.max_samples_per_task is not None and (
            type(self.max_samples_per_task) is not int or self.max_samples_per_task <= 0
        ):
            raise ValueError("max_samples_per_task must be a positive integer or None")
        if type(self.label_chunk_rows) is not int or self.label_chunk_rows <= 0:
            raise ValueError("label_chunk_rows must be positive")


def _log(message: str) -> None:
    print(message, flush=True)


def submixture_preset(spec: SubmixtureSpec) -> DatasetPreset:
    return DatasetPreset(
        path=spec.dataset_id,
        revision=spec.revision,
        name=None,
        train_split=spec.split,
        eval_split=spec.split,
        adapter=SMART_ADAPTER,
    )


def _ensure_sources(
    settings: QuickRunSettings,
    specs: Sequence[SubmixtureSpec],
    *,
    auto_prepare: bool,
    need_model: bool,
) -> None:
    missing: list[str] = []
    if not _tokenizer_cache_ready(settings):
        missing.append(f"tokenizer={settings.model}")
    missing_sources = [
        spec
        for spec in specs
        if not _dataset_cache_ready(submixture_preset(spec), settings.artifact_root)
    ]
    missing.extend(f"submixture={spec.key}" for spec in missing_sources)
    missing_model = need_model and not _model_cache_ready(settings)
    if missing_model:
        missing.append(f"model={settings.model}")
    if not missing:
        return
    if not auto_prepare:
        raise PresetCacheError(
            "Anchor creation cache is incomplete: "
            + ", ".join(missing)
            + ". Set AUTO_PREPARE_CACHE=True to download the fixed tokenizer, the SMART "
            "FLAN 2022 submixtures (about 16 GB of parquet files) and the model weights."
        )
    _log("Preparing anchor creation cache: " + ", ".join(missing))
    if f"tokenizer={settings.model}" in missing:
        _prepare_tokenizer(settings)
        if not _tokenizer_cache_ready(settings):
            raise PresetCacheError(
                f"tokenizer cache preparation did not complete: {settings.model}"
            )
    for spec in missing_sources:
        _prepare_dataset(submixture_preset(spec), settings)
        if not _dataset_cache_ready(submixture_preset(spec), settings.artifact_root):
            raise PresetCacheError(f"submixture cache preparation did not complete: {spec.key}")
    if missing_model:
        _prepare_model(settings)
        if not _model_cache_ready(settings):
            raise PresetCacheError(f"model cache preparation did not complete: {settings.model}")


def load_submixture(spec: SubmixtureSpec, settings: QuickRunSettings) -> Any:
    _, dataset_cache = _cache_paths(settings.artifact_root)
    dataset = load_dataset(
        spec.dataset_id,
        None,
        split=spec.split,
        revision=spec.revision,
        cache_dir=str(dataset_cache),
        download_config=DownloadConfig(local_files_only=True),
    )
    # In offline mode `datasets` opens the most recently modified cached snapshot
    # regardless of `revision`; insist that the arrow files belong to the pin.
    cache_files = [str(entry.get("filename", "")) for entry in getattr(dataset, "cache_files", [])]
    if not cache_files or not all(spec.revision in Path(name).parts for name in cache_files):
        raise PresetCacheError(
            f"{spec.key}: the offline dataset cache resolved to a snapshot other than revision "
            f"{spec.revision} (files: {cache_files[:1]}); remove the other cached revisions of "
            f"{spec.dataset_id} or prepare the fixed revision again"
        )
    return dataset


def _task_codes(dataset: Any) -> tuple[np.ndarray, list[str]]:
    feature = dataset.features["task_name"]
    values = dataset["task_name"]
    if isinstance(feature, ClassLabel):
        names = [str(name) for name in feature.names]
        codes = np.asarray(values, dtype=np.int64)
    else:
        names = sorted({str(value) for value in values})
        index = {name: position for position, name in enumerate(names)}
        codes = np.fromiter(
            (index[str(value)] for value in values), dtype=np.int64, count=len(values)
        )
    if codes.size and (int(codes.min()) < 0 or int(codes.max()) >= len(names)):
        raise ValueError("task_name codes are outside the ClassLabel range")
    return codes, names


def _task_pools(
    key: str,
    dataset: Any,
    *,
    max_samples_per_task: int | None,
    seed: int,
) -> list[TaskPool]:
    codes, names = _task_codes(dataset)
    order = np.argsort(codes, kind="stable")
    sorted_codes = codes[order]
    unique_codes, starts = np.unique(sorted_codes, return_index=True)
    ends = np.append(starts[1:], sorted_codes.size)
    pools: list[TaskPool] = []
    rng = np.random.default_rng([int(seed), int.from_bytes(key.encode("utf-8"), "big")])
    for code, start, end in zip(unique_codes.tolist(), starts.tolist(), ends.tolist(), strict=True):
        indices = np.sort(order[start:end])
        if max_samples_per_task is not None and indices.size > max_samples_per_task:
            indices = np.sort(rng.choice(indices, size=max_samples_per_task, replace=False))
        pools.append(TaskPool(submixture=key, task_name=names[code], sample_indices=indices))
    return pools


def _row_view(dataset: Any) -> Any:
    select_columns = getattr(dataset, "select_columns", None)
    if callable(select_columns):
        return select_columns(["inputs", "targets"])
    return dataset


def _label_table(
    spec: SubmixtureSpec,
    dataset: Any,
    settings: QuickRunSettings,
    tokenizer: Any,
    model: Mapping[str, str],
    options: AnchorCreationOptions,
) -> tuple[LabelSetTable, bool]:
    cache_dir = anchor_label_cache_dir(settings.artifact_root, settings.model, settings.max_length)
    identity = label_cache_identity(
        spec, rows=len(dataset), model=model, max_length=settings.max_length
    )
    cached = load_label_set_table(cache_dir, spec, identity)
    if cached is not None:
        _log(f"[{spec.key}] reusing cached label sets for {len(cached):,} rows: {cache_dir}")
        return cached, True
    started = time.monotonic()
    table = build_label_set_table(
        _row_view(dataset),
        tokenizer,
        max_length=settings.max_length,
        workers=options.workers,
        chunk_rows=options.label_chunk_rows,
        progress_description=f"[{spec.key}]",
    )
    save_label_set_table(cache_dir, spec, table, identity)
    _log(
        f"[{spec.key}] tokenized {len(table):,} rows in {time.monotonic() - started:.0f}s "
        f"(unsupervised={int(np.count_nonzero(table.lengths() == 0)):,}, "
        f"boundary_excluded={table.boundary_excluded:,}); cached at {cache_dir}"
    )
    return table, False


def _materialize_records(
    result: TaskVocabCoverResult,
    datasets_by_key: Mapping[str, Any],
    specs_by_key: Mapping[str, SubmixtureSpec],
    tables: Mapping[str, LabelSetTable],
    task_names_by_key: Mapping[str, list[str]],
    tokenizer: Any,
    *,
    max_length: int,
) -> list[dict[str, Any]]:
    from tqdm.auto import tqdm

    namespaces = {
        key: _sample_namespace(
            DatasetConfig(
                source="hf",
                path=spec.dataset_id,
                split=spec.split,
                revision=spec.revision,
                name=None,
                adapter=SMART_ADAPTER,
            ),
            SMART_TEMPLATE_VERSION,
        )
        for key, spec in specs_by_key.items()
    }
    records: list[dict[str, Any]] = []
    progress = tqdm(
        total=len(result.selected),
        desc="materializing selected records",
        unit="sample",
        dynamic_ncols=True,
    )
    try:
        for chosen in result.selected:
            dataset = datasets_by_key[chosen.submixture]
            spec = specs_by_key[chosen.submixture]
            row = dict(dataset[int(chosen.source_index)])
            prompt, response = render_smart_row(row)
            tokenized = tokenize_prompt_response(
                tokenizer,
                prompt,
                response,
                max_length=max_length,
                model_variant="base",
                allow_response_whitespace_bridge=True,
            )
            expected = tables[chosen.submixture].label_ids(chosen.source_index)
            actual = unique_label_token_ids(tokenized["labels"])
            if not np.array_equal(expected, actual):
                raise RuntimeError(
                    f"label set mismatch for {chosen.submixture}[{chosen.source_index}] between "
                    "batched and single tokenization"
                )
            task_value = row.get("task_name")
            if isinstance(task_value, int):
                names = task_names_by_key[chosen.submixture]
                task_value = names[task_value] if 0 <= task_value < len(names) else str(task_value)
            if str(task_value) != chosen.task_name:
                raise RuntimeError("selected record task name does not match its task pool")
            normalized_text = " ".join(f"{prompt}\n{response}".casefold().split())
            record: dict[str, Any] = {
                "sample_id": make_sample_id(
                    namespaces[chosen.submixture], chosen.source_index, row
                ),
                "input_ids": list(tokenized["input_ids"]),
                "attention_mask": list(tokenized["attention_mask"]),
                "labels": list(tokenized["labels"]),
                # Top-level task keys let the pretokenized adapter expose task
                # metadata for task_balanced Anchor sampling.
                "task_name": chosen.task_name,
                "task_source": str(row.get("task_source", "")),
                "template_type": str(row.get("template_type", "")),
                "source_metadata": {
                    "dataset_name": spec.dataset_id,
                    "dataset_revision": spec.revision,
                    "split": spec.split,
                    "template_version": SMART_TEMPLATE_VERSION,
                    "source_index": int(chosen.source_index),
                    "submixture": chosen.submixture,
                    "task_name": chosen.task_name,
                    "task_source": str(row.get("task_source", "")),
                    "template_type": str(row.get("template_type", "")),
                    "task_metadata_available": True,
                    "selection_round": int(chosen.round_index),
                    "selection_order": int(chosen.selection_order),
                    "new_label_token_count": int(chosen.new_label_token_count),
                    "guaranteed": bool(chosen.guaranteed),
                    "normalized_content_sha256": hashlib.sha256(
                        normalized_text.encode()
                    ).hexdigest(),
                },
            }
            records.append(record)
            progress.update(1)
    finally:
        progress.close()
    return records


def _score_records(
    settings: QuickRunSettings,
    records: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
    destination: Path,
) -> list[dict[str, Any]]:
    import torch

    from sft_conflict.models import load_model_bundle
    from sft_conflict.quick_run import build_quick_run_config

    config = build_quick_run_config(settings)
    model_config = replace(
        config.model,
        training_mode=TrainingMode.FULL_FT,
        freeze_unembedding=False,
        untie_tied_embeddings=False,
        gradient_checkpointing=False,
    )
    hub_cache, _ = _cache_paths(settings.artifact_root)
    bundle = load_model_bundle(model_config, config.lora, config.training, cache_dir=hub_cache)
    model = bundle.model
    model.requires_grad_(False)
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    _log(
        f"Scoring {len(records):,} selected samples with {settings.model} "
        f"({config.training.mixed_precision}) on {device}, batch size {batch_size}"
    )
    score_path = destination.parent / f".{destination.name}.scores-{uuid4().hex}.jsonl"
    try:
        score_anchor_shard(
            model=model,
            dataset=records,
            collator=CausalLMCollator(
                bundle.tokenizer.pad_token_id, config.data.pad_to_multiple_of
            ),
            device=device,
            batch_size=batch_size,
            rank=0,
            world_size=1,
            output_path=score_path,
        )
        scores = read_filter_scores([score_path], expected_samples=len(records))
    finally:
        if score_path.exists():
            score_path.unlink()
        del model, bundle
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return [
        {
            "source_index": score.source_index,
            "sample_id": records[score.source_index]["sample_id"],
            "correct_tokens": score.correct_tokens,
            "total_tokens": score.total_tokens,
        }
        for score in scores
    ]


def _print_summary(manifest: Mapping[str, Any], destination: Path, *, reused: bool) -> None:
    counts = manifest["counts"]
    accuracy = manifest.get("accuracy")
    coverage = counts["covered_token_types"]
    vocab = counts["tokenizer_vocab_size"]
    _log("")
    _log("==== create_anchor_dataset summary ====")
    _log(f"output: {destination}{' (reused existing publication)' if reused else ''}")
    _log(
        f"samples: {counts['selected_samples']:,} selected from "
        f"{counts['source_samples']:,} source samples in "
        f"{manifest['selection']['rounds']:,} rounds"
    )
    _log(f"tasks: {counts['selected_tasks']:,} / {counts['source_tasks']:,} source tasks")
    if manifest["selection"].get("guarantee_task_coverage"):
        _log(
            f"task coverage guarantee: on; {counts['guaranteed_samples']:,} first-turn samples "
            f"accepted unconditionally, {counts['guaranteed_without_new_tokens']:,} of them "
            "without a new label token"
        )
    else:
        _log("task coverage guarantee: off (every acceptance required a new label token)")
    _log(
        f"vocabulary coverage: {coverage:,} / {vocab:,} tokenizer token types "
        f"({100.0 * coverage / max(vocab, 1):.2f}%); all {counts['source_token_types']:,} "
        "label token types present in the sources are covered"
    )
    if accuracy is None:
        _log("average accuracy: not computed (ANCHOR_CREATION_COMPUTE_ACCURACY=False)")
    else:
        _log(
            f"average accuracy: mean per-sample {accuracy['mean_sample_accuracy']:.4f}, "
            f"token-level {accuracy['token_accuracy']:.4f}, fully correct samples "
            f"{accuracy['fully_correct_samples']:,} / {accuracy['samples']:,} "
            f"({accuracy['metric']}, {accuracy['model']['preset']}, {accuracy['precision']})"
        )
    _log("=======================================")


def _result(manifest: Mapping[str, Any], destination: Path, *, reused: bool) -> dict[str, Any]:
    counts = manifest["counts"]
    return {
        "action": ACTION,
        "policy": ANCHOR_CREATION_POLICY,
        "output_dataset": manifest["dataset_name"],
        "output_dir": str(destination),
        "manifest": str(destination / "manifest.json"),
        "source_samples": counts["source_samples"],
        "source_tasks": counts["source_tasks"],
        "selected_samples": counts["selected_samples"],
        "selected_tasks": counts["selected_tasks"],
        "covered_token_types": counts["covered_token_types"],
        "tokenizer_vocab_size": counts["tokenizer_vocab_size"],
        "rounds": manifest["selection"]["rounds"],
        "seed": manifest["selection"]["seed"],
        "guarantee_task_coverage": manifest["selection"]["guarantee_task_coverage"],
        "guaranteed_samples": counts["guaranteed_samples"],
        "guaranteed_without_new_tokens": counts["guaranteed_without_new_tokens"],
        "accuracy": manifest.get("accuracy"),
        "reused_existing_dataset": reused,
    }


def _matching_publication(
    destination: Path,
    *,
    model_name: str,
    model_id: str,
    model_revision: str,
    max_length: int,
    options: AnchorCreationOptions,
    specs: Sequence[SubmixtureSpec],
) -> dict[str, Any] | None:
    if not (destination / "manifest.json").is_file():
        return None
    try:
        # Full verification: signed manifest, file hashes and the replay of the
        # "every record introduces a new label token" property.
        manifest = load_created_anchor_manifest(
            destination,
            model_name=model_name,
            model_id=model_id,
            model_revision=model_revision,
            max_length=max_length,
            verify_records=True,
        )
    except (ValueError, OSError) as exc:
        _log(
            f"Existing publication at {destination} failed verification ({exc}); "
            "it will be replaced after a fresh selection."
        )
        return None
    selection = manifest["selection"]
    sources = [
        (source["submixture"], source["dataset_id"], source["revision"], source["split"])
        for source in manifest["sources"]
    ]
    expected = [(spec.key, spec.dataset_id, spec.revision, spec.split) for spec in specs]
    mismatches: list[str] = []
    if selection.get("seed") != options.seed:
        mismatches.append(f"seed {selection.get('seed')!r} != {options.seed!r}")
    if selection.get("max_samples_per_task") != options.max_samples_per_task:
        mismatches.append(
            f"max_samples_per_task {selection.get('max_samples_per_task')!r} != "
            f"{options.max_samples_per_task!r}"
        )
    if sources != expected:
        mismatches.append("submixtures or their revisions differ")
    if selection.get("guarantee_task_coverage") != options.guarantee_task_coverage:
        mismatches.append(
            f"guarantee_task_coverage {selection.get('guarantee_task_coverage')!r} != "
            f"{options.guarantee_task_coverage!r}"
        )
    if mismatches:
        _log(
            f"Existing publication at {destination} does not match the requested settings "
            f"({'; '.join(mismatches)}); it will be replaced after a fresh selection."
        )
        return None
    return manifest


def create_anchor_dataset(
    settings: QuickRunSettings,
    options: AnchorCreationOptions | None = None,
    *,
    auto_prepare: bool,
) -> dict[str, Any]:
    """Create (or reuse) the task-vocabulary-cover Anchor dataset for ``settings.model``."""
    options = options or AnchorCreationOptions()
    options.validate()
    settings.validate()
    if (
        settings.action != ACTION
        or MethodName(settings.method) is not MethodName.CREATE_ANCHOR_DATASET
    ):
        raise ValueError("create_anchor_dataset requires action=create-anchor-dataset")
    if (
        "RANK" in os.environ
        or "LOCAL_RANK" in os.environ
        or os.environ.get("WORLD_SIZE", "1") != "1"
    ):
        raise PresetCacheError("create_anchor_dataset is a single-process job; do not use torchrun")
    specs = tuple(submixture_spec(key) for key in options.submixtures)
    model_preset = MODEL_PRESETS[settings.model]
    model_id = _label_coverage_model_id(settings)
    model = model_identity(settings.model, model_id, model_preset.revision)
    destination = created_anchor_dataset_dir(
        settings.artifact_root, settings.model, settings.max_length
    )
    need_model = options.compute_accuracy and model_preset.model_id is not None

    existing = _matching_publication(
        destination,
        model_name=settings.model,
        model_id=model_id,
        model_revision=model_preset.revision,
        max_length=settings.max_length,
        options=options,
        specs=specs,
    )
    if existing is not None:
        _log(f"Existing created Anchor publication matches the requested settings: {destination}")
        accuracy = existing.get("accuracy")
        stale_accuracy = (
            accuracy is not None and accuracy.get("precision") != settings.mixed_precision
        )
        if options.compute_accuracy and (accuracy is None or stale_accuracy):
            if stale_accuracy:
                _log(
                    f"Recomputing accuracy: the existing report used "
                    f"{accuracy.get('precision')}, requested {settings.mixed_precision}"
                )
            if need_model:
                _ensure_sources(settings, (), auto_prepare=auto_prepare, need_model=True)
            records = [json.loads(line) for line in (destination / "train.jsonl").open("rb")]
            existing = _attach_accuracy(settings, options, records, destination, model)
        _print_summary(existing, destination, reused=True)
        return _result(existing, destination, reused=True)

    _ensure_sources(settings, specs, auto_prepare=auto_prepare, need_model=need_model)
    tokenizer = _load_label_coverage_tokenizer(settings)
    tokenizer_vocab_size = int(len(tokenizer))

    datasets_by_key: dict[str, Any] = {}
    tables: dict[str, LabelSetTable] = {}
    task_names_by_key: dict[str, list[str]] = {}
    tasks: list[TaskPool] = []
    sources: list[dict[str, Any]] = []
    reused_caches: list[str] = []
    for spec in specs:
        dataset = load_submixture(spec, settings)
        datasets_by_key[spec.key] = dataset
        pools = _task_pools(
            spec.key,
            dataset,
            max_samples_per_task=options.max_samples_per_task,
            seed=options.seed,
        )
        task_names_by_key[spec.key] = _task_codes(dataset)[1]
        table, reused = _label_table(spec, dataset, settings, tokenizer, model, options)
        if reused:
            reused_caches.append(spec.key)
        tables[spec.key] = table
        tasks.extend(pools)
        pooled = int(sum(pool.sample_indices.size for pool in pools))
        sources.append(
            {
                "submixture": spec.key,
                "dataset_id": spec.dataset_id,
                "revision": spec.revision,
                "split": spec.split,
                "rows": int(len(dataset)),
                "tasks": len(pools),
                "pooled_samples": pooled,
                "unsupervised_rows": int(np.count_nonzero(table.lengths() == 0)),
                "boundary_excluded_rows": int(table.boundary_excluded),
            }
        )
        _log(f"[{spec.key}] {len(dataset):,} rows, {len(pools):,} tasks, {pooled:,} pooled samples")
    names = [task.task_name for task in tasks]
    if len(set(names)) != len(names):
        raise RuntimeError("task names collide across submixtures; selection requires unique tasks")

    _log(
        f"Selecting with seed={options.seed} over {len(tasks):,} tasks and "
        f"{sum(int(task.sample_indices.size) for task in tasks):,} samples..."
    )
    result = select_task_vocab_cover(
        tasks,
        tables,
        seed=options.seed,
        vocab_size=tokenizer_vocab_size,
        log=_log,
        guarantee_task_coverage=options.guarantee_task_coverage,
    )
    if not result.selected:
        raise PresetCacheError("the SMART submixtures produced no shifted supervised labels")
    records = _materialize_records(
        result,
        datasets_by_key,
        {spec.key: spec for spec in specs},
        tables,
        task_names_by_key,
        tokenizer,
        max_length=settings.max_length,
    )
    manifest = write_created_anchor_artifacts(
        destination=destination,
        records=records,
        result=result,
        tables=tables,
        tokenizer=tokenizer,
        model_name=settings.model,
        model_id=model_id,
        model_revision=model_preset.revision,
        max_length=settings.max_length,
        sources=sources,
        selection={
            "seed": options.seed,
            "submixtures": list(options.submixtures),
            "max_samples_per_task": options.max_samples_per_task,
            "round_acceptances": [int(value) for value in result.round_acceptances],
            "label_caches_reused": reused_caches,
        },
        tokenizer_vocab_size=tokenizer_vocab_size,
        pooled_rows={
            spec.key: np.concatenate(
                [task.sample_indices for task in tasks if task.submixture == spec.key]
            )
            for spec in specs
        },
    )
    _log(f"Published {len(records):,} records to {destination}")
    del tables, datasets_by_key
    gc.collect()
    if options.compute_accuracy:
        manifest = _attach_accuracy(settings, options, records, destination, model)
    _print_summary(manifest, destination, reused=False)
    return _result(manifest, destination, reused=False)


def _attach_accuracy(
    settings: QuickRunSettings,
    options: AnchorCreationOptions,
    records: Sequence[Mapping[str, Any]],
    destination: Path,
    model: Mapping[str, str],
) -> dict[str, Any]:
    scores = _score_records(
        settings, records, batch_size=options.scoring_batch_size, destination=destination
    )
    summary = summarize_scores(scores)
    accuracy = {
        "metric": "teacher_forced_shifted_response_token_argmax",
        "model": dict(model),
        "precision": settings.mixed_precision,
        "batch_size": options.scoring_batch_size,
        **summary,
    }
    return attach_accuracy_report(
        destination,
        model_name=settings.model,
        model_id=model["model_id"],
        model_revision=model["revision"],
        max_length=settings.max_length,
        accuracy=accuracy,
        scores=scores,
    )


__all__ = [
    "ACTION",
    "AnchorCreationOptions",
    "create_anchor_dataset",
    "load_submixture",
    "submixture_preset",
]
