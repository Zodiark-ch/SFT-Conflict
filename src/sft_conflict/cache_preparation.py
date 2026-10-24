"""Prepare fixed-revision preset caches before strict offline training starts."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from datasets import DownloadConfig, load_dataset
from huggingface_hub import snapshot_download, try_to_load_from_cache
from transformers import AutoTokenizer

from sft_conflict.config import DatasetConfig, MethodName
from sft_conflict.data.anchor_filtering import filtered_anchor_dataset_dir
from sft_conflict.data.label_coverage import (
    LABEL_COVERAGE_POLICY,
    LEGACY_FILTER_POLICY,
    label_coverage_dataset_dir,
    label_coverage_dataset_name,
    load_label_coverage_manifest,
    select_label_coverage,
    write_label_coverage_artifacts,
)
from sft_conflict.data.loading import load_tokenized_dataset
from sft_conflict.quick_run import (
    ANCHOR_DATASET_PRESETS,
    FILTERED_ANCHOR_DATASETS,
    MODEL_PRESETS,
    TARGET_DATASET_PRESETS,
    DatasetPreset,
    QuickRunSettings,
    target_evaluation_preset,
)

# Deliberately not "*.json" or an unrestricted model snapshot: even weight index
# files are unnecessary for CPU-only label coverage. config.json permits the
# AutoTokenizer fallback for repositories without a tokenizer_class entry.
TOKENIZER_ALLOW_PATTERNS = (
    "config.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "vocab.txt",
    "merges.txt",
    "spiece.model",
    "sentencepiece.bpe.model",
    "chat_template.jinja",
)


class PresetCacheError(RuntimeError):
    """Raised when a selected real preset is not available in the fixed cache."""


def _normalized_cache_identifier(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


def _dataset_repository_directories(
    dataset_cache: Path,
    dataset_id: str,
) -> tuple[Path, ...]:
    expected_name = dataset_id.replace("/", "___")
    exact = dataset_cache / expected_name
    if exact.is_dir():
        return (exact,)
    normalized = _normalized_cache_identifier(expected_name)
    return tuple(
        path
        for path in dataset_cache.iterdir()
        if path.is_dir() and _normalized_cache_identifier(path.name) == normalized
    )


def _split_arrow_cache_ready(
    cache_directory: Path,
    split: str,
    split_information: dict,
) -> bool:
    unsharded = list(cache_directory.glob(f"*-{split}.arrow"))
    shard_lengths = split_information.get("shard_lengths")
    if not shard_lengths:
        return len(unsharded) == 1

    expected_count = len(shard_lengths)
    pattern = re.compile(rf"^.+-{re.escape(split)}-(\d{{5}})-of-(\d{{5}})\.arrow$")
    shard_numbers: set[int] = set()
    for path in cache_directory.glob(f"*-{split}-*-of-*.arrow"):
        match = pattern.fullmatch(path.name)
        if match is None or int(match.group(2)) != expected_count:
            continue
        shard_numbers.add(int(match.group(1)))
    return shard_numbers == set(range(expected_count))


def _cache_paths(artifact_root: str | Path) -> tuple[Path, Path]:
    root = Path(artifact_root).expanduser().resolve() / "hf_cache"
    return root / "hub", root / "datasets"


def _model_cache_ready(settings: QuickRunSettings) -> bool:
    preset = MODEL_PRESETS[settings.model]
    if preset.model_id is None:
        return True
    hub_cache, _ = _cache_paths(settings.artifact_root)
    config_path = try_to_load_from_cache(
        preset.model_id,
        "config.json",
        cache_dir=hub_cache,
        revision=preset.revision,
    )
    if not isinstance(config_path, str):
        return False
    snapshot = Path(config_path).parent
    index_path = snapshot / "model.safetensors.index.json"
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
        return all((snapshot / filename).is_file() for filename in set(weight_map.values()))
    return (snapshot / "model.safetensors").is_file() or (snapshot / "pytorch_model.bin").is_file()


def _label_coverage_model_id(settings: QuickRunSettings) -> str:
    preset = MODEL_PRESETS[settings.model]
    if preset.model_id is not None:
        return preset.model_id
    return str(
        Path(settings.artifact_root).expanduser().resolve()
        / "tmp"
        / "click-run"
        / "assets-v3"
        / settings.model
    )


def _tokenizer_cache_ready(settings: QuickRunSettings) -> bool:
    preset = MODEL_PRESETS[settings.model]
    if preset.model_id is None:
        # The existing Tiny vocabulary can construct its tokenizer in memory;
        # _ensure_tiny_model would also create weights and must not be used here.
        return True
    hub_cache, _ = _cache_paths(settings.artifact_root)

    def cached(filename: str) -> bool:
        path = try_to_load_from_cache(
            preset.model_id, filename, cache_dir=hub_cache, revision=preset.revision
        )
        return isinstance(path, str) and Path(path).is_file()

    return cached("tokenizer_config.json") and (
        cached("tokenizer.json")
        or cached("tokenizer.model")
        or cached("spiece.model")
        or cached("sentencepiece.bpe.model")
        or (cached("vocab.json") and cached("merges.txt"))
        or cached("vocab.txt")
    )


def _load_label_coverage_tokenizer(settings: QuickRunSettings) -> Any:
    preset = MODEL_PRESETS[settings.model]
    model_id = _label_coverage_model_id(settings)
    if preset.model_id is None and not (Path(model_id) / "tokenizer.json").is_file():
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Sequence, Split, WhitespaceSplit
        from transformers import PreTrainedTokenizerFast

        from sft_conflict.quick_run import _vocabulary

        backend = Tokenizer(WordLevel(_vocabulary(), unk_token="<unk>"))
        backend.pre_tokenizer = Sequence([WhitespaceSplit(), Split(".", behavior="isolated")])
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend,
            unk_token="<unk>",
            pad_token="<pad>",
            eos_token="<eos>",
            model_max_length=128,
        )
    else:
        hub_cache, _ = _cache_paths(settings.artifact_root)
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=preset.revision,
            cache_dir=str(hub_cache),
            local_files_only=True,
            trust_remote_code=False,
            use_fast=True,
        )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise PresetCacheError("label coverage requires a tokenizer with a pad or EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    return tokenizer


def _dataset_cache_ready(preset: DatasetPreset | None, artifact_root: str | Path) -> bool:
    if preset is None or preset.path is None:
        return True
    hub_cache, dataset_cache = _cache_paths(artifact_root)
    data_files = {
        value
        for value in (
            preset.train_data_file,
            preset.validation_data_file,
            preset.eval_data_file,
        )
        if value is not None
    }
    if data_files:
        return all(
            isinstance(
                try_to_load_from_cache(
                    preset.path,
                    data_file,
                    cache_dir=hub_cache,
                    revision=preset.revision,
                    repo_type="dataset",
                ),
                str,
            )
            for data_file in data_files
        )
    if not dataset_cache.is_dir() or preset.revision is None:
        return False
    required_splits = {
        split
        for split in (preset.train_split, preset.validation_split, preset.eval_split)
        if split is not None
    }
    for repository_directory in _dataset_repository_directories(dataset_cache, preset.path):
        configuration_directory = repository_directory / (preset.name or "default")
        for information_path in configuration_directory.rglob("dataset_info.json"):
            if preset.revision not in information_path.parts:
                continue
            try:
                information = json.loads(information_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if preset.name is not None and information.get("config_name") != preset.name:
                continue
            cached_splits = set(information.get("splits", {}))
            if not required_splits.issubset(cached_splits):
                continue
            if all(
                _split_arrow_cache_ready(
                    information_path.parent,
                    split,
                    information["splits"][split],
                )
                for split in required_splits
            ):
                return True
    return False


def _online_environment(artifact_root: str | Path) -> dict[str, str]:
    environment = os.environ.copy()
    for key in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
        environment.pop(key, None)

    proxy_keys = (
        "http_proxy",
        "https_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "all_proxy",
    )
    for key in proxy_keys:
        value = environment.get(key)
        if value and urlparse(value).hostname in {"127.0.0.1", "localhost"}:
            environment.pop(key, None)

    hub_cache, dataset_cache = _cache_paths(artifact_root)
    environment["HF_HOME"] = str(hub_cache.parent)
    environment["HF_HUB_CACHE"] = str(hub_cache)
    environment["HF_DATASETS_CACHE"] = str(dataset_cache)
    return environment


def _prepare_model(settings: QuickRunSettings) -> None:
    preset = MODEL_PRESETS[settings.model]
    if preset.model_id is None:
        return
    command = [
        sys.executable,
        "-m",
        "sft_conflict.cache_preparation",
        "model",
        "--repo-id",
        preset.model_id,
        "--revision",
        preset.revision,
        "--artifact-root",
        settings.artifact_root,
    ]
    subprocess.run(
        command,
        check=True,
        env=_online_environment(settings.artifact_root),
    )


def _prepare_tokenizer(settings: QuickRunSettings) -> None:
    preset = MODEL_PRESETS[settings.model]
    if preset.model_id is None:
        return
    environment = _online_environment(settings.artifact_root)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    subprocess.run(
        [
            sys.executable,
            "-m",
            "sft_conflict.cache_preparation",
            "tokenizer",
            "--repo-id",
            preset.model_id,
            "--revision",
            preset.revision,
            "--artifact-root",
            str(settings.artifact_root),
        ],
        check=True,
        env=environment,
    )


def _prepare_dataset(preset: DatasetPreset, settings: QuickRunSettings) -> None:
    if preset.path is None or preset.revision is None:
        return
    splits = sorted(
        split
        for split in (preset.train_split, preset.validation_split, preset.eval_split)
        if split is not None
    )
    command = [
        sys.executable,
        "-m",
        "sft_conflict.cache_preparation",
        "dataset",
        "--repo-id",
        preset.path,
        "--revision",
        preset.revision,
        "--splits",
        *splits,
        "--artifact-root",
        settings.artifact_root,
    ]
    if preset.name is not None:
        command.extend(["--name", preset.name])
    data_files = sorted(
        value
        for value in (
            preset.train_data_file,
            preset.validation_data_file,
            preset.eval_data_file,
        )
        if value is not None
    )
    if data_files:
        command.extend(["--data-files", *data_files])
    try:
        subprocess.run(
            command,
            check=True,
            env=_online_environment(settings.artifact_root),
        )
    except subprocess.CalledProcessError as exc:
        if preset.path == "BAAI/Infinity-Instruct":
            raise PresetCacheError(
                "BAAI/Infinity-Instruct is gated. Accept its terms on Hugging Face and "
                "authenticate this machine with `huggingface-cli login`, then rerun with "
                "AUTO_PREPARE_CACHE=True. Do not place a token in run.py."
            ) from exc
        raise


def _label_coverage_source_preset() -> DatasetPreset:
    preset = ANCHOR_DATASET_PRESETS["smart_flan"]
    if preset is None or preset.path is None or preset.revision is None:
        raise PresetCacheError("label coverage requires the fixed-revision raw SMART preset")
    return preset


def _ensure_label_coverage_source_cache(settings: QuickRunSettings, *, auto_prepare: bool) -> None:
    preset = _label_coverage_source_preset()
    missing_tokenizer = not _tokenizer_cache_ready(settings)
    missing_source = not _dataset_cache_ready(preset, settings.artifact_root)
    if not (missing_tokenizer or missing_source):
        return
    missing = []
    if missing_tokenizer:
        missing.append(f"tokenizer={settings.model}")
    if missing_source:
        missing.append("raw anchor_dataset=smart_flan")
    if not auto_prepare:
        raise PresetCacheError(
            "Label-coverage CPU cache is incomplete: "
            + ", ".join(missing)
            + ". Set AUTO_PREPARE_CACHE=True to prepare the fixed tokenizer and raw SMART "
            "dataset; model weights and GPUs are not required."
        )
    print("Preparing label-coverage CPU cache: " + ", ".join(missing), flush=True)
    if missing_tokenizer:
        _prepare_tokenizer(settings)
        if not _tokenizer_cache_ready(settings):
            raise PresetCacheError(
                f"tokenizer cache preparation did not complete: {settings.model}"
            )
    if missing_source:
        _prepare_dataset(preset, settings)
        if not _dataset_cache_ready(preset, settings.artifact_root):
            raise PresetCacheError("raw SMART dataset cache preparation did not complete")


def _label_coverage_result(
    directory: Path, manifest: dict[str, Any], *, reused: bool
) -> dict[str, Any]:
    counts = manifest["counts"]
    return {
        "action": "filtering-anchor",
        "policy": LABEL_COVERAGE_POLICY,
        "output_dataset": manifest["dataset_name"],
        "output_dir": str(directory),
        "manifest": str(directory / "manifest.json"),
        "source_samples": counts["source_samples"],
        "tokenized_source_samples": counts["tokenized_samples"],
        "selected_samples": counts["selected_samples"],
        "covered_token_types": counts["covered_token_types"],
        "reused_existing_dataset": reused,
    }


def prepare_label_coverage_dataset(
    settings: QuickRunSettings, *, auto_prepare: bool
) -> dict[str, Any]:
    """Create/reuse a full-SMART label cover in the CPU parent, without weights.

    A verified publication is sufficient for reuse, even when source/tokenizer
    caches are no longer present. Automatic training preparation and replacements
    require auto_prepare=True. An explicit filtering action may create a missing
    publication offline from ready caches. Workers may only reuse a publication.
    """
    settings.validate()
    expected_name = label_coverage_dataset_name(settings.model)
    if settings.anchor_dataset not in {"smart_flan", expected_name}:
        raise ValueError(f"label coverage requires smart_flan or {expected_name}")
    if (
        settings.action == "filtering-anchor"
        and getattr(settings, "filtering_policy", LABEL_COVERAGE_POLICY) != LABEL_COVERAGE_POLICY
    ):
        raise ValueError(f"prepare_label_coverage_dataset requires {LABEL_COVERAGE_POLICY}")
    preset = _label_coverage_source_preset()
    assert preset.path is not None  # The fixed-source helper rejects incomplete presets.
    model_preset = MODEL_PRESETS[settings.model]
    model_id = _label_coverage_model_id(settings)
    destination = label_coverage_dataset_dir(settings.artifact_root, settings.model)
    existing_publication = destination.exists() or destination.is_symlink()
    try:
        manifest = load_label_coverage_manifest(
            destination,
            model_name=settings.model,
            model_id=model_id,
            model_revision=model_preset.revision,
            max_length=settings.max_length,
        )
        if (
            manifest["source"]["dataset_id"] != preset.path
            or manifest["source"]["revision"] != preset.revision
        ):
            raise ValueError("label-coverage source is not the fixed raw SMART revision")
        if manifest["counts"]["selected_samples"] == 0:
            raise ValueError("label-coverage dataset has no shifted supervised labels")
    except (ValueError, OSError) as exc:
        if not auto_prepare and (settings.action != "filtering-anchor" or existing_publication):
            raise PresetCacheError(
                f"Label-coverage Anchor dataset is missing or incompatible: {destination}. "
                f"Set AUTO_PREPARE_CACHE=True, or run METHOD='filtering_anchor' with "
                f"FILTERING_POLICY='{LABEL_COVERAGE_POLICY}' and ANCHOR_DATASET='smart_flan' "
                "to prepare it once on CPU."
            ) from exc
    else:
        return _label_coverage_result(destination, manifest, reused=True)

    # Do not initialize/query torch.distributed (or CUDA) for a preprocessing job.
    # Workers must reuse the artifact already published by their launcher parent.
    if (
        "RANK" in os.environ
        or "LOCAL_RANK" in os.environ
        or os.environ.get("WORLD_SIZE", "1") != "1"
    ):
        raise PresetCacheError(
            "Prepare the label-coverage dataset once in the CPU parent before distributed launch"
        )
    _ensure_label_coverage_source_cache(settings, auto_prepare=auto_prepare)
    tokenizer = _load_label_coverage_tokenizer(settings)
    _, dataset_cache = _cache_paths(settings.artifact_root)
    dataset = load_tokenized_dataset(
        DatasetConfig(
            source="hf",
            path=preset.path,
            revision=preset.revision,
            name=preset.name,
            split=preset.train_split,
            data_file=preset.train_data_file,
            adapter=preset.adapter,
            selection="all",
            max_samples=None,
        ),
        tokenizer,
        max_length=settings.max_length,
        model_variant="base",
        cache_dir=dataset_cache,
        progress_description="Anchor label-coverage source",
    )
    selection = select_label_coverage(dataset)
    if not selection.selected_indices:
        raise PresetCacheError("raw SMART has no shifted supervised labels for label coverage")
    manifest = write_label_coverage_artifacts(
        destination=destination,
        dataset=dataset,
        selection=selection,
        tokenizer=tokenizer,
        model_name=settings.model,
        model_id=model_id,
        model_revision=model_preset.revision,
        max_length=settings.max_length,
        source_fingerprint=dataset.fingerprint,
        source_dataset_id=preset.path,
        source_dataset_revision=preset.revision,
    )
    return _label_coverage_result(destination, manifest, reused=False)


def ensure_preset_cache(settings: QuickRunSettings, *, auto_prepare: bool) -> None:
    settings.validate()
    if settings.action == "record-s0":
        # Recording uses only the fixed Unembedding. Never touch Target/Anchor
        # caches or build label coverage as an accidental preprocessing step.
        if not _model_cache_ready(settings):
            if not auto_prepare:
                raise PresetCacheError("record_s0 requires the fixed model snapshot in cache")
            _prepare_model(settings)
            if not _model_cache_ready(settings):
                raise PresetCacheError("record_s0 model cache preparation did not complete")
        return
    if settings.action == "filtering-anchor":
        policy = getattr(settings, "filtering_policy", LABEL_COVERAGE_POLICY)
        if policy == LABEL_COVERAGE_POLICY:
            _ensure_label_coverage_source_cache(settings, auto_prepare=auto_prepare)
            return
        if policy != LEGACY_FILTER_POLICY:
            raise ValueError(f"unsupported filtering_policy: {policy}")
    label_coverage_anchor = settings.anchor_dataset in {
        label_coverage_dataset_name(model_name) for model_name in MODEL_PRESETS
    }
    if label_coverage_anchor:
        # The parent runs this before launching workers. Subsequent calls verify
        # the manifest and selected records, but never retokenize the full source.
        prepare_label_coverage_dataset(settings, auto_prepare=auto_prepare)
    lvd_plus_preparation = MethodName(settings.method).is_lvd_plus_preparation
    target_preset = TARGET_DATASET_PRESETS[settings.target_dataset]
    if lvd_plus_preparation:
        # Calibration uses the unchanged Target training selection plus Anchor.
        # Do not require a final test or internal-validation split just because
        # the general training preset also advertises those evaluation sources.
        target_preset = replace(
            target_preset,
            eval_split=target_preset.train_split,
            eval_data_file=target_preset.train_data_file,
            validation_split=None,
            validation_data_file=None,
        )
    evaluation_preset = (
        target_preset if lvd_plus_preparation else target_evaluation_preset(settings.target_dataset)
    )
    anchor_preset = ANCHOR_DATASET_PRESETS.get(settings.anchor_dataset)
    missing_model = not _model_cache_ready(settings)
    anchor_only = MethodName(settings.method) in {
        MethodName.PREPARE_ANCHOR_GEOMETRY_REFERENCE,
        MethodName.FILTERING_ANCHOR,
        MethodName.RECORD_ANCHOR_D0,
        MethodName.TEST_INTERFERENCE,
        MethodName.TEST_CROSSING,
    }
    missing_target = (
        False if anchor_only else not _dataset_cache_ready(target_preset, settings.artifact_root)
    )
    missing_evaluation = (
        False
        if anchor_only or lvd_plus_preparation
        else not _dataset_cache_ready(evaluation_preset, settings.artifact_root)
    )
    filtered_anchor = settings.anchor_dataset in FILTERED_ANCHOR_DATASETS
    filtered_anchor_dir = filtered_anchor_dataset_dir(settings.artifact_root, settings.model)
    if label_coverage_anchor:
        missing_anchor = False
    elif filtered_anchor:
        missing_anchor = (
            not (filtered_anchor_dir / "train.jsonl").is_file()
            or not (filtered_anchor_dir / "manifest.json").is_file()
        )
    else:
        missing_anchor = not _dataset_cache_ready(anchor_preset, settings.artifact_root)
    if not any((missing_model, missing_target, missing_evaluation, missing_anchor)):
        return

    if missing_anchor and filtered_anchor:
        raise PresetCacheError(
            f"Filtered Anchor dataset is missing: {filtered_anchor_dir}. Run "
            "METHOD='filtering_anchor' with ANCHOR_DATASET='smart_flan' first."
        )

    missing = []
    if missing_model:
        missing.append(f"model={settings.model}")
    if missing_target:
        missing.append(f"target_dataset={settings.target_dataset}")
    if missing_evaluation and evaluation_preset is not target_preset:
        missing.append(f"target_evaluation={evaluation_preset.adapter}")
    if missing_anchor:
        missing.append(f"anchor_dataset={settings.anchor_dataset}")
    if not auto_prepare:
        hub_cache, dataset_cache = _cache_paths(settings.artifact_root)
        raise PresetCacheError(
            "Selected preset cache is incomplete: "
            + ", ".join(missing)
            + f". Model cache: {hub_cache}; dataset cache: {dataset_cache}. "
            "Set AUTO_PREPARE_CACHE=True in run.py to prepare fixed revisions."
        )

    print("Preparing missing fixed-revision cache: " + ", ".join(missing), flush=True)
    if missing_model:
        _prepare_model(settings)
    if missing_target:
        _prepare_dataset(target_preset, settings)
    if missing_evaluation and evaluation_preset is not target_preset:
        _prepare_dataset(evaluation_preset, settings)
    if missing_anchor and anchor_preset is not None:
        _prepare_dataset(anchor_preset, settings)

    if not _model_cache_ready(settings):
        raise PresetCacheError(f"model cache preparation did not complete: {settings.model}")
    if not anchor_only:
        if not _dataset_cache_ready(target_preset, settings.artifact_root):
            raise PresetCacheError(
                f"target dataset cache preparation did not complete: {settings.target_dataset}"
            )
        if not lvd_plus_preparation and not _dataset_cache_ready(
            evaluation_preset, settings.artifact_root
        ):
            raise PresetCacheError(
                "target evaluation cache preparation did not complete: "
                f"{evaluation_preset.path}/{evaluation_preset.name or 'default'}"
            )
    if not _dataset_cache_ready(anchor_preset, settings.artifact_root):
        raise PresetCacheError(
            f"anchor dataset cache preparation did not complete: {settings.anchor_dataset}"
        )


def prepare_preset_derived_cache(settings: QuickRunSettings) -> None:
    settings.validate()
    if settings.action == "record-s0" or settings.target_dataset != "infinity_instruct_code":
        return
    preset = TARGET_DATASET_PRESETS[settings.target_dataset]
    if preset.path is None or preset.revision is None:
        raise PresetCacheError("Infinity-Instruct preset is incomplete")
    _, dataset_cache = _cache_paths(settings.artifact_root)
    from sft_conflict.data.code import is_infinity_code_row

    dataset = load_dataset(
        preset.path,
        preset.name,
        split=preset.train_split,
        revision=preset.revision,
        cache_dir=dataset_cache,
        download_config=DownloadConfig(local_files_only=True),
    )
    filtered = dataset.filter(
        is_infinity_code_row,
        desc="Preparing Infinity-Instruct code filter cache once",
    )
    required_samples = settings.target_max_samples or 20_000
    if len(filtered) < required_samples:
        raise PresetCacheError(
            f"Infinity-Instruct code filter produced {len(filtered):,} rows, fewer than "
            f"the requested {required_samples:,}"
        )
    print(
        f"Infinity-Instruct derived code cache ready: {len(filtered):,} rows",
        flush=True,
    )


def _prepare_model_command(args: argparse.Namespace) -> None:
    hub_cache, _ = _cache_paths(args.artifact_root)
    snapshot = snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision,
        cache_dir=hub_cache,
    )
    print(f"Model cache ready: {snapshot}")


def _prepare_tokenizer_command(args: argparse.Namespace) -> None:
    hub_cache, _ = _cache_paths(args.artifact_root)
    snapshot = snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision,
        cache_dir=hub_cache,
        allow_patterns=list(TOKENIZER_ALLOW_PATTERNS),
    )
    print(f"Tokenizer-only cache ready: {snapshot}")


def _prepare_dataset_command(args: argparse.Namespace) -> None:
    hub_cache, dataset_cache = _cache_paths(args.artifact_root)
    if args.data_files:
        snapshot = snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            revision=args.revision,
            cache_dir=hub_cache,
            allow_patterns=args.data_files,
        )
        for data_file in args.data_files:
            path = Path(snapshot) / data_file
            if not path.is_file():
                raise PresetCacheError(
                    f"dataset repository file was not downloaded: {args.repo_id}/{data_file}"
                )
            print(f"Dataset source file cache ready: {path}")
        return
    for split in args.splits:
        dataset = load_dataset(
            args.repo_id,
            args.name,
            split=split,
            revision=args.revision,
            cache_dir=dataset_cache,
        )
        print(
            f"Dataset cache ready: {args.repo_id}/{args.name or 'default'} {split} {len(dataset)}"
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    model_parser = commands.add_parser("model")
    model_parser.add_argument("--repo-id", required=True)
    model_parser.add_argument("--revision", required=True)
    model_parser.add_argument("--artifact-root", required=True)
    tokenizer_parser = commands.add_parser("tokenizer")
    tokenizer_parser.add_argument("--repo-id", required=True)
    tokenizer_parser.add_argument("--revision", required=True)
    tokenizer_parser.add_argument("--artifact-root", required=True)
    dataset_parser = commands.add_parser("dataset")
    dataset_parser.add_argument("--repo-id", required=True)
    dataset_parser.add_argument("--revision", required=True)
    dataset_parser.add_argument("--name")
    dataset_parser.add_argument("--data-files", nargs="*")
    dataset_parser.add_argument("--splits", nargs="+", required=True)
    dataset_parser.add_argument("--artifact-root", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "model":
        _prepare_model_command(args)
    elif args.command == "tokenizer":
        _prepare_tokenizer_command(args)
    else:
        _prepare_dataset_command(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
