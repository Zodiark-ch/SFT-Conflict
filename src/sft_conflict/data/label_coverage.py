"""Source-order coverage of CE labels, with no model inference or quality filtering.

Schema 2 binds the model revision and the SMART-v3 CE preprocessing contract, not
model weights or predictions. Its SHA-256 signature is an integrity checksum, not
an authenticity signature. Source incidences describe the full tokenized source;
only selected incidences can be independently reconstructed from the publication.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any
from uuid import uuid4

from .anchor_filtering import (
    FILTER_SCORE_ROOT,
    FILTERED_ANCHOR_ROOT,
    TokenCoverSelection,
    supervised_output_token_ids,
)

LABEL_COVERAGE_POLICY = "label_coverage_v1"
LEGACY_FILTER_POLICY = "legacy_quality_v1"
LABEL_COVERAGE_ROOT = "label_coverage_datasets"
LABEL_COVERAGE_SCHEMA_VERSION = 2

_SIGNATURE_SCOPE = "canonical_manifest_without_signature_including_artifact_sha256"
_TOKENIZATION_SIGNATURE_SCOPE = "model_identity_and_smart_flan_v3_ce_contract"
_SELECTION_PROTOCOL = {
    "algorithm": "sequential_first_unseen_label_token",
    "order": "source_order",
    "index_scope": "zero_based_tokenized_source",
    "label_scope": "shifted_supervised_labels_including_eos",
    "causal_shift": 1,
    "ignore_index": -100,
}
_COUNT_KEYS = {
    "source_samples",
    "tokenized_samples",
    "selected_samples",
    "excluded_samples",
    "excluded_during_tokenization",
    "excluded_by_coverage",
    "covered_token_types",
    "source_token_sample_incidences",
    "selected_token_sample_incidences",
}
_MANIFEST_KEYS = {
    "schema_version",
    "policy",
    "dataset_name",
    "format",
    "model",
    "tokenization",
    "source",
    "selection",
    "counts",
    "selected_source_indices",
    "selected_sample_ids",
    "selected_original_source_indices",
    "covered_token_ids",
    "files",
    "signature",
}


def label_coverage_dataset_name(model_name: str) -> str:
    if not isinstance(model_name, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]*", model_name
    ):
        raise ValueError("model_name must be a non-empty, path-safe preset name")
    return f"smart_flan_{model_name}_{LABEL_COVERAGE_POLICY}"


def label_coverage_dataset_dir(artifact_root: str | Path, model_name: str) -> Path:
    return (
        Path(artifact_root).expanduser().resolve()
        / LABEL_COVERAGE_ROOT
        / label_coverage_dataset_name(model_name)
    )


def select_label_coverage(dataset: Iterable[Mapping[str, Any]]) -> TokenCoverSelection:
    """Keep a row iff it introduces a label type not seen in earlier kept rows.

    Only one row's token set is live at a time. Counts are sample incidences, not
    occurrences; prompts and position zero are excluded, supervised EOS is not.
    Empty rows (and an empty source) contribute nothing. No sorting of rows,
    greedy lookahead, scoring, or later removal of an earlier row is performed.
    """
    source_counts: Counter[int] = Counter()
    selected_counts: Counter[int] = Counter()
    selected: list[int] = []
    for index, record in enumerate(dataset):
        token_ids = supervised_output_token_ids(record)
        source_counts.update(token_ids)
        if any(token_id not in selected_counts for token_id in token_ids):
            selected.append(index)
            selected_counts.update(token_ids)
    return TokenCoverSelection(
        selected_indices=tuple(selected),
        universe_token_ids=tuple(sorted(source_counts)),
        eligible_sample_counts=dict(sorted(source_counts.items())),
        selected_sample_counts=dict(sorted(selected_counts.items())),
    )


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _object(value: Any, name: str, keys: set[str] | None = None) -> dict[str, Any]:
    if not isinstance(value, dict) or (keys is not None and set(value) != keys):
        raise ValueError(f"invalid label-coverage {name}")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"label-coverage {name} must be a non-empty string")
    return value


def _token_spelling(value: Any) -> str:
    # Whitespace (and even an empty spelling) can be a legitimate vocabulary
    # entry. Do not apply identifier validation to a tokenizer's actual output.
    if not isinstance(value, str):
        raise ValueError("label-coverage token spelling must be a string")
    return value


def _count(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"label-coverage {name} must be a non-negative integer")
    return value


def _ordered_indices(value: Any, name: str) -> list[int]:
    if not isinstance(value, list):
        raise ValueError(f"label-coverage {name} must be a list")
    for index in value:
        _count(index, name)
    if any(left >= right for left, right in zip(value, value[1:], strict=False)):
        raise ValueError(f"label-coverage {name} must be strictly increasing")
    return value


def _model_identity(
    model_name: str, model_id: str, model_revision: str, max_length: int
) -> dict[str, str]:
    label_coverage_dataset_name(model_name)
    _text(model_id, "model_id")
    _text(model_revision, "model_revision")
    if type(max_length) is not int or max_length <= 1:
        raise ValueError("label-coverage max_length must be an integer greater than one")
    return {"preset": model_name, "model_id": model_id, "revision": model_revision}


def _tokenization_contract(model: dict[str, str], max_length: int) -> dict[str, Any]:
    contract = {
        "adapter": "smart_flan",
        "template_version": "smart-flan-v3",
        "model_variant": "base",
        "max_length": max_length,
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
        "signature_sha256": _sha256({"model": model, "preprocessing": contract}),
    }


def _json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key in label-coverage artifact: {key}")
        result[key] = value
    return result


def _invalid_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON value in label-coverage artifact: {value}")


def _parse_json(contents: str | bytes) -> dict[str, Any]:
    return _object(
        json.loads(contents, object_pairs_hook=_json_pairs, parse_constant=_invalid_constant),
        "JSON object",
    )


def _directory(directory: str | Path) -> Path:
    try:
        path = Path(directory).expanduser().absolute()
    except (TypeError, RuntimeError) as exc:
        raise ValueError("invalid label-coverage directory") from exc
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError(f"label-coverage directory must not traverse symlinks: {path}")
    return path


def _safe_file(directory: Path, relative: Any) -> Path:
    if (
        not isinstance(relative, str)
        or not relative
        or any(character in relative for character in ("\\", ":", "\x00"))
        or any(part in {"", ".", ".."} for part in relative.split("/"))
    ):
        raise ValueError("label-coverage artifact paths must be safe relative paths")
    path = directory / relative
    if path.is_absolute() and not path.is_relative_to(directory):
        raise ValueError("label-coverage artifact path escapes its directory")
    current = directory
    for part in relative.split("/"):
        current = current / part
        if current.is_symlink():
            raise ValueError("label-coverage artifact paths must not traverse symlinks")
    return path


def _digest_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"invalid label-coverage {name} SHA-256")
    return value


def _validate_record(record: dict[str, Any], max_length: int) -> tuple[str, int | None]:
    sample_id = _text(record.get("sample_id"), "sample_id")
    fields = [record.get(key) for key in ("input_ids", "attention_mask", "labels")]
    if any(not isinstance(field, list) for field in fields):
        raise ValueError("label-coverage tokenized records require list token fields")
    if len({len(field) for field in fields}) != 1 or not 1 <= len(fields[0]) <= max_length:
        raise ValueError("label-coverage tokenized record lengths are invalid")
    for token_id in record["input_ids"]:
        _count(token_id, "input token ID")
    for token_id in record["labels"]:
        if type(token_id) is not int or (token_id < 0 and token_id != -100):
            raise ValueError("invalid label-coverage label token ID")
    if any(type(value) is not int or value not in {0, 1} for value in record["attention_mask"]):
        raise ValueError("invalid label-coverage attention mask")
    metadata = _object(record.get("source_metadata", {}), "source_metadata")
    original_index = metadata.get("source_index")
    if original_index is not None:
        _count(original_index, "original source index")
    return sample_id, original_index


def _validate_source_metadata(record: dict[str, Any], source: dict[str, Any]) -> None:
    metadata = record.get("source_metadata", {})
    for key, expected in (
        ("dataset_name", source["dataset_id"]),
        ("dataset_revision", source["revision"]),
        ("template_version", "smart-flan-v3"),
    ):
        if key in metadata and metadata[key] != expected:
            raise ValueError(f"label-coverage record {key} does not match the source contract")


def load_label_coverage_manifest(
    directory: str | Path,
    *,
    model_name: str,
    model_id: str,
    model_revision: str,
    max_length: int,
) -> dict[str, Any]:
    """Verify schema, compatibility, safe files, hashes, IDs and label incidences.

    Verification reads only this publication, never the raw source, tokenizer or
    model. Invalid/incompatible publications raise ValueError or OSError. Legacy
    manifests are deliberately not accepted by this API.
    """
    model = _model_identity(model_name, model_id, model_revision, max_length)
    root = _directory(directory)
    manifest = _parse_json(_safe_file(root, "manifest.json").read_bytes())
    _object(manifest, "manifest schema", _MANIFEST_KEYS)
    if (
        type(manifest["schema_version"]) is not int
        or manifest["schema_version"] != LABEL_COVERAGE_SCHEMA_VERSION
        or manifest["policy"] != LABEL_COVERAGE_POLICY
        or manifest["format"] != "tokenized_jsonl"
        or manifest["dataset_name"] != label_coverage_dataset_name(model_name)
        or manifest["model"] != model
        or _json_bytes(manifest["tokenization"])
        != _json_bytes(_tokenization_contract(model, max_length))
        or _json_bytes(manifest["selection"]) != _json_bytes(_SELECTION_PROTOCOL)
    ):
        raise ValueError("incompatible label-coverage policy, model or tokenization schema")
    signature = _object(manifest["signature"], "signature", {"algorithm", "scope", "sha256"})
    if (
        signature["algorithm"] != "sha256"
        or signature["scope"] != _SIGNATURE_SCOPE
        or signature["sha256"] != _sha256({k: v for k, v in manifest.items() if k != "signature"})
    ):
        raise ValueError("label-coverage manifest signature mismatch")

    source = _object(manifest["source"], "source", {"dataset_id", "revision", "fingerprint"})
    _text(source["dataset_id"], "source dataset_id")
    _text(source["fingerprint"], "source fingerprint")
    if source["revision"] is not None:
        _text(source["revision"], "source revision")
    counts = _object(manifest["counts"], "counts", _COUNT_KEYS)
    for name, value in counts.items():
        _count(value, name)
    if not (
        counts["source_samples"] >= counts["tokenized_samples"] >= counts["selected_samples"]
        and counts["excluded_samples"] == counts["source_samples"] - counts["selected_samples"]
        and counts["excluded_during_tokenization"]
        == counts["source_samples"] - counts["tokenized_samples"]
        and counts["excluded_by_coverage"]
        == counts["tokenized_samples"] - counts["selected_samples"]
    ):
        raise ValueError("inconsistent label-coverage source/selected/excluded counts")
    indices = _ordered_indices(manifest["selected_source_indices"], "selected_source_indices")
    universe = _ordered_indices(manifest["covered_token_ids"], "covered_token_ids")
    sample_ids = manifest["selected_sample_ids"]
    original_indices = manifest["selected_original_source_indices"]
    if (
        len(indices) != counts["selected_samples"]
        or (indices and indices[-1] >= counts["tokenized_samples"])
        or len(universe) != counts["covered_token_types"]
        or not isinstance(sample_ids, list)
        or not isinstance(original_indices, list)
        or len(sample_ids) != len(indices)
        or len(original_indices) != len(indices)
    ):
        raise ValueError("inconsistent label-coverage sample IDs, indices or coverage counts")
    for sample_id in sample_ids:
        _text(sample_id, "selected sample ID")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("duplicate label-coverage sample IDs")
    for original_index in original_indices:
        if original_index is not None:
            _count(original_index, "original source index")

    files = _object(
        manifest["files"],
        "files",
        {"dataset", "dataset_sha256", "output_token_dict", "output_token_dict_sha256"},
    )
    dataset_path = _safe_file(root, files["dataset"])
    tokens_path = _safe_file(root, files["output_token_dict"])
    if len({dataset_path, tokens_path, root / "manifest.json"}) != 3:
        raise ValueError("label-coverage artifact paths must be distinct")
    expected_digest = _digest_text(files["dataset_sha256"], "dataset")
    digest = hashlib.sha256()
    selected_counts: Counter[int] = Counter()
    rows = 0
    with dataset_path.open("rb") as handle:
        for line in handle:
            digest.update(line)
            record = _parse_json(line)
            if rows >= len(indices):
                raise ValueError("label-coverage dataset has too many records")
            sample_id, original_index = _validate_record(record, max_length)
            _validate_source_metadata(record, source)
            if sample_id != sample_ids[rows] or original_index != original_indices[rows]:
                raise ValueError("label-coverage dataset source/sample IDs do not match manifest")
            token_ids = supervised_output_token_ids(record)
            if not any(token_id not in selected_counts for token_id in token_ids):
                raise ValueError("label-coverage selected row introduces no new label token")
            selected_counts.update(token_ids)
            rows += 1
    if digest.hexdigest() != expected_digest:
        raise ValueError("label-coverage dataset SHA-256 mismatch")
    if (
        rows != len(indices)
        or sorted(selected_counts) != universe
        or sum(selected_counts.values()) != counts["selected_token_sample_incidences"]
    ):
        raise ValueError("label-coverage dataset counts or token coverage mismatch")

    token_bytes = tokens_path.read_bytes()
    if hashlib.sha256(token_bytes).hexdigest() != _digest_text(
        files["output_token_dict_sha256"], "output token dictionary"
    ):
        raise ValueError("label-coverage output token dictionary SHA-256 mismatch")
    token_dict = _parse_json(token_bytes)
    _object(
        token_dict, "output token dictionary", {"schema_version", "policy", "count_unit", "tokens"}
    )
    if (
        type(token_dict["schema_version"]) is not int
        or token_dict["schema_version"] != LABEL_COVERAGE_SCHEMA_VERSION
        or token_dict["policy"] != LABEL_COVERAGE_POLICY
        or token_dict["count_unit"] != "sample_incidence"
    ):
        raise ValueError("incompatible label-coverage output token dictionary")
    tokens = _object(token_dict["tokens"], "tokens", {str(token_id) for token_id in universe})
    source_incidences = 0
    for token_id in universe:
        row = _object(
            tokens[str(token_id)],
            "token incidence",
            {"token", "source_sample_count", "selected_sample_count"},
        )
        _token_spelling(row["token"])
        source_count = _count(row["source_sample_count"], "source token incidence")
        selected_count = _count(row["selected_sample_count"], "selected token incidence")
        if not (
            selected_count == selected_counts[token_id]
            and 1 <= selected_count <= source_count <= counts["tokenized_samples"]
        ):
            raise ValueError("inconsistent label-coverage token incidences")
        source_incidences += source_count
    if source_incidences != counts["source_token_sample_incidences"]:
        raise ValueError("inconsistent label-coverage total source token incidences")
    if source_incidences > counts["tokenized_samples"] * (max_length - 1):
        raise ValueError("label-coverage source token incidences exceed sequence-length bounds")
    return manifest


def _publication_directory(destination: str | Path) -> Path:
    directory = _directory(destination)
    if any(
        part in {FILTERED_ANCHOR_ROOT, FILTER_SCORE_ROOT}
        or (part.startswith("smart_flan_") and not part.endswith(f"_{LABEL_COVERAGE_POLICY}"))
        for part in directory.parts
    ):
        raise ValueError("label-coverage publication must not modify legacy Anchor paths")
    if directory.exists() and not directory.is_dir():
        raise ValueError("label-coverage publication destination is not a directory")
    manifest_path = _safe_file(directory, "manifest.json")
    previous: dict[str, Any] = {}
    if manifest_path.is_file():
        with suppress(ValueError):
            previous = _parse_json(manifest_path.read_bytes())
        if (
            previous.get("policy") == LEGACY_FILTER_POLICY
            or "filter" in previous
            or "score_files" in previous
        ):
            raise ValueError("label-coverage publication must not overwrite legacy artifacts")
    if (
        directory.is_dir()
        and any(directory.iterdir())
        and not directory.name.endswith(f"_{LABEL_COVERAGE_POLICY}")
        and previous.get("policy") != LABEL_COVERAGE_POLICY
    ):
        raise ValueError("label-coverage publication must not replace an unrelated directory")
    return directory


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


def write_label_coverage_artifacts(
    *,
    destination: str | Path,
    dataset: Sequence[Mapping[str, Any]],
    selection: TokenCoverSelection,
    tokenizer: Any,
    model_name: str,
    model_id: str,
    model_revision: str,
    max_length: int,
    source_fingerprint: str,
    source_dataset_id: str,
    source_dataset_revision: str | None,
) -> dict[str, Any]:
    """Publish unchanged selected records and a verified schema-2 manifest.

    Matching, verified publications are returned without rewriting any files.
    Replacements are staged on the same filesystem and rolled back on failure;
    legacy dataset and correctness-score directories are never publication targets.
    """
    output = _publication_directory(destination)
    model = _model_identity(model_name, model_id, model_revision, max_length)
    tokenization = _tokenization_contract(model, max_length)
    source = {
        "dataset_id": _text(source_dataset_id, "source dataset_id"),
        "revision": source_dataset_revision,
        "fingerprint": _text(source_fingerprint, "source fingerprint"),
    }
    if source_dataset_revision is not None:
        _text(source_dataset_revision, "source revision")
    for side in ("padding_side", "truncation_side"):
        if getattr(tokenizer, side, "right") != "right":
            raise ValueError(f"label-coverage SMART-v3 requires {side}=right")
    dataset_manifest = getattr(dataset, "manifest", {})
    preprocessing = dataset_manifest.get("preprocessing", {})
    for key, expected in tokenization.items():
        if key in preprocessing and preprocessing[key] != expected:
            raise ValueError(f"label-coverage source preprocessing mismatch: {key}")
    stats = getattr(dataset, "preprocessing_stats", {})
    source_samples = _count(stats.get("scanned_samples", len(dataset)), "source_samples")
    accepted = _count(stats.get("accepted_samples", len(dataset)), "accepted_samples")
    excluded = _count(
        stats.get("filtered_no_supervision", source_samples - len(dataset)),
        "excluded_during_tokenization",
    )
    if source_samples < len(dataset) or accepted != len(dataset):
        raise ValueError("inconsistent label-coverage source preprocessing counts")
    if excluded != source_samples - len(dataset):
        raise ValueError("inconsistent label-coverage excluded preprocessing count")
    indices = _ordered_indices(list(selection.selected_indices), "selected_source_indices")
    universe = _ordered_indices(list(selection.universe_token_ids), "covered_token_ids")
    if indices and indices[-1] >= len(dataset):
        raise ValueError("label-coverage selected source index is out of range")
    if set(selection.eligible_sample_counts) != set(universe) or set(
        selection.selected_sample_counts
    ) != set(universe):
        raise ValueError("label-coverage selection token universes do not match")

    digest = hashlib.sha256()
    selected_counts: Counter[int] = Counter()
    sample_ids: list[str] = []
    original_indices: list[int | None] = []
    for index in indices:
        record = dict(dataset[index])
        sample_id, original_index = _validate_record(record, max_length)
        _validate_source_metadata(record, source)
        sample_ids.append(sample_id)
        original_indices.append(original_index)
        token_ids = supervised_output_token_ids(record)
        if not any(token_id not in selected_counts for token_id in token_ids):
            raise ValueError("label-coverage selected row introduces no new label token")
        selected_counts.update(token_ids)
        digest.update(_json_bytes(record) + b"\n")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("duplicate label-coverage selected sample IDs")
    if (
        dict(selected_counts) != selection.selected_sample_counts
        or sorted(selected_counts) != universe
    ):
        raise ValueError("label-coverage selection does not match selected record labels")
    token_rows: dict[str, Any] = {}
    for token_id in universe:
        source_count = _count(selection.eligible_sample_counts[token_id], "source token incidence")
        if not selected_counts[token_id] <= source_count <= len(dataset):
            raise ValueError("invalid label-coverage source token incidence")
        token_rows[str(token_id)] = {
            "token": _token_spelling(tokenizer.convert_ids_to_tokens(token_id)),
            "source_sample_count": source_count,
            "selected_sample_count": selected_counts[token_id],
        }
    token_bytes = (
        _json_bytes(
            {
                "schema_version": LABEL_COVERAGE_SCHEMA_VERSION,
                "policy": LABEL_COVERAGE_POLICY,
                "count_unit": "sample_incidence",
                "tokens": token_rows,
            }
        )
        + b"\n"
    )
    manifest = {
        "schema_version": LABEL_COVERAGE_SCHEMA_VERSION,
        "policy": LABEL_COVERAGE_POLICY,
        "dataset_name": label_coverage_dataset_name(model_name),
        "format": "tokenized_jsonl",
        "model": model,
        "tokenization": tokenization,
        "source": source,
        "selection": dict(_SELECTION_PROTOCOL),
        "counts": {
            "source_samples": source_samples,
            "tokenized_samples": len(dataset),
            "selected_samples": len(indices),
            "excluded_samples": source_samples - len(indices),
            "excluded_during_tokenization": source_samples - len(dataset),
            "excluded_by_coverage": len(dataset) - len(indices),
            "covered_token_types": len(universe),
            "source_token_sample_incidences": sum(selection.eligible_sample_counts.values()),
            "selected_token_sample_incidences": sum(selected_counts.values()),
        },
        "selected_source_indices": indices,
        "selected_sample_ids": sample_ids,
        "selected_original_source_indices": original_indices,
        "covered_token_ids": universe,
        "files": {
            "dataset": "train.jsonl",
            "dataset_sha256": digest.hexdigest(),
            "output_token_dict": "output-token-dict.json",
            "output_token_dict_sha256": hashlib.sha256(token_bytes).hexdigest(),
        },
    }
    manifest["signature"] = {
        "algorithm": "sha256",
        "scope": _SIGNATURE_SCOPE,
        "sha256": _sha256(manifest),
    }
    compatibility = {
        "model_name": model_name,
        "model_id": model_id,
        "model_revision": model_revision,
        "max_length": max_length,
    }
    if (output / "manifest.json").is_file():
        try:
            existing = load_label_coverage_manifest(output, **compatibility)
        except (ValueError, OSError):
            pass
        else:
            if existing == manifest:
                return existing

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        with (temporary / "train.jsonl").open("wb") as handle:
            for index in indices:
                handle.write(_json_bytes(dict(dataset[index])) + b"\n")
        (temporary / "output-token-dict.json").write_bytes(token_bytes)
        (temporary / "manifest.json").write_bytes(_json_bytes(manifest) + b"\n")
        load_label_coverage_manifest(temporary, **compatibility)
        _publication_directory(output)
        _publish_directory(temporary, output)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    return manifest
