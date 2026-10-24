"""Resumable Safetensors cache for selected reference-model Anchor activations."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from sft_conflict.models import UnembeddingInputCapture, get_unembedding

from .anchor_positions import AnchorPositionSelector
from .collator import CausalLMCollator

REFERENCE_CACHE_SCHEMA_VERSION = 5


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


class ReferenceFilterMode(str, Enum):
    ALL_REFERENCE = "all_reference"
    BASE_CORRECT = "base_correct"
    MIN_CONFIDENCE = "minimum_confidence"
    MIN_MARGIN = "minimum_margin"


@dataclass(frozen=True)
class ReferencePrecisionBudget:
    hidden_max_absolute_error: float
    hidden_mean_absolute_error: float
    minimum_argmax_agreement: float
    margin_max_absolute_error: float
    margin_max_relative_error: float
    maximum_margin_sign_flip_rate: float

    def validate(self) -> None:
        values = asdict(self)
        if any(value < 0 for value in values.values()):
            raise ValueError("reference precision thresholds must be non-negative")
        for name in (
            "minimum_argmax_agreement",
            "maximum_margin_sign_flip_rate",
        ):
            if values[name] > 1:
                raise ValueError(f"{name} must be at most one")


@dataclass(frozen=True)
class ReferenceCacheConfig:
    output_dir: str
    samples_per_shard: int = 32
    storage_dtype: str = "float32"
    filter_mode: ReferenceFilterMode = ReferenceFilterMode.BASE_CORRECT
    minimum_confidence: float | None = None
    minimum_margin: float | None = None
    precision_budget: ReferencePrecisionBudget | None = None
    normal_epsilon: float = 1e-12

    def validate(self) -> None:
        if self.normal_epsilon <= 0:
            raise ValueError("normal_epsilon must be positive")
        if self.samples_per_shard <= 0:
            raise ValueError("samples_per_shard must be positive")
        if self.storage_dtype not in {"float32", "bfloat16", "float16"}:
            raise ValueError("storage_dtype must be float32, bfloat16, or float16")
        mode = ReferenceFilterMode(self.filter_mode)
        if mode is ReferenceFilterMode.MIN_CONFIDENCE and (
            self.minimum_confidence is None or not 0 <= self.minimum_confidence <= 1
        ):
            raise ValueError("minimum_confidence filtering requires a threshold in [0, 1]")
        if mode is ReferenceFilterMode.MIN_MARGIN and (
            self.minimum_margin is None or self.minimum_margin < 0
        ):
            raise ValueError("minimum_margin filtering requires a non-negative threshold")
        if self.storage_dtype != "float32" and self.precision_budget is None:
            raise ValueError("compressed reference caches require a precision_budget")
        if self.precision_budget is not None:
            self.precision_budget.validate()


def estimate_reference_cache_bytes(
    *,
    sample_count: int,
    positions_per_sample: int,
    hidden_size: int,
    vocabulary_size: int,
    storage_dtype: str,
) -> int:
    bytes_per_float = {"float32": 4, "bfloat16": 2, "float16": 2}[storage_dtype]
    positions = sample_count * positions_per_sample
    floating_values = hidden_size + vocabulary_size + 2
    integer_values = 3
    return positions * (floating_values * bytes_per_float + integer_values * 8 + 1)


class ReferenceCacheBuilder:
    def __init__(
        self,
        *,
        model,
        tokenizer,
        dataset,
        selector: AnchorPositionSelector,
        config: ReferenceCacheConfig,
        model_id: str,
        model_revision: str,
        dataset_config: dict[str, Any],
        device: torch.device | str | None = None,
    ) -> None:
        config.validate()
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.selector = selector
        self.config = config
        self.model_id = model_id
        self.model_revision = model_revision
        self.dataset_config = dataset_config
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.output_dir = Path(config.output_dir).expanduser().resolve()
        self.manifest_path = self.output_dir / "manifest.json"

    @property
    def _storage_dtype(self) -> torch.dtype:
        return {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[self.config.storage_dtype]

    def _identity(self) -> dict[str, Any]:
        return {
            "schema_version": REFERENCE_CACHE_SCHEMA_VERSION,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "vocabulary_size": int(get_unembedding(self.model).weight.shape[0]),
            "tokenizer_class": type(self.tokenizer).__name__,
            "dataset_fingerprint": self.dataset.fingerprint,
            "dataset_config": self.dataset_config,
            "position_config": self.selector.config.to_dict(),
            "storage_dtype": self.config.storage_dtype,
            "filter_mode": ReferenceFilterMode(self.config.filter_mode).value,
            "minimum_confidence": self.config.minimum_confidence,
            "minimum_margin": self.config.minimum_margin,
            "precision_budget": (
                asdict(self.config.precision_budget)
                if self.config.precision_budget is not None
                else None
            ),
            "normal_epsilon": self.config.normal_epsilon,
        }

    def _new_manifest(self) -> dict[str, Any]:
        return {
            **self._identity(),
            "status": "building",
            "sample_count": len(self.dataset),
            "position_count": 0,
            "shards": [],
        }

    def _load_manifest(self) -> dict[str, Any]:
        if not self.manifest_path.is_file():
            return self._new_manifest()
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        identity = self._identity()
        mismatches = {
            key: (manifest.get(key), value)
            for key, value in identity.items()
            if manifest.get(key) != value
        }
        if mismatches:
            raise ValueError(f"reference cache identity mismatch: {mismatches}")
        return manifest

    def _collate_chunk(self, start: int, end: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        examples: list[dict[str, Any]] = []
        source_records: list[dict[str, Any]] = []
        for index in range(start, end):
            source = dict(self.dataset[index])
            selection = self.selector.select(source)
            selected_positions = list(selection.positions)
            selected_target_ids = list(selection.target_token_ids)
            if "selected_anchor_positions" in source and (
                source["selected_anchor_positions"] != selected_positions
                or source.get("selected_anchor_target_ids") != selected_target_ids
            ):
                raise ValueError("positioned Anchor dataset disagrees with cache selector")
            source["selected_anchor_positions"] = selected_positions
            source["selected_anchor_target_ids"] = selected_target_ids
            examples.append(source)
            source_records.append(source)
        collator = CausalLMCollator(
            self.tokenizer.pad_token_id,
            pad_to_multiple_of=None,
        )
        return collator(examples), source_records

    @torch.no_grad()
    def _all_token_margins(
        self,
        logits: torch.Tensor,
        reference_token_ids: torch.Tensor,
        head_weight: torch.Tensor,
    ) -> torch.Tensor:
        row_count, vocabulary_size = logits.shape
        hidden_size = head_weight.shape[1]
        chunk_size = max(1, min(vocabulary_size, (64 * 1024**2) // (4 * hidden_size)))
        margins = torch.empty_like(logits, dtype=torch.float32)
        reference_logits = logits.gather(1, reference_token_ids.unsqueeze(1))
        for row_index in range(row_count):
            reference_id = int(reference_token_ids[row_index].item())
            reference_weight = head_weight[reference_id]
            for start in range(0, vocabulary_size, chunk_size):
                end = min(start + chunk_size, vocabulary_size)
                normal_norms = torch.linalg.vector_norm(
                    reference_weight.unsqueeze(0) - head_weight[start:end],
                    dim=-1,
                )
                token_ids = torch.arange(start, end, device=logits.device)
                is_reference = token_ids.eq(reference_id)
                if bool((normal_norms.le(self.config.normal_epsilon) & ~is_reference).any()):
                    raise ValueError("d0 recording found a degenerate Unembedding boundary normal")
                values = (
                    reference_logits[row_index] - logits[row_index, start:end]
                ) / normal_norms.clamp_min(self.config.normal_epsilon)
                margins[row_index, start:end] = values.masked_fill(is_reference, 0.0)
        return margins

    @torch.no_grad()
    def _build_shard(self, start: int, end: int, shard_index: int) -> dict[str, Any]:
        batch, records = self._collate_chunk(start, end)
        positions = batch["selected_anchor_positions"].to(self.device)
        position_mask = batch["selected_anchor_position_mask"].to(self.device)
        target_ids = batch["selected_anchor_target_ids"].to(self.device)
        model_inputs = {
            "input_ids": batch["input_ids"].to(self.device),
            "attention_mask": batch["attention_mask"].to(self.device),
        }
        try:
            with UnembeddingInputCapture(
                self.model,
                positions=positions,
                stop_forward=True,
            ) as capture:
                self.model(**model_inputs)
        except UnembeddingInputCapture.ForwardStopped:
            pass
        if capture.hidden_states is None:
            raise RuntimeError("reference forward did not reach the Unembedding")

        hidden = capture.hidden_states.float()
        head = get_unembedding(self.model)
        bias = getattr(head, "bias", None)
        head_weight = head.weight.float()
        linear_logits = functional.linear(
            hidden,
            head_weight,
            bias.float() if bias is not None else None,
        )
        vocabulary_size = linear_logits.shape[-1]
        if vocabulary_size <= 1:
            raise ValueError("d0 recording requires at least two vocabulary tokens")
        reference_logits, reference_ids = linear_logits.max(dim=-1)
        confidence = (reference_logits - linear_logits.logsumexp(dim=-1)).exp()

        valid = position_mask
        base_correct = target_ids.eq(reference_ids) & target_ids.ne(-100)
        filter_mode = ReferenceFilterMode(self.config.filter_mode)
        if filter_mode is ReferenceFilterMode.BASE_CORRECT:
            valid = valid & base_correct
        elif filter_mode is ReferenceFilterMode.MIN_CONFIDENCE:
            valid = valid & confidence.ge(float(self.config.minimum_confidence))

        flat_hidden = hidden[valid]
        flat_logits = linear_logits[valid]
        flat_reference_ids = reference_ids[valid]
        flat_reference_logits = reference_logits[valid]
        flat_confidence = confidence[valid]
        flat_target_ids = target_ids[valid]
        flat_base_correct = base_correct[valid]
        baseline_margins = self._all_token_margins(
            flat_logits,
            flat_reference_ids,
            head_weight,
        )

        row_indices = torch.arange(end - start, device=self.device).unsqueeze(1)
        row_indices = row_indices.expand_as(position_mask)[valid]
        flat_positions = positions[valid]
        if filter_mode is ReferenceFilterMode.MIN_MARGIN:
            margin_rows = torch.arange(len(baseline_margins), device=self.device)
            baseline_margins[margin_rows, flat_reference_ids] = torch.inf
            margin_valid = baseline_margins.amin(dim=-1).ge(float(self.config.minimum_margin))
            baseline_margins[margin_rows, flat_reference_ids] = 0.0
            flat_hidden = flat_hidden[margin_valid]
            flat_reference_ids = flat_reference_ids[margin_valid]
            flat_reference_logits = flat_reference_logits[margin_valid]
            flat_confidence = flat_confidence[margin_valid]
            flat_target_ids = flat_target_ids[margin_valid]
            flat_base_correct = flat_base_correct[margin_valid]
            baseline_margins = baseline_margins[margin_valid]
            row_indices = row_indices[margin_valid]
            flat_positions = flat_positions[margin_valid]
        tensors = {
            "hidden_states": flat_hidden.to(self._storage_dtype).cpu().contiguous(),
            "reference_token_ids": flat_reference_ids.cpu().contiguous(),
            "reference_logits": flat_reference_logits.to(self._storage_dtype).cpu().contiguous(),
            "baseline_margins": baseline_margins.to(self._storage_dtype).cpu().contiguous(),
            "reference_confidence": flat_confidence.to(self._storage_dtype).cpu().contiguous(),
            "target_token_ids": flat_target_ids.cpu().contiguous(),
            "base_correct": flat_base_correct.cpu().contiguous(),
            "token_positions": flat_positions.cpu().contiguous(),
        }
        precision_audit = self._audit_storage_precision(
            flat_hidden,
            flat_reference_ids,
            baseline_margins,
            head,
            bias,
        )

        stem = f"shard-{shard_index:06d}"
        tensor_path = self.output_dir / f"{stem}.safetensors"
        sidecar_path = self.output_dir / f"{stem}.jsonl"
        temporary_tensor = tensor_path.with_suffix(".safetensors.tmp")
        temporary_sidecar = sidecar_path.with_suffix(".jsonl.tmp")
        save_file(tensors, temporary_tensor)
        sidecar_lines = []
        for cache_row, source_row in enumerate(row_indices.tolist()):
            record = records[source_row]
            sidecar_lines.append(
                json.dumps(
                    {
                        "cache_row": cache_row,
                        "sample_id": record["sample_id"],
                        "token_position": int(flat_positions[cache_row].item()),
                        "source_metadata": record.get("source_metadata", {}),
                    },
                    ensure_ascii=True,
                    sort_keys=True,
                )
            )
        temporary_sidecar.write_text(
            "\n".join(sidecar_lines) + ("\n" if sidecar_lines else ""),
            encoding="utf-8",
        )
        os.replace(temporary_tensor, tensor_path)
        os.replace(temporary_sidecar, sidecar_path)
        return {
            "index": shard_index,
            "sample_start": start,
            "sample_end": end,
            "position_count": int(flat_positions.numel()),
            "tensor_file": tensor_path.name,
            "tensor_sha256": _sha256(tensor_path),
            "sidecar_file": sidecar_path.name,
            "sidecar_sha256": _sha256(sidecar_path),
            "precision_audit": precision_audit,
        }

    def _audit_storage_precision(
        self,
        hidden_states: torch.Tensor,
        reference_token_ids: torch.Tensor,
        baseline_margins: torch.Tensor,
        head,
        bias,
    ) -> dict[str, Any] | None:
        if self.config.storage_dtype == "float32" or not hidden_states.numel():
            return None
        candidate_hidden = hidden_states.to(self._storage_dtype).float()
        candidate_reference_rows: list[torch.Tensor] = []
        candidate_margin_rows: list[torch.Tensor] = []
        head_weight = head.weight.float()
        head_bias = bias.float() if bias is not None else None
        for start in range(0, len(candidate_hidden), 32):
            chunk = candidate_hidden[start : start + 32]
            candidate_logits = functional.linear(
                chunk,
                head_weight,
                head_bias,
            )
            chunk_reference_ids = candidate_logits.argmax(dim=-1)
            candidate_reference_rows.append(chunk_reference_ids)
            candidate_margin_rows.append(
                self._all_token_margins(
                    candidate_logits,
                    chunk_reference_ids,
                    head_weight,
                )
            )
        candidate_reference_ids = torch.cat(candidate_reference_rows, dim=0)
        candidate_margins = torch.cat(candidate_margin_rows, dim=0)
        report = compare_reference_cache_values(
            {
                "hidden_states": hidden_states,
                "reference_token_ids": reference_token_ids,
                "baseline_margins": baseline_margins,
            },
            {
                "hidden_states": candidate_hidden,
                "reference_token_ids": candidate_reference_ids,
                "baseline_margins": candidate_margins,
            },
        )
        audit = audit_reference_precision(report, self.config.precision_budget)
        if not audit["passed"]:
            raise ValueError(f"compressed reference cache exceeds precision budget: {audit}")
        return audit

    def build(self) -> dict[str, Any]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        manifest = self._load_manifest()
        if manifest.get("status") == "complete":
            ReferenceCache(self.output_dir).validate_integrity()
            return manifest

        existing = {int(item["index"]): item for item in manifest.get("shards", [])}
        shard_count = (
            len(self.dataset) + self.config.samples_per_shard - 1
        ) // self.config.samples_per_shard
        shards: list[dict[str, Any]] = []
        was_training = self.model.training
        original_device = next(self.model.parameters()).device
        self.model.eval().to(self.device)
        try:
            for shard_index in range(shard_count):
                start = shard_index * self.config.samples_per_shard
                end = min(start + self.config.samples_per_shard, len(self.dataset))
                shard = existing.get(shard_index)
                if shard is not None:
                    tensor_path = self.output_dir / shard["tensor_file"]
                    sidecar_path = self.output_dir / shard["sidecar_file"]
                    if (
                        tensor_path.is_file()
                        and sidecar_path.is_file()
                        and _sha256(tensor_path) == shard["tensor_sha256"]
                        and _sha256(sidecar_path) == shard["sidecar_sha256"]
                    ):
                        shards.append(shard)
                        continue
                shard = self._build_shard(start, end, shard_index)
                shards.append(shard)
                manifest["shards"] = shards
                manifest["position_count"] = sum(item["position_count"] for item in shards)
                _atomic_json(self.manifest_path, manifest)
        finally:
            self.model.to(original_device)
            self.model.train(was_training)
        manifest["shards"] = shards
        manifest["position_count"] = sum(item["position_count"] for item in shards)
        if manifest["position_count"] == 0:
            raise ValueError(
                "Anchor d0 recording found no eligible positions; increase the Anchor sample "
                "budget or use d0_filter_mode=all_reference for a pipeline smoke test"
            )
        manifest["status"] = "complete"
        _atomic_json(self.manifest_path, manifest)
        return manifest


class ReferenceCache:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.manifest_path = self.root / "manifest.json"
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"reference cache manifest is missing: {self.manifest_path}")
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("status") != "complete":
            raise ValueError("reference cache is not complete")
        if int(self.manifest.get("position_count", 0)) <= 0:
            raise ValueError("reference cache contains no eligible Anchor d0 positions")
        if int(self.manifest.get("schema_version", -1)) != REFERENCE_CACHE_SCHEMA_VERSION:
            raise ValueError(
                "reference cache schema is incompatible; rebuild it with record_anchor_d0"
            )
        self._index: dict[str, list[tuple[int, int, dict[str, Any]]]] = {}
        self._position_index: dict[tuple[str, int], tuple[int, int, dict[str, Any]]] = {}
        self._loaded_shards: dict[int, dict[str, torch.Tensor]] = {}
        self._loaded_tensors: dict[tuple[int, str], torch.Tensor] = {}
        for shard in self.manifest["shards"]:
            shard_index = int(shard["index"])
            sidecar = self.root / shard["sidecar_file"]
            for line in sidecar.read_text(encoding="utf-8").splitlines():
                value = json.loads(line)
                self._index.setdefault(value["sample_id"], []).append(
                    (shard_index, int(value["cache_row"]), value)
                )
                key = (str(value["sample_id"]), int(value["token_position"]))
                if key in self._position_index:
                    raise ValueError(f"duplicate reference cache position: {key}")
                self._position_index[key] = (shard_index, int(value["cache_row"]), value)

    def validate_integrity(self) -> None:
        total = 0
        vocabulary_size = int(self.manifest.get("vocabulary_size", 0))
        if vocabulary_size <= 1:
            raise ValueError("reference cache vocabulary size is invalid")
        for shard in self.manifest["shards"]:
            tensor_path = self.root / shard["tensor_file"]
            sidecar_path = self.root / shard["sidecar_file"]
            if _sha256(tensor_path) != shard["tensor_sha256"]:
                raise ValueError(f"reference tensor shard hash mismatch: {tensor_path}")
            if _sha256(sidecar_path) != shard["sidecar_sha256"]:
                raise ValueError(f"reference sidecar hash mismatch: {sidecar_path}")
            tensors = load_file(tensor_path)
            counts = {tensor.shape[0] for tensor in tensors.values()}
            if counts != {int(shard["position_count"])}:
                raise ValueError(f"reference shard row-count mismatch: {tensor_path}")
            margins = tensors.get("baseline_margins")
            if margins is None or margins.ndim != 2 or margins.shape[1] != vocabulary_size:
                raise ValueError(
                    f"reference shard does not contain vocabulary-indexed d0: {tensor_path}"
                )
            total += int(shard["position_count"])
        if total != int(self.manifest["position_count"]):
            raise ValueError("reference cache total position count mismatch")

    def _shard(self, shard_index: int) -> dict[str, torch.Tensor]:
        if shard_index not in self._loaded_shards:
            shard = next(
                item for item in self.manifest["shards"] if int(item["index"]) == shard_index
            )
            self._loaded_shards[shard_index] = load_file(self.root / shard["tensor_file"])
        return self._loaded_shards[shard_index]

    def _shard_tensor(self, shard_index: int, name: str) -> torch.Tensor:
        key = (shard_index, name)
        if key not in self._loaded_tensors:
            shard = next(
                item for item in self.manifest["shards"] if int(item["index"]) == shard_index
            )
            path = self.root / shard["tensor_file"]
            with safe_open(path, framework="pt", device="cpu") as handle:
                available_fields = set(handle.keys())
                if name not in available_fields:
                    raise KeyError(f"reference cache tensor field is missing: {name}")
                self._loaded_tensors[key] = handle.get_tensor(name)
        return self._loaded_tensors[key]

    def sample_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._index))

    def get(self, sample_id: str) -> dict[str, Any]:
        references = self._index.get(sample_id)
        if not references:
            raise KeyError(sample_id)
        tensor_rows: dict[str, list[torch.Tensor]] = {}
        metadata: list[dict[str, Any]] = []
        for shard_index, cache_row, sidecar in references:
            for name, tensor in self._shard(shard_index).items():
                tensor_rows.setdefault(name, []).append(tensor[cache_row])
            metadata.append(sidecar)
        return {
            **{name: torch.stack(rows) for name, rows in tensor_rows.items()},
            "metadata": metadata,
        }

    def get_position(self, sample_id: str, token_position: int) -> dict[str, Any] | None:
        reference = self._position_index.get((sample_id, int(token_position)))
        if reference is None:
            return None
        shard_index, cache_row, sidecar = reference
        return {
            **{name: tensor[cache_row] for name, tensor in self._shard(shard_index).items()},
            "metadata": sidecar,
        }

    def get_position_fields(
        self,
        sample_id: str,
        token_position: int,
        fields: tuple[str, ...],
    ) -> dict[str, Any] | None:
        if not fields:
            raise ValueError("reference cache field selection must not be empty")
        reference = self._position_index.get((sample_id, int(token_position)))
        if reference is None:
            return None
        shard_index, cache_row, sidecar = reference
        return {
            **{
                name: self._shard_tensor(shard_index, name)[cache_row]
                for name in dict.fromkeys(fields)
            },
            "metadata": sidecar,
        }

    def validate_binding(
        self,
        *,
        model_id: str,
        model_revision: str,
        vocabulary_size: int,
        dataset_fingerprint: str,
        position_config: dict[str, Any],
        normal_epsilon: float,
        filter_mode: str,
        storage_dtype: str,
        minimum_confidence: float | None,
        minimum_margin: float | None,
    ) -> None:
        expected = {
            "model_id": model_id,
            "model_revision": model_revision,
            "vocabulary_size": vocabulary_size,
            "dataset_fingerprint": dataset_fingerprint,
            "position_config": position_config,
            "normal_epsilon": normal_epsilon,
            "filter_mode": filter_mode,
            "storage_dtype": storage_dtype,
            "minimum_confidence": minimum_confidence,
            "minimum_margin": minimum_margin,
        }
        mismatches = {
            key: (self.manifest.get(key), value)
            for key, value in expected.items()
            if self.manifest.get(key) != value
        }
        if mismatches:
            raise ValueError(f"Anchor d0 cache binding mismatch: {mismatches}")


def compare_reference_cache_values(
    reference: dict[str, torch.Tensor],
    candidate: dict[str, torch.Tensor],
) -> dict[str, float]:
    hidden_error = (reference["hidden_states"].float() - candidate["hidden_states"].float()).abs()
    margin_error = (
        reference["baseline_margins"].float() - candidate["baseline_margins"].float()
    ).abs()
    margin_relative_error = margin_error / reference["baseline_margins"].float().abs().clamp_min(
        1e-12
    )
    argmax_match = reference["reference_token_ids"].eq(candidate["reference_token_ids"])
    return {
        "hidden_max_absolute_error": (
            float(hidden_error.max().item()) if hidden_error.numel() else 0.0
        ),
        "hidden_mean_absolute_error": (
            float(hidden_error.mean().item()) if hidden_error.numel() else 0.0
        ),
        "argmax_agreement": (
            float(argmax_match.float().mean().item()) if argmax_match.numel() else 1.0
        ),
        "margin_max_absolute_error": (
            float(margin_error.max().item()) if margin_error.numel() else 0.0
        ),
        "margin_mean_absolute_error": (
            float(margin_error.mean().item()) if margin_error.numel() else 0.0
        ),
        "margin_max_relative_error": (
            float(margin_relative_error.max().item()) if margin_relative_error.numel() else 0.0
        ),
        "margin_sign_flip_rate": float(
            reference["baseline_margins"]
            .float()
            .sign()
            .ne(candidate["baseline_margins"].float().sign())
            .float()
            .mean()
            .item()
        )
        if margin_error.numel()
        else 0.0,
    }


def audit_reference_precision(
    report: dict[str, float],
    budget: ReferencePrecisionBudget | None,
) -> dict[str, Any]:
    if budget is None:
        raise ValueError("a preregistered precision budget is required")
    budget.validate()
    thresholds = asdict(budget)
    checks = {
        "hidden_max_absolute_error": (
            report["hidden_max_absolute_error"] <= budget.hidden_max_absolute_error
        ),
        "hidden_mean_absolute_error": (
            report["hidden_mean_absolute_error"] <= budget.hidden_mean_absolute_error
        ),
        "argmax_agreement": report["argmax_agreement"] >= budget.minimum_argmax_agreement,
        "margin_max_absolute_error": (
            report["margin_max_absolute_error"] <= budget.margin_max_absolute_error
        ),
        "margin_max_relative_error": (
            report["margin_max_relative_error"] <= budget.margin_max_relative_error
        ),
        "margin_sign_flip_rate": (
            report["margin_sign_flip_rate"] <= budget.maximum_margin_sign_flip_rate
        ),
    }
    return {
        "passed": all(checks.values()),
        "metrics": report,
        "thresholds": thresholds,
        "checks": checks,
    }
