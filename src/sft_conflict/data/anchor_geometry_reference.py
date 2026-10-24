"""Immutable, sharded Base outputs for dynamic geometry (not a cell/S0 cache)."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from tqdm.auto import tqdm

from sft_conflict.config import AnchorGeometryConfig
from sft_conflict.data.collator import CausalLMCollator, model_tensor_inputs
from sft_conflict.models import UnembeddingInputCapture, get_unembedding


def json_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_digest(model) -> str:
    digest = hashlib.sha256(b"anchor-geometry-model-v1")
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(json_digest([name, str(tensor.dtype), list(tensor.shape)]).encode())
        digest.update(memoryview(tensor.reshape(-1).view(torch.uint8).numpy()))
    return digest.hexdigest()


def sample_digest(record: dict[str, Any]) -> str:
    return json_digest(
        {
            name: record[name]
            for name in (
                "sample_id",
                "input_ids",
                "attention_mask",
                "labels",
            )
        }
    )


def reference_binding(config, dataset, tokenizer) -> dict[str, Any]:
    fingerprint = getattr(dataset, "fingerprint", None)
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError("geometry reference requires a tokenized dataset fingerprint")
    return {
        "model_id": config.model.model_id,
        "model_revision": config.model.revision,
        "max_length": config.data.max_length,
        "dataset_fingerprint": fingerprint,
        "sample_count": len(dataset),
        "base_top_k": config.anchor_geometry.base_top_k,
        "compute_precision": config.training.mixed_precision,
        "storage_dtype": "float32",
        "tokenizer_sha256": json_digest(
            {
                "vocabulary": tokenizer.get_vocab(),
                "special_ids": list(tokenizer.all_special_ids),
                "padding_side": tokenizer.padding_side,
                "class": type(tokenizer).__name__,
            }
        ),
    }


class AnchorGeometryReference:
    def __init__(self, path: str | Path, *, expected_sha256: str | None = None) -> None:
        self.path = Path(path).expanduser().resolve()
        manifest_path = self.path / "manifest.json"
        self.sha256 = file_digest(manifest_path)
        if expected_sha256 is not None and self.sha256 != expected_sha256:
            raise ValueError("geometry reference manifest SHA mismatch")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            self.manifest.get("version") != 1
            or self.manifest.get("kind") != "anchor_geometry_reference"
        ):
            raise ValueError("not a dynamic geometry output reference")
        self._shards: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._verified: set[str] = set()
        head = self._load("head.pt")
        self.weight = head["weight"]
        self.bias = head["bias"]
        self.frequencies = head["frequencies"]
        if (
            self.weight.ndim != 2
            or self.weight.dtype != torch.float32
            or not torch.isfinite(self.weight).all()
            or self.frequencies.shape != (len(self.weight),)
            or torch.any(self.frequencies < 0)
        ):
            raise ValueError("invalid geometry reference head/frequencies")
        self.weight.requires_grad_(False)
        if self.bias is not None:
            if self.bias.shape != (len(self.weight),) or not torch.isfinite(self.bias).all():
                raise ValueError("invalid geometry reference bias")
            self.bias.requires_grad_(False)

    def _load(self, name: str) -> dict[str, Any]:
        if Path(name).name != name or name not in self.manifest["files"]:
            raise ValueError("invalid reference shard name")
        path = self.path / name
        if name not in self._verified:
            if file_digest(path) != self.manifest["files"][name]:
                raise ValueError(f"geometry reference file SHA mismatch: {name}")
            self._verified.add(name)
        return torch.load(path, map_location="cpu", weights_only=True)

    def validate(self, binding: dict[str, Any], *, initial_model_sha256: str | None = None) -> None:
        if self.manifest["binding"] != binding:
            raise ValueError("geometry reference dataset/tokenizer/model recipe mismatch")
        if (
            initial_model_sha256 is not None
            and self.manifest["initial_model_sha256"] != initial_model_sha256
        ):
            raise ValueError("geometry reference actual Base model mismatch")

    def validate_dataset(self, dataset) -> None:
        expected = self.manifest["records"]
        if len(dataset) != len(expected):
            raise ValueError("geometry reference sample count mismatch")
        for record in dataset:
            entry = expected.get(record["sample_id"])
            if entry is None or entry["sample_sha256"] != sample_digest(record):
                raise ValueError("geometry reference tokenized sample mismatch")

    def record(self, sample_id: str) -> dict[str, Tensor]:
        entry = self.manifest["records"].get(sample_id)
        if entry is None:
            raise ValueError(f"missing geometry reference sample: {sample_id}")
        name = entry["shard"]
        if name not in self._shards:
            self._shards[name] = self._load(name)
            while len(self._shards) > 2:
                self._shards.popitem(last=False)
        self._shards.move_to_end(name)
        return self._shards[name][sample_id]


@torch.no_grad()
def build_anchor_geometry_reference(
    path: str | Path,
    *,
    model,
    dataset,
    tokenizer,
    config,
) -> AnchorGeometryReference:
    geometry: AnchorGeometryConfig = config.anchor_geometry
    geometry.validate()
    binding = reference_binding(config, dataset, tokenizer)
    fingerprint = model_digest(model)
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with (output.parent / f".{output.name}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if output.exists():
            cache = AnchorGeometryReference(output)
            cache.validate(binding, initial_model_sha256=fingerprint)
            cache.validate_dataset(dataset)
            return cache
        head = get_unembedding(model)
        weight = head.weight.detach().float().cpu().clone()
        output_bias = getattr(head, "bias", None)
        bias = None if output_bias is None else output_bias.detach().float().cpu().clone()
        frequencies = torch.zeros(len(weight), dtype=torch.long)
        modes = {module: module.training for module in model.modules()}
        model.eval()
        collator = CausalLMCollator(tokenizer.pad_token_id, config.data.pad_to_multiple_of)
        device = next(model.parameters()).device
        try:
            with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as stage:
                directory = Path(stage)
                manifest = {
                    "version": 1,
                    "kind": "anchor_geometry_reference",
                    "binding": binding,
                    "initial_model_sha256": fingerprint,
                    "files": {},
                    "records": {},
                }
                shard: dict[str, Any] = {}
                shard_index = 0

                def flush() -> None:
                    nonlocal shard_index
                    if not shard:
                        return
                    name = f"samples-{shard_index:06d}.pt"
                    torch.save(shard, directory / name)
                    manifest["files"][name] = file_digest(directory / name)
                    for sample_id in shard:
                        manifest["records"][sample_id]["shard"] = name
                    shard.clear()
                    shard_index += 1

                for record in tqdm(dataset, desc="Base geometry reference", unit="sample"):
                    batch = collator([record])
                    positions = batch["labels"][0, 1:].ne(-100).nonzero().flatten()
                    labels = batch["labels"][0, positions + 1]
                    if len(positions) == 0:
                        raise ValueError("geometry reference samples require shifted supervision")
                    frequencies += torch.bincount(labels, minlength=len(weight))
                    selected_positions = positions[None, :].to(device)
                    inputs = {
                        key: value.to(device) for key, value in model_tensor_inputs(batch).items()
                    }
                    with UnembeddingInputCapture(model, positions=selected_positions) as capture:
                        outputs = model(**inputs)
                    hidden = capture.hidden_states[0].float().cpu()
                    logits = outputs.logits[0, positions.to(device)].float()
                    correct = logits.argmax(dim=-1).cpu().eq(labels)
                    logits.scatter_(1, labels[:, None].to(device), -torch.inf)
                    top_k = min(geometry.base_top_k, len(weight) - 1)
                    top_ids = logits.topk(top_k, dim=-1, sorted=True).indices.cpu()
                    if not torch.isfinite(hidden).all():
                        raise ValueError("non-finite Base reference hidden states")
                    sample_id = record["sample_id"]
                    if sample_id in manifest["records"]:
                        raise ValueError("duplicate geometry reference sample ID")
                    shard[sample_id] = {
                        "positions": positions,
                        "labels": labels,
                        "hidden": hidden,
                        "correct": correct,
                        "top_ids": top_ids,
                    }
                    manifest["records"][sample_id] = {"sample_sha256": sample_digest(record)}
                    if len(shard) >= geometry.reference_samples_per_shard:
                        flush()
                    del outputs, logits, hidden, capture
                flush()
                torch.save(
                    {"weight": weight, "bias": bias, "frequencies": frequencies},
                    directory / "head.pt",
                )
                manifest["files"]["head.pt"] = file_digest(directory / "head.pt")
                (directory / "manifest.json").write_text(
                    json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8",
                )
                os.rename(directory, output)
        finally:
            for module, training in modes.items():
                module.training = training
    return AnchorGeometryReference(output)
