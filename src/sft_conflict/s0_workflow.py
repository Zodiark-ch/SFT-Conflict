"""Model-head-only S0 recording and early training preflight.

No tokenizer, Anchor/Target data, Transformer forward, optimizer checkpoint,
reference d0 or pilot is involved in recording. Published anchors describe a
bounded full-vocabulary probability optimization, not an attained infinity.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path, PurePosixPath
from typing import Any

import torch
from huggingface_hub import try_to_load_from_cache
from safetensors import safe_open

from sft_conflict.config import ConfigError, ExperimentConfig, LVDPlusAnchorMode, MethodName
from sft_conflict.data.s0_cache import S0Cache, head_fingerprint, write_s0_cache
from sft_conflict.geometry.s0 import optimize_s0
from sft_conflict.geometry.s0_classification import classify_s0_optima
from sft_conflict.tracking.metadata import write_json_atomic


def load_s0_head(config: ExperimentConfig) -> tuple[torch.Tensor, torch.Tensor | None, dict]:
    """Read only lm_head (or an explicitly tied embedding) from local Safetensors."""
    source = Path(config.model.model_id).expanduser()
    if source.is_dir():
        snapshot = source.resolve()
    else:
        config_path = try_to_load_from_cache(
            config.model.model_id,
            "config.json",
            revision=config.model.revision,
            cache_dir=Path(config.paths.artifact_root) / "hf_cache/hub",
        )
        if not isinstance(config_path, str):
            raise ConfigError("record_s0 requires the fixed model snapshot in the local cache")
        snapshot = Path(config_path).parent
    model_config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    index = snapshot / "model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
    else:
        single = snapshot / "model.safetensors"
        if not single.is_file():
            raise ConfigError("record_s0 requires a local Safetensors model, not pickle weights")
        with safe_open(single, framework="pt", device="cpu") as handle:
            weight_map = dict.fromkeys(handle.keys(), single.name)
    key = "lm_head.weight"
    if key not in weight_map and model_config.get("tie_word_embeddings") is True:
        tied = [
            name
            for name in ("model.embed_tokens.weight", "transformer.wte.weight")
            if name in weight_map
        ]
        if len(tied) == 1:
            key = tied[0]
    if key not in weight_map:
        raise ConfigError("record_s0 cannot identify the fixed model's Unembedding tensor")

    def tensor(name: str) -> torch.Tensor:
        shard = PurePosixPath(weight_map[name])
        if shard.is_absolute() or ".." in shard.parts or shard.suffix != ".safetensors":
            raise ConfigError("invalid Safetensors shard name in model index")
        with safe_open(snapshot / shard, framework="pt", device="cpu") as handle:
            value = handle.get_tensor(name).detach().float().clone()
        if not bool(torch.isfinite(value).all()):
            raise ConfigError(f"non-finite Unembedding tensor: {name}")
        return value

    weight = tensor(key)
    bias = tensor("lm_head.bias") if "lm_head.bias" in weight_map else None
    if weight.ndim != 2 or weight.shape[0] != model_config["vocab_size"]:
        raise ConfigError("Unembedding shape does not match the vocabulary")
    return (
        weight,
        bias,
        {
            "model_snapshot": str(snapshot),
            "weight_key": key,
            "bias_key": "lm_head.bias" if bias is not None else None,
            "vocabulary_size": weight.shape[0],
            "hidden_size": weight.shape[1],
            "transformer_loaded": False,
            "storage_conversion": "source floating weights explicitly converted to FP32",
        },
    )


def require_s0_cache(config: ExperimentConfig) -> S0Cache | None:
    """Reject incomplete/inapplicable anchors before any training model is loaded.

    Full head binding happens again against the actual trainer's frozen snapshot.
    This preflight verifies file integrity, solver/model identities and usability.
    Cell membership is diagnostic only; all rows must pass bounded numerical
    convergence without replacing or dropping any recorded site.
    """
    if not config.method.name.uses_lvd_plus or (
        config.lvd_plus.anchor_mode is not LVDPlusAnchorMode.RECORDED_S0
    ):
        return None
    try:
        cache = S0Cache(config.lvd_plus.s0_cache_path)
        cache.validate_integrity()
        manifest = cache.manifest
        if not manifest["usable_for_training"]:
            raise ValueError(
                "some vocabulary anchors fail numerical acceptance (finite values, ball "
                "feasibility, reported convergence and Frank-Wolfe gap <= gap_tolerance)"
            )
        if manifest["model_id"] != config.model.model_id or (
            manifest["model_revision"] != config.model.revision
        ):
            raise ValueError("model/revision binding mismatch")
        if manifest["solver_config"] != asdict(config.s0):
            raise ValueError("S0 optimization settings differ from the recorded cache")
        requested = config.lvd_plus.s0_cache_sha256
        if requested is not None and requested != manifest["manifest_sha256"]:
            raise ValueError("resolved S0 manifest SHA mismatch")
    except (OSError, ValueError, TypeError) as error:
        raise ConfigError(
            f"S0 cache is missing or incompatible: {error}. Run METHOD='record_s0' "
            "with this model and S0 configuration first; no W_j/2 fallback is allowed."
        ) from error
    return cache


def record_s0(config: ExperimentConfig, *, device: str) -> dict[str, Any]:
    """Record exactly one bounded anchor per vocabulary row and report all outcomes."""
    if config.method.name is not MethodName.RECORD_S0:
        raise ConfigError("record_s0 requires method=record_s0")
    config.validate()
    if config.training.resume_from_checkpoint:
        raise ConfigError("record_s0 uses the frozen Base head, never a training checkpoint")
    weight, bias, head_info = load_s0_head(config)
    destination = Path(config.lvd_plus.s0_cache_path).expanduser().resolve()
    solver_config = asdict(config.s0)
    run_id = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(config.paths.run_name)).strip("-.").lower()
    if not run_id or run_id == "none":
        raise ConfigError("record_s0 requires a unique run name for its diagnostic report")
    result_dir = Path(config.paths.artifact_root) / "results" / run_id
    result_dir.mkdir(parents=True, exist_ok=False)
    if destination.exists():
        cache = S0Cache(destination)
        cache.validate_integrity()
        manifest = cache.manifest
        expected = {
            "head_sha256": head_fingerprint(weight, bias),
            "model_id": config.model.model_id,
            "model_revision": config.model.revision,
            "solver_config": solver_config,
        }
        if any(manifest[key] != value for key, value in expected.items()):
            raise ConfigError("refusing to overwrite an incompatible immutable S0 cache")
        reused = True
    else:
        options = dict(solver_config)
        options.pop("config_version")

        def progress(event: dict[str, object]) -> None:
            write_json_atomic(result_dir / "progress.json", event)
            print(
                f"S0 {event['completed_tokens']}/{event['vocabulary_size']} tokens; "
                f"converged={event['converged_tokens']}; "
                f"batch gap={event['batch_max_frank_wolfe_gap']:.6g}",
                flush=True,
            )

        optimized = optimize_s0(weight, bias, **options, device=device, progress_callback=progress)
        print("S0: verifying unconstrained-optimum and exact-duplicate certificates", flush=True)
        classification = classify_s0_optima(
            weight,
            bias,
            optimized.sites,
            device=device,
            token_batch_size=config.s0.token_batch_size,
            vocabulary_chunk_size=config.s0.vocabulary_chunk_size,
        )
        facts = classification.per_token
        outside = optimized.per_token["outside_cell"]
        cell_known = (
            facts["classification_cell_unbounded_certified"]
            | facts["classification_cell_bounded_certified"]
            | facts["classification_cell_empty_certified"]
        )
        statistics = {
            "vocabulary_rows": len(weight),
            "bounded_problem_converged": optimized.summary["converged_count"],
            "bounded_problem_unresolved": optimized.summary["unresolved_count"],
            "radius": config.s0.radius,
            "maximum_frank_wolfe_gap": optimized.summary["max_frank_wolfe_gap"],
            "unconstrained_finite_certified": classification.summary["finite_certified_count"],
            "unconstrained_infinite_certified": classification.summary["infinite_certified_count"],
            "unconstrained_unknown": classification.summary["unknown_count"],
            "probability_one_supremum": classification.summary["p1_certified_count"],
            "duplicate_group_supremum": classification.summary[
                "duplicate_supremum_certified_count"
            ],
            "cell_membership": optimized.summary["cell_counts"],
            "cell_geometry": classification.summary["cell_counts"],
            "outside_bounded_cells_certified": int(
                (outside & facts["classification_cell_bounded_certified"]).sum()
            ),
            "outside_cells_unknown_geometry": int((outside & ~cell_known).sum()),
        }
        manifest = write_s0_cache(
            destination,
            sites=optimized.sites,
            per_token={**optimized.per_token, **classification.per_token},
            summary={
                **optimized.summary,
                "unconstrained_classification": classification.summary,
                "recording_statistics": statistics,
            },
            weight=weight,
            bias=bias,
            model_id=config.model.model_id,
            model_revision=config.model.revision,
            solver_config=solver_config,
            certificates=classification.certificates,
        )
        reused = False
    result = {
        "action": "record-s0",
        "status": "usable" if manifest["usable_for_training"] else "diagnostic_only",
        "cache_path": str(destination),
        "manifest_sha256": manifest["manifest_sha256"],
        "usable_for_training": manifest["usable_for_training"],
        "reused_existing_cache": reused,
        "head": head_info,
        "counts": manifest["counts"],
        "solver_summary": manifest["solver_summary"],
        "classification": manifest["solver_summary"].get("unconstrained_classification"),
        "statistics": manifest["solver_summary"].get("recording_statistics"),
        "result_dir": str(result_dir),
        "limitations": [
            "S0 solves a finite-radius ball problem, not an attained infinite maximizer.",
            "Unconstrained finite/infinite labels require certificates; unresolved is not finite.",
            "Cell membership is diagnostic only: tied/outside rows use the same recorded S0.",
            "Training requires every row to be finite, ball-feasible and reported converged "
            "with Frank-Wolfe gap <= gap_tolerance.",
            "An S0 anchor does not imply radial softmax monotonicity or downstream quality.",
        ],
    }
    write_json_atomic(result_dir / "summary.json", result)
    return result
