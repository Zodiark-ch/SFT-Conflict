"""Offline model construction and training-mode configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from sft_conflict.adapters import LoraAudit, apply_standard_lora
from sft_conflict.config import (
    LoraConfig,
    ModelConfig,
    ReferenceConfig,
    ReferenceMode,
    TrainingConfig,
    TrainingMode,
)

from .unembedding import (
    get_unembedding,
    has_tied_input_output_embeddings,
    untie_output_embeddings,
)


class FSDPUnembeddingLinear(nn.Linear):
    """Marker class that gives an untied output head its own FSDP unit."""


def _mark_untied_unembedding_for_fsdp(model: nn.Module) -> tuple[str, ...]:
    if has_tied_input_output_embeddings(model):
        return ()
    output_embeddings = get_unembedding(model)
    if type(output_embeddings) is nn.Linear:
        output_embeddings.__class__ = FSDPUnembeddingLinear
    if isinstance(output_embeddings, FSDPUnembeddingLinear):
        return (FSDPUnembeddingLinear.__name__,)
    return ()


@dataclass(frozen=True)
class ModelAudit:
    model_type: str
    total_parameters: int
    trainable_parameters: int
    trainable_parameter_names: tuple[str, ...]
    unembedding_module: str
    unembedding_frozen: bool
    tied_before: bool
    tied_after: bool
    lora: dict[str, Any] | None
    no_split_modules: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModelBundle:
    model: nn.Module
    tokenizer: Any
    audit: ModelAudit


def _module_name(model: nn.Module, target_module: nn.Module) -> str:
    for name, module in model.named_modules():
        if module is target_module:
            return name
    return type(target_module).__name__


def _set_all_parameters(model: nn.Module, requires_grad: bool) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(requires_grad)


def _freeze_unembedding(model: nn.Module) -> None:
    for parameter in get_unembedding(model).parameters():
        parameter.requires_grad_(False)


def _resolve_untie(model: nn.Module, config: ModelConfig, input_embeddings_train: bool) -> None:
    if not has_tied_input_output_embeddings(model) or not input_embeddings_train:
        return
    if config.untie_tied_embeddings is False:
        raise ValueError(
            "the output head is tied to a trainable input embedding; set "
            "model.untie_tied_embeddings=true or auto"
        )
    untie_output_embeddings(model)


def _build_audit(
    model: nn.Module,
    *,
    tied_before: bool,
    lora_audit: LoraAudit | None,
) -> ModelAudit:
    output_embeddings = get_unembedding(model)
    trainable_names = tuple(
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    no_split_modules = tuple(
        dict.fromkeys(
            (
                *(getattr(model, "_no_split_modules", ()) or ()),
                *_mark_untied_unembedding_for_fsdp(model),
            )
        )
    )
    return ModelAudit(
        model_type=type(model).__name__,
        total_parameters=total_parameters,
        trainable_parameters=trainable_parameters,
        trainable_parameter_names=trainable_names,
        unembedding_module=_module_name(model, output_embeddings),
        unembedding_frozen=all(
            not parameter.requires_grad for parameter in output_embeddings.parameters()
        ),
        tied_before=tied_before,
        tied_after=has_tied_input_output_embeddings(model),
        lora=asdict(lora_audit) if lora_audit is not None else None,
        no_split_modules=no_split_modules,
    )


def audit_existing_training_model(model: nn.Module) -> ModelAudit:
    """Audit a checkpoint model whose trainable parameter policy is already configured."""
    return _build_audit(
        model,
        tied_before=has_tied_input_output_embeddings(model),
        lora_audit=None,
    )


def configure_model_for_training(
    model: nn.Module,
    model_config: ModelConfig,
    lora_config: LoraConfig,
) -> tuple[nn.Module, ModelAudit]:
    tied_before = has_tied_input_output_embeddings(model)
    lora_audit: LoraAudit | None = None

    if model_config.training_mode is TrainingMode.FULL_FT:
        _set_all_parameters(model, True)
    elif model_config.training_mode is TrainingMode.BODY_FT_FROZEN_HEAD:
        _set_all_parameters(model, True)
        _resolve_untie(model, model_config, input_embeddings_train=True)
        _freeze_unembedding(model)
    elif model_config.training_mode is TrainingMode.LORA:
        _resolve_untie(
            model,
            model_config,
            input_embeddings_train=lora_config.train_input_embeddings,
        )
        _set_all_parameters(model, False)
        model, lora_audit = apply_standard_lora(model, lora_config)
        _freeze_unembedding(model)
        if lora_config.train_input_embeddings:
            input_embeddings = model.get_input_embeddings()
            if not any(parameter.requires_grad for parameter in input_embeddings.parameters()):
                raise RuntimeError(
                    "LoRA requested trainable input embeddings but PEFT did not enable them"
                )
    else:
        raise ValueError(f"unsupported training mode: {model_config.training_mode}")

    if model_config.freeze_unembedding != all(
        not parameter.requires_grad for parameter in get_unembedding(model).parameters()
    ):
        raise RuntimeError("resolved Unembedding state does not match model.freeze_unembedding")
    audit = _build_audit(model, tied_before=tied_before, lora_audit=lora_audit)
    return model, audit


def torch_dtype_for_precision(mixed_precision: str) -> torch.dtype:
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[mixed_precision]


def enable_gradient_checkpointing(model: nn.Module) -> None:
    enable_input_require_grads = getattr(model, "enable_input_require_grads", None)
    if callable(enable_input_require_grads):
        enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(model, "config"):
        model.config.use_cache = False
    if not getattr(model, "is_gradient_checkpointing", False):
        raise RuntimeError("gradient checkpointing was requested but is not active")


def _cache_error(model_id: str, revision: str, cache_dir: str | None) -> RuntimeError:
    location = cache_dir or "the configured Hugging Face cache"
    return RuntimeError(
        f"Offline model cache is incomplete for {model_id}@{revision}. "
        f"Expected files under {location}. In run.py, set AUTO_PREPARE_CACHE=True "
        "or prepare that fixed revision before training."
    )


def load_model_bundle(
    model_config: ModelConfig,
    lora_config: LoraConfig,
    training_config: TrainingConfig,
    *,
    cache_dir: str | Path | None = None,
) -> ModelBundle:
    resolved_cache_dir = str(cache_dir) if cache_dir is not None else None
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_config.model_id,
            revision=model_config.tokenizer_revision,
            cache_dir=resolved_cache_dir,
            trust_remote_code=model_config.trust_remote_code,
            local_files_only=model_config.local_files_only,
        )
    except OSError:
        raise _cache_error(
            model_config.model_id,
            model_config.tokenizer_revision or model_config.revision,
            resolved_cache_dir,
        ) from None
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer has neither a pad token nor an EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model_kwargs: dict[str, Any] = {
        "revision": model_config.revision,
        "trust_remote_code": model_config.trust_remote_code,
        "local_files_only": model_config.local_files_only,
        "cache_dir": resolved_cache_dir,
        "torch_dtype": torch_dtype_for_precision(training_config.mixed_precision),
    }
    if model_config.use_flash_attention_2:
        model_kwargs["attn_implementation"] = "flash_attention_2"
    try:
        model = AutoModelForCausalLM.from_pretrained(model_config.model_id, **model_kwargs)
    except OSError:
        raise _cache_error(
            model_config.model_id,
            model_config.revision,
            resolved_cache_dir,
        ) from None
    model.to(dtype=torch_dtype_for_precision(training_config.mixed_precision))
    model, audit = configure_model_for_training(model, model_config, lora_config)

    if model_config.gradient_checkpointing:
        if training_config.distributed_strategy == "fsdp":
            model.config.use_cache = False
            if model_config.training_mode is TrainingMode.LORA:
                enable_input_require_grads = getattr(model, "enable_input_require_grads", None)
                if callable(enable_input_require_grads):
                    enable_input_require_grads()
        else:
            enable_gradient_checkpointing(model)
    if model_config.use_flash_attention_2:
        actual_backend = getattr(model.config, "_attn_implementation", None)
        if actual_backend != "flash_attention_2":
            raise RuntimeError(
                "Flash Attention 2 was requested but the model did not activate that backend"
            )
    return ModelBundle(model=model, tokenizer=tokenizer, audit=audit)


def load_reference_model(
    reference_config: ReferenceConfig,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    *,
    cache_dir: str | Path | None = None,
) -> nn.Module | None:
    if reference_config.mode is ReferenceMode.NONE:
        return None
    if reference_config.mode is ReferenceMode.CACHED:
        return None
    if reference_config.mode is not ReferenceMode.ONLINE:
        raise ValueError(f"unsupported reference mode: {reference_config.mode}")
    if not reference_config.model_id or not reference_config.revision:
        raise ValueError("online reference mode requires a model ID and immutable revision")
    resolved_cache_dir = str(cache_dir) if cache_dir is not None else None
    try:
        reference_model = AutoModelForCausalLM.from_pretrained(
            reference_config.model_id,
            revision=reference_config.revision,
            cache_dir=resolved_cache_dir,
            trust_remote_code=model_config.trust_remote_code,
            local_files_only=model_config.local_files_only,
            torch_dtype=torch_dtype_for_precision(training_config.mixed_precision),
        )
    except OSError:
        raise _cache_error(
            reference_config.model_id,
            reference_config.revision,
            resolved_cache_dir,
        ) from None
    reference_model.requires_grad_(False)
    reference_model.eval()
    if any(parameter.requires_grad for parameter in reference_model.parameters()):
        raise RuntimeError("the reference model is not fully frozen")
    return reference_model
