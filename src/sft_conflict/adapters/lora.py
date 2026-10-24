"""Standard PEFT LoRA integration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from peft import LoraConfig as PeftLoraConfig
from peft import PeftModel, TaskType, get_peft_model
from torch import Tensor, nn

from sft_conflict.config import LoraConfig


@dataclass(frozen=True)
class LoraAudit:
    requested_targets: tuple[str, ...]
    matched_modules: tuple[str, ...]
    modules_to_save: tuple[str, ...]
    trainable_parameters: tuple[str, ...]


@dataclass(frozen=True)
class LoraMergeAudit:
    adapter_path: str
    max_absolute_error: float
    argmax_matches: bool


def _matches_target(module_name: str, target: str) -> bool:
    return module_name == target or module_name.endswith(f".{target}")


def _module_name(model: nn.Module, target_module: nn.Module) -> str:
    for name, module in model.named_modules():
        if module is target_module:
            return name
    raise RuntimeError("module is not registered on the model")


def apply_standard_lora(model: nn.Module, config: LoraConfig) -> tuple[nn.Module, LoraAudit]:
    matched_modules = tuple(
        sorted(
            name
            for name, _ in model.named_modules()
            if any(_matches_target(name, target) for target in config.target_modules)
        )
    )
    if not matched_modules:
        requested = ", ".join(config.target_modules)
        raise ValueError(f"none of the configured LoRA target modules matched: {requested}")

    modules_to_save = list(config.modules_to_save)
    if config.train_input_embeddings:
        input_name = _module_name(model, model.get_input_embeddings())
        input_leaf_name = input_name.rsplit(".", maxsplit=1)[-1]
        if input_leaf_name not in modules_to_save:
            modules_to_save.append(input_leaf_name)

    peft_config = PeftLoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.rank,
        lora_alpha=config.alpha,
        lora_dropout=config.dropout,
        bias=config.bias,
        target_modules=list(config.target_modules),
        modules_to_save=modules_to_save or None,
    )
    peft_model = get_peft_model(model, peft_config)
    trainable_parameters = tuple(
        name for name, parameter in peft_model.named_parameters() if parameter.requires_grad
    )
    if not trainable_parameters:
        raise RuntimeError("PEFT injected LoRA modules but produced no trainable parameters")

    audit = LoraAudit(
        requested_targets=config.target_modules,
        matched_modules=matched_modules,
        modules_to_save=tuple(modules_to_save),
        trainable_parameters=trainable_parameters,
    )
    return peft_model, audit


@torch.no_grad()
def load_and_merge_lora(
    base_model: nn.Module,
    adapter_path: str | Path,
    model_inputs: dict[str, Tensor],
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> tuple[nn.Module, LoraMergeAudit]:
    resolved_adapter_path = Path(adapter_path).expanduser().resolve()
    if not (resolved_adapter_path / "adapter_config.json").is_file():
        raise FileNotFoundError(
            f"LoRA adapter is missing adapter_config.json: {resolved_adapter_path}"
        )
    adapter_model = PeftModel.from_pretrained(
        base_model,
        resolved_adapter_path,
        is_trainable=False,
        local_files_only=True,
    )
    adapter_model.eval()
    adapter_logits = adapter_model(**model_inputs).logits.detach().float().cpu()
    merge = getattr(adapter_model, "merge_and_unload", None)
    if not callable(merge):
        raise TypeError(f"adapter at {resolved_adapter_path} does not support merge_and_unload")
    merged_model = merge(safe_merge=True)
    merged_model.eval()
    merged_logits = merged_model(**model_inputs).logits.detach().float().cpu()
    if not torch.allclose(
        adapter_logits,
        merged_logits,
        atol=absolute_tolerance,
        rtol=relative_tolerance,
    ):
        maximum_error = (adapter_logits - merged_logits).abs().max().item()
        raise RuntimeError(
            "merged model logits do not match adapter logits "
            f"(max absolute error={maximum_error:.6g})"
        )
    audit = LoraMergeAudit(
        adapter_path=str(resolved_adapter_path),
        max_absolute_error=float((adapter_logits - merged_logits).abs().max().item()),
        argmax_matches=bool(
            torch.equal(adapter_logits.argmax(dim=-1), merged_logits.argmax(dim=-1))
        ),
    )
    if not audit.argmax_matches:
        raise RuntimeError("merged model and adapter model produce different argmax tokens")
    return merged_model, audit
