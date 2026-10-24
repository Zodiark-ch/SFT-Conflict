"""Read model-only DCP tensors on CPU; never create or restore an optimizer."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import torch
import torch.distributed as distributed
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.metadata import TensorStorageMetadata
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.modeling_utils import no_init_weights


def load_dcp_model_for_diagnostics(checkpoint: str | Path):
    """Single-process FP32 evaluation of canonical SHARDED_STATE_DICT weights.

    Source BF16 tensors are cast, not reconstructed to their pre-quantized values.
    All model tensors are required, and optimizer/RNG/trainer state is not loaded.
    Only use trusted local checkpoints (DCP metadata uses Python serialization).
    """
    if distributed.is_initialized():
        raise RuntimeError("model-only CPU diagnostic loader requires a single process")
    source = Path(checkpoint).expanduser().resolve()
    reader = dcp.FileSystemReader(source / "pytorch_model_fsdp_0")
    metadata = reader.read_metadata()
    weights = {}
    dtypes: Counter[str] = Counter()
    for name, value in metadata.state_dict_metadata.items():
        if not name.startswith("model.") or not isinstance(value, TensorStorageMetadata):
            raise ValueError(f"unexpected non-model DCP entry: {name}")
        dtype = value.properties.dtype
        dtypes[str(dtype)] += 1
        weights[name.removeprefix("model.")] = torch.empty(
            value.size, dtype=torch.float32 if dtype.is_floating_point else dtype, device="cpu"
        )
    config = AutoConfig.from_pretrained(source, local_files_only=True)
    config.use_cache = False
    # RoPE and other nonpersistent buffers are absent from state_dict. Construct
    # them on CPU, not meta; skip random parameter initialization before assign.
    with torch.device("cpu"), no_init_weights():
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float32)
    if weights.keys() != model.state_dict().keys():
        raise ValueError("DCP keys do not exactly match the configured diagnostic model")
    dcp.load({"model": weights}, storage_reader=reader)
    model.load_state_dict(weights, strict=True, assign=True)
    model.requires_grad_(False)
    model.eval()
    return model, {"tensor_count": len(weights), "source_tensor_dtypes": dict(dtypes)}
