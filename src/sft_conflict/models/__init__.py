"""Model loading and Unembedding adapters."""

from .factory import (
    ModelAudit,
    ModelBundle,
    audit_existing_training_model,
    configure_model_for_training,
    enable_gradient_checkpointing,
    load_model_bundle,
    load_reference_model,
    torch_dtype_for_precision,
)
from .unembedding import (
    UnembeddingForwardAudit,
    UnembeddingInputCapture,
    audit_unembedding_forward,
    get_unembedding,
    has_tied_input_output_embeddings,
)

__all__ = [
    "ModelAudit",
    "ModelBundle",
    "audit_existing_training_model",
    "UnembeddingForwardAudit",
    "UnembeddingInputCapture",
    "audit_unembedding_forward",
    "configure_model_for_training",
    "enable_gradient_checkpointing",
    "get_unembedding",
    "has_tied_input_output_embeddings",
    "load_model_bundle",
    "load_reference_model",
    "torch_dtype_for_precision",
]
