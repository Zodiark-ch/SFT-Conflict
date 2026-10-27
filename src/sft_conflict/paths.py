"""Machine-independent default locations of artifacts and the isolated lm-eval tools."""

from __future__ import annotations

import os
from pathlib import Path

ARTIFACT_ROOT_ENV = "SFT_CONFLICT_ARTIFACT_ROOT"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def default_artifact_root() -> str:
    """``$SFT_CONFLICT_ARTIFACT_ROOT``, otherwise ``artifacts/`` next to the repository."""
    return os.environ.get(ARTIFACT_ROOT_ENV) or str(REPOSITORY_ROOT.parent / "artifacts")


DEFAULT_ARTIFACT_ROOT = default_artifact_root()
DEFAULT_LM_EVAL_PYTHON = str(Path(DEFAULT_ARTIFACT_ROOT) / "envs" / "lm-eval" / "bin" / "python")
DEFAULT_LM_EVAL_HARNESS_PATH = str(Path(DEFAULT_ARTIFACT_ROOT) / "lm-harness")
