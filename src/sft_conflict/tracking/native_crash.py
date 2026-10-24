"""Native fatal-signal diagnostics; no hooks or CUDA sync in the training loop."""

from __future__ import annotations

import os
import sys
import warnings
from functools import lru_cache

import torch


@lru_cache(maxsize=1)
def enable_native_crash_diagnostics() -> bool:
    """Print native stacks to stderr on fatal signals, when supported by PyTorch.

    Python faulthandler alone cannot name crashing C++ autograd/NCCL frames.
    This does not catch/recover SIGSEGV or promise to prevent a native failure.
    Use SFT_CONFLICT_NATIVE_STACKS=0 to leave native signal handlers untouched.
    """
    if os.environ.get("SFT_CONFLICT_NATIVE_STACKS", "1") == "0":
        return False
    if not sys.platform.startswith("linux"):
        return False
    enable = getattr(torch._C, "_set_print_stack_traces_on_fatal_signal", None)
    if not callable(enable):
        warnings.warn(
            "PyTorch native fatal-signal stacks unavailable", RuntimeWarning, stacklevel=2
        )
        return False
    try:
        enable(True)
    except RuntimeError as error:
        warnings.warn(
            f"Native fatal-signal stacks unavailable: {error}", RuntimeWarning, stacklevel=2
        )
        return False
    print(
        f"Native fatal-signal stacks enabled: rank={os.environ.get('RANK', '0')}, "
        f"pid={os.getpid()}, torch={torch.__version__}",
        flush=True,
    )
    return True
