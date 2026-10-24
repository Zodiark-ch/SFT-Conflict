"""Code-generation metrics and opt-in local execution helpers."""

from __future__ import annotations

import json
import math
import os
import resource
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

CODEXGLUE_IGNORED_TOKENS = frozenset({"<s>", "</s>", "<EOL>", "<pad>"})


def estimate_pass_at_k(num_samples: int, num_correct: int, k: int) -> float:
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if not 0 <= num_correct <= num_samples:
        raise ValueError("num_correct must be between zero and num_samples")
    if not 1 <= k <= num_samples:
        raise ValueError("k must be between one and num_samples")
    if num_samples - num_correct < k:
        return 1.0
    product = 1.0
    for value in range(num_samples - num_correct + 1, num_samples + 1):
        product *= 1.0 - k / value
    return 1.0 - product


def aggregate_pass_at_k(
    correct_counts: Sequence[int],
    *,
    samples_per_problem: int,
    ks: Sequence[int] = (1, 5, 10),
) -> dict[str, float]:
    if not correct_counts:
        raise ValueError("pass@k requires at least one problem")
    metrics: dict[str, float] = {}
    for k in ks:
        if k > samples_per_problem:
            raise ValueError(f"pass@{k} requires at least {k} samples per problem")
        metrics[f"pass@{k}"] = sum(
            estimate_pass_at_k(samples_per_problem, correct, k) for correct in correct_counts
        ) / len(correct_counts)
    return metrics


def codexglue_token_accuracy(
    predictions: Sequence[Sequence[str]],
    references: Sequence[Sequence[str]],
) -> dict[str, float | int]:
    if not predictions or len(predictions) != len(references):
        raise ValueError("CodeXGLUE predictions and references must be aligned and non-empty")
    correct = 0
    total = 0
    for predicted_tokens, reference_tokens in zip(predictions, references, strict=True):
        if len(predicted_tokens) != len(reference_tokens):
            raise ValueError("CodeXGLUE prediction and reference token lengths must match")
        for prediction, reference in zip(predicted_tokens, reference_tokens, strict=True):
            if reference in CODEXGLUE_IGNORED_TOKENS:
                continue
            total += 1
            correct += int(prediction == reference)
    if total == 0:
        raise ValueError("CodeXGLUE evaluation found no scored tokens")
    return {"token_accuracy": correct / total, "correct_tokens": correct, "scored_tokens": total}


_WORKER_SOURCE = r"""
import builtins
import json
import os
import socket
import subprocess
import sys

payload = json.loads(sys.stdin.read())

def blocked(*args, **kwargs):
    raise PermissionError("operation disabled by HumanEval reliability guard")

os.system = blocked
os.remove = blocked
os.unlink = blocked
os.rename = blocked
os.replace = blocked
os.rmdir = blocked
os.removedirs = blocked
os.fork = blocked
os.forkpty = blocked
os.kill = blocked
os.killpg = blocked
subprocess.Popen = blocked
socket.socket = blocked
builtins.open = blocked

try:
    namespace = {}
    exec(payload["program"], namespace)
except BaseException as exc:
    print(json.dumps({"passed": False, "result": f"failed: {type(exc).__name__}: {exc}"}))
else:
    print(json.dumps({"passed": True, "result": "passed"}))
"""


def _execution_limits(timeout_seconds: float, maximum_memory_bytes: int) -> None:
    cpu_seconds = max(1, math.ceil(timeout_seconds))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))
    resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
    resource.setrlimit(resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1_048_576, 1_048_576))
    resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))


def execute_python_program(
    program: str,
    *,
    allow_code_execution: bool,
    timeout_seconds: float = 3.0,
    maximum_memory_bytes: int = 1_073_741_824,
) -> dict[str, Any]:
    if not allow_code_execution:
        raise PermissionError(
            "HumanEval executes model-generated Python. Set ALLOW_CODE_EXECUTION=True "
            "only after accepting the local execution risk."
        )
    if timeout_seconds <= 0 or maximum_memory_bytes <= 0:
        raise ValueError("HumanEval execution limits must be positive")
    unshare = shutil.which("unshare")
    if unshare is None:
        raise RuntimeError("HumanEval requires the Linux unshare executable")
    command = [
        unshare,
        "--user",
        "--map-root-user",
        "--net",
        "--mount",
        "--pid",
        "--mount-proc",
        "--fork",
        sys.executable,
        "-I",
        "-S",
        "-c",
        _WORKER_SOURCE,
    ]
    environment = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "PYTHONHASHSEED": "0",
        "OMP_NUM_THREADS": "1",
    }
    with tempfile.TemporaryDirectory(prefix="sft-conflict-humaneval-") as directory:
        try:
            completed = subprocess.run(
                command,
                input=json.dumps({"program": program}),
                text=True,
                capture_output=True,
                cwd=Path(directory),
                env=environment,
                timeout=timeout_seconds + 2.0,
                check=False,
                preexec_fn=lambda: _execution_limits(timeout_seconds, maximum_memory_bytes),
            )
        except subprocess.TimeoutExpired:
            return {"passed": False, "result": "timed out"}
    output_lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not output_lines:
        detail = completed.stderr.strip() or f"worker exited {completed.returncode}"
        return {"passed": False, "result": detail}
    try:
        result = json.loads(output_lines[-1])
    except json.JSONDecodeError:
        return {"passed": False, "result": "invalid worker response"}
    return {"passed": bool(result["passed"]), "result": str(result["result"])}
