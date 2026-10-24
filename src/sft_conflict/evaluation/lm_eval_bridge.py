"""Isolated invocation of the pinned lm-evaluation-harness environment."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

TASK_CACHE_CHECK_SCRIPT = """
import json
import sys

from lm_eval.tasks import TaskManager, get_task_dict

tasks = json.loads(sys.argv[1])
get_task_dict(tasks, task_manager=TaskManager())
print(json.dumps(tasks))
""".strip()

IFEVAL_DEPENDENCIES = (
    "langdetect==1.0.9",
    "immutabledict==4.2.1",
    "nltk==3.10.3",
)


@dataclass(frozen=True)
class LmEvalRequest:
    python_executable: str
    model: str
    model_args: str
    tasks: tuple[str, ...]
    output_path: str
    harness_path: str = "/data1/chenhang/SFT_Conflict/artifacts/lm-harness"
    batch_size: str = "auto"
    num_fewshot: int | None = None
    limit: float | None = None
    seed: int = 42
    log_samples: bool = True
    check_integrity: bool = False
    write_out: bool = False
    use_cache: bool = True
    request_cache_root: str | None = None
    apply_chat_template: bool = False
    fewshot_as_multiturn: bool = False
    generation_kwargs: str | None = None
    auto_prepare: bool = False
    stream_output: bool = True


def _result_file(output_path: Path) -> Path:
    if output_path.is_file():
        return output_path
    candidates = sorted(output_path.rglob("*.json"), key=lambda path: path.stat().st_mtime)
    candidates = [path for path in candidates if not path.name.endswith(".metadata.json")]
    if not candidates:
        raise RuntimeError(f"lm-eval produced no JSON result under {output_path}")
    return candidates[-1]


def _offline_environment(harness_path: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    existing_python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        f"{harness_path}{os.pathsep}{existing_python_path}"
        if existing_python_path
        else str(harness_path)
    )
    return environment


def _online_environment(offline_environment: dict[str, str]) -> dict[str, str]:
    environment = offline_environment.copy()
    for key in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
        environment.pop(key, None)
    for key in (
        "http_proxy",
        "https_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "all_proxy",
    ):
        value = environment.get(key)
        if value and urlparse(value).hostname in {"127.0.0.1", "localhost"}:
            environment.pop(key, None)
    return environment


def _task_cache_check(
    request: LmEvalRequest,
    environment: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            request.python_executable,
            "-c",
            TASK_CACHE_CHECK_SCRIPT,
            json.dumps(request.tasks),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )


def _failure_summary(completed: subprocess.CompletedProcess[str]) -> str:
    lines = (completed.stderr or completed.stdout).strip().splitlines()
    return lines[-1] if lines else f"exit code {completed.returncode}"


def _stream_pipe(pipe, terminal, log_handle) -> None:
    while True:
        chunk = os.read(pipe.fileno(), 65_536)
        if not chunk:
            break
        log_handle.write(chunk)
        log_handle.flush()
        terminal.write(chunk.decode("utf-8", errors="replace"))
        terminal.flush()


def _log_tail(path: Path, maximum_bytes: int = 65_536) -> str:
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - maximum_bytes))
        return handle.read().decode("utf-8", errors="replace")


def _run_streaming(
    command: list[str],
    *,
    environment: dict[str, str],
    stdout_path: Path,
    stderr_path: Path,
) -> subprocess.CompletedProcess[str]:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("wb") as stdout_log, stderr_path.open("wb") as stderr_log:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
        )
        assert process.stdout is not None and process.stderr is not None
        threads = (
            threading.Thread(
                target=_stream_pipe,
                args=(process.stdout, sys.stdout, stdout_log),
                daemon=True,
            ),
            threading.Thread(
                target=_stream_pipe,
                args=(process.stderr, sys.stderr, stderr_log),
                daemon=True,
            ),
        )
        for thread in threads:
            thread.start()
        returncode = process.wait()
        for thread in threads:
            thread.join()
    return subprocess.CompletedProcess(
        command,
        returncode,
        stdout=_log_tail(stdout_path),
        stderr=_log_tail(stderr_path),
    )


def _prepare_task_dependencies(
    request: LmEvalRequest,
    offline_environment: dict[str, str],
) -> bool:
    if "ifeval" not in request.tasks:
        return False
    harness_path = Path(request.harness_path).expanduser().resolve()
    completed = subprocess.run(
        [
            request.python_executable,
            "-m",
            "pip",
            "install",
            "--cache-dir",
            str(harness_path.parent / "pip_cache"),
            *IFEVAL_DEPENDENCIES,
        ],
        check=False,
        capture_output=True,
        text=True,
        env=_online_environment(offline_environment),
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "lm-eval could not prepare the official IFEval dependencies: "
            + _failure_summary(completed)
        )
    frozen = subprocess.run(
        [request.python_executable, "-m", "pip", "freeze", "--all"],
        check=False,
        capture_output=True,
        text=True,
        env=offline_environment,
    )
    if frozen.returncode != 0:
        raise RuntimeError(
            "lm-eval could not refresh its environment lock: " + _failure_summary(frozen)
        )
    lock_path = harness_path.parent / "env-manifests" / "lm-eval-requirements.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_lock = lock_path.with_suffix(lock_path.suffix + ".tmp")
    temporary_lock.write_text(frozen.stdout, encoding="utf-8")
    os.replace(temporary_lock, lock_path)
    return True


def _ensure_task_cache(
    request: LmEvalRequest,
    offline_environment: dict[str, str],
) -> dict[str, Any]:
    print(
        f"Checking lm-eval task data cache for {len(request.tasks)} requested tasks...",
        flush=True,
    )
    offline_check = _task_cache_check(request, offline_environment)
    if offline_check.returncode == 0:
        print("lm-eval task data cache is ready.", flush=True)
        return {
            "offline_ready": True,
            "auto_prepared": False,
            "dependencies_auto_prepared": False,
        }
    if not request.auto_prepare:
        raise RuntimeError(
            "lm-eval task data or optional dependencies are incomplete. Set "
            "AUTO_PREPARE_CACHE=True in run.py to prepare them. " + _failure_summary(offline_check)
        )

    print("lm-eval task data cache is incomplete; preparing missing data...", flush=True)
    dependencies_auto_prepared = _prepare_task_dependencies(request, offline_environment)
    online_check = _task_cache_check(request, _online_environment(offline_environment))
    if online_check.returncode != 0:
        raise RuntimeError(
            "lm-eval could not prepare the selected task data cache: "
            + _failure_summary(online_check)
        )
    offline_verification = _task_cache_check(request, offline_environment)
    if offline_verification.returncode != 0:
        raise RuntimeError(
            "lm-eval task data remained unavailable after cache preparation: "
            + _failure_summary(offline_verification)
        )
    print("lm-eval task data cache preparation completed.", flush=True)
    return {
        "offline_ready": True,
        "auto_prepared": True,
        "dependencies_auto_prepared": dependencies_auto_prepared,
    }


def ensure_lm_eval_task_cache(request: LmEvalRequest) -> dict[str, Any]:
    harness_path = Path(request.harness_path).expanduser().resolve()
    if not (harness_path / "lm_eval" / "__init__.py").is_file():
        raise FileNotFoundError(f"lm-eval worktree is incomplete: {harness_path}")
    environment = _offline_environment(harness_path)
    registered_tasks = _registered_tasks(request, environment)
    unknown_tasks = set(request.tasks) - registered_tasks
    if unknown_tasks:
        raise ValueError(
            "lm-eval tasks are not registered in the pinned harness: "
            + ", ".join(sorted(unknown_tasks))
        )
    return _ensure_task_cache(request, environment)


def _registered_tasks(request: LmEvalRequest, environment: dict[str, str]) -> set[str]:
    script = (
        "import json; from lm_eval.tasks import TaskManager; "
        "print(json.dumps(sorted(TaskManager().all_tasks)))"
    )
    completed = subprocess.run(
        [request.python_executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"lm-eval task registry check failed: {completed.stderr.strip()}")
    try:
        return set(json.loads(completed.stdout.strip().splitlines()[-1]))
    except (IndexError, json.JSONDecodeError, TypeError) as exc:
        raise RuntimeError("lm-eval task registry returned invalid output") from exc


def _request_cache_path(
    request: LmEvalRequest,
    *,
    harness_revision: str,
    task_registry_sha256: str,
    output_path: Path,
) -> Path:
    identity = {
        "harness_revision": harness_revision,
        "task_registry_sha256": task_registry_sha256,
        "model": request.model,
        "model_args": request.model_args,
        "tasks": request.tasks,
        "num_fewshot": request.num_fewshot,
        "seed": request.seed,
        "apply_chat_template": request.apply_chat_template,
        "fewshot_as_multiturn": request.fewshot_as_multiturn,
        "generation_kwargs": request.generation_kwargs,
    }
    digest = hashlib.sha256(
        json.dumps(identity, ensure_ascii=True, sort_keys=True).encode()
    ).hexdigest()[:24]
    root = (
        Path(request.request_cache_root).expanduser().resolve()
        if request.request_cache_root is not None
        else output_path.parent / "request-cache"
    )
    root.mkdir(parents=True, exist_ok=True)
    return root / digest


def _task_registry_hash(harness_path: Path) -> str:
    task_root = harness_path / "lm_eval" / "tasks"
    digest = hashlib.sha256()
    files = sorted(
        path
        for path in task_root.rglob("*")
        if path.is_file() and path.suffix in {".py", ".yaml", ".yml"}
    )
    if not files:
        raise RuntimeError(f"lm-eval task definitions are missing under {task_root}")
    for path in files:
        digest.update(path.relative_to(task_root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _validate_result_metrics(payload: dict[str, Any], tasks: set[str]) -> None:
    for task_name in tasks:
        metrics = payload["results"].get(task_name)
        if not isinstance(metrics, dict) or not metrics:
            raise RuntimeError(f"lm-eval task '{task_name}' returned no metrics")
        for metric_name, value in metrics.items():
            if isinstance(value, float) and not math.isfinite(value):
                raise RuntimeError(
                    f"lm-eval task '{task_name}' returned non-finite metric {metric_name}={value}"
                )


def _resolve_completed_tasks(
    payload: dict[str, Any], requested_tasks: set[str]
) -> tuple[set[str], set[str]]:
    results = payload.get("results", {})
    group_subtasks = payload.get("group_subtasks", {})
    completed: set[str] = set()
    missing: set[str] = set()
    for task_name in requested_tasks:
        if task_name in results:
            completed.add(task_name)
            continue
        subtasks = group_subtasks.get(task_name)
        if not isinstance(subtasks, list) or not subtasks:
            missing.add(task_name)
            continue
        missing_subtasks = set(subtasks) - set(results)
        if missing_subtasks:
            missing.add(task_name)
            continue
        completed.update(subtasks)
    return completed, missing


def run_lm_eval(
    request: LmEvalRequest,
    *,
    prepared_task_cache: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not request.tasks:
        raise ValueError("lm-eval requires at least one task")
    output_path = Path(request.output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        request.python_executable,
        "-m",
        "lm_eval",
        "--model",
        request.model,
        "--model_args",
        request.model_args,
        "--tasks",
        ",".join(request.tasks),
        "--batch_size",
        request.batch_size,
        "--output_path",
        str(output_path),
        "--seed",
        str(request.seed),
    ]
    if request.num_fewshot is not None:
        command.extend(["--num_fewshot", str(request.num_fewshot)])
    if request.limit is not None:
        command.extend(["--limit", str(request.limit)])
    if request.log_samples:
        command.append("--log_samples")
    if request.check_integrity:
        command.append("--check_integrity")
    if request.apply_chat_template:
        command.append("--apply_chat_template")
    if request.fewshot_as_multiturn:
        command.append("--fewshot_as_multiturn")
    if request.generation_kwargs is not None:
        command.extend(["--gen_kwargs", request.generation_kwargs])

    harness_path = Path(request.harness_path).expanduser().resolve()
    if not (harness_path / "lm_eval" / "__init__.py").is_file():
        raise FileNotFoundError(f"lm-eval worktree is incomplete: {harness_path}")
    environment = _offline_environment(harness_path)
    revision = subprocess.run(
        ["git", "-C", str(harness_path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()
    task_registry_sha256 = _task_registry_hash(harness_path)
    registered_tasks = _registered_tasks(request, environment)
    unknown_tasks = set(request.tasks) - registered_tasks
    if unknown_tasks:
        raise ValueError(
            "lm-eval tasks are not registered in the pinned harness: "
            + ", ".join(sorted(unknown_tasks))
        )
    if prepared_task_cache is not None and prepared_task_cache.get("offline_ready") is not True:
        raise ValueError("prepared lm-eval task cache status is not offline-ready")
    task_cache = prepared_task_cache or _ensure_task_cache(request, environment)
    request_cache_path = None
    if request.use_cache:
        request_cache_path = _request_cache_path(
            request,
            harness_revision=revision,
            task_registry_sha256=task_registry_sha256,
            output_path=output_path,
        )
        command.extend(["--use_cache", str(request_cache_path)])
    if request.write_out:
        command.append("--write_out")
    metadata_path = output_path.with_name(output_path.name + ".metadata.json")
    stdout_path = output_path.with_name(output_path.name + ".stdout.log")
    stderr_path = output_path.with_name(output_path.name + ".stderr.log")
    if request.stream_output:
        completed = _run_streaming(
            command,
            environment=environment,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )
    else:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
    metadata = {
        "request": asdict(request),
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "stdout_log": str(stdout_path) if request.stream_output else None,
        "stderr_log": str(stderr_path) if request.stream_output else None,
        "harness_path": str(harness_path),
        "harness_revision": revision or None,
        "task_registry_sha256": task_registry_sha256,
        "registered_tasks_verified": sorted(request.tasks),
        "task_cache": task_cache,
        "request_cache_path": str(request_cache_path) if request_cache_path else None,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"lm-eval failed with exit code {completed.returncode}: "
            f"{_failure_summary(completed)}; see {metadata_path}"
        )

    result_path = _result_file(output_path)
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    actual_tasks = set(payload.get("results", {}))
    requested_tasks = set(request.tasks)
    completed_tasks, missing_tasks = _resolve_completed_tasks(payload, requested_tasks)
    if missing_tasks:
        raise RuntimeError(
            "lm-eval result is missing requested tasks: " + ", ".join(sorted(missing_tasks))
        )
    if not actual_tasks:
        raise RuntimeError("lm-eval returned an empty task result set")
    _validate_result_metrics(payload, completed_tasks)
    return {
        "result_path": str(result_path),
        "metadata_path": str(metadata_path),
        "tasks": sorted(actual_tasks),
        "requested_tasks": sorted(requested_tasks),
        "task_cache": task_cache,
        "harness_revision": revision or None,
        "results": payload,
    }
