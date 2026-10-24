"""Pinned subprocess bridge to the official AlpacaEval evaluator."""

from __future__ import annotations

import csv
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

ALPACA_EVAL_VERSION = "0.6.6"
ALPACA_EVAL_SOURCE_REVISION = "f19c323d8309d0d6f306bd26597db44fc6c62d57"
ALPACA_EVAL_REPOSITORY = "https://github.com/tatsu-lab/alpaca_eval.git"
ALPACA_EVAL_ANNOTATORS = (
    "chatgpt_fn",
    "weighted_alpaca_eval_gpt4_turbo",
)


class AlpacaEvalBridgeError(RuntimeError):
    """Raised when the isolated official AlpacaEval workflow cannot run."""


def _environment_paths(artifact_root: str | Path) -> tuple[Path, Path, Path]:
    root = Path(artifact_root).expanduser().resolve()
    environment = root / "envs" / "alpaca-eval"
    executable = environment / "bin" / "alpaca_eval"
    marker = environment / ".sft-conflict-source-revision"
    return environment, executable, marker


def _online_environment(*, include_openai_api_key: bool) -> dict[str, str]:
    environment = os.environ.copy()
    if not include_openai_api_key:
        environment.pop("OPENAI_API_KEY", None)
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


def ensure_alpaca_eval_environment(
    artifact_root: str | Path,
    *,
    auto_prepare: bool,
) -> Path:
    environment, executable, marker = _environment_paths(artifact_root)
    if (
        executable.is_file()
        and marker.is_file()
        and marker.read_text(encoding="utf-8").strip() == ALPACA_EVAL_SOURCE_REVISION
    ):
        return executable
    if not auto_prepare:
        raise AlpacaEvalBridgeError(
            "The pinned AlpacaEval environment is missing. Set AUTO_PREPARE_CACHE=True "
            "in run.py once to install the official evaluator in artifacts/envs/alpaca-eval."
        )

    installation_environment = _online_environment(include_openai_api_key=False)
    environment.parent.mkdir(parents=True, exist_ok=True)
    python = environment / "bin" / "python"
    if not python.is_file():
        subprocess.run(
            [sys.executable, "-m", "venv", str(environment)],
            check=True,
            env=installation_environment,
        )
    package = f"alpaca-eval @ git+{ALPACA_EVAL_REPOSITORY}@{ALPACA_EVAL_SOURCE_REVISION}"
    subprocess.run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--cache-dir",
            str(Path(artifact_root).expanduser().resolve() / "pip_cache"),
            package,
        ],
        check=True,
        env=installation_environment,
    )
    subprocess.run(
        [str(python), "-c", "import alpaca_eval"],
        check=True,
        env=installation_environment,
    )
    if not executable.is_file():
        raise AlpacaEvalBridgeError("AlpacaEval installed without its official CLI executable")
    marker.write_text(ALPACA_EVAL_SOURCE_REVISION + "\n", encoding="utf-8")
    lock_path = (
        Path(artifact_root).expanduser().resolve()
        / "env-manifests"
        / "alpaca-eval-requirements.lock"
    )
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [str(python), "-m", "pip", "freeze", "--all"],
        check=True,
        capture_output=True,
        text=True,
        env=installation_environment,
    )
    lock_path.write_text(completed.stdout, encoding="utf-8")
    return executable


def alpaca_eval_command(
    executable: str | Path,
    model_outputs: str | Path,
    *,
    annotators_config: str,
    generator_name: str,
    output_dir: str | Path,
) -> list[str]:
    if annotators_config not in ALPACA_EVAL_ANNOTATORS:
        raise ValueError(
            f"annotators_config must be one of {ALPACA_EVAL_ANNOTATORS}, got {annotators_config!r}"
        )
    return [
        str(executable),
        "--model_outputs",
        str(Path(model_outputs).expanduser().resolve()),
        "--annotators_config",
        annotators_config,
        "--name",
        generator_name,
        "--output_path",
        str(Path(output_dir).expanduser().resolve()),
    ]


def write_alpaca_eval_scoring_instructions(
    model_outputs: str | Path,
    *,
    artifact_root: str | Path,
    generator_name: str,
    output_dir: str | Path,
) -> str:
    _, executable, _ = _environment_paths(artifact_root)
    commands = [
        alpaca_eval_command(
            executable,
            model_outputs,
            annotators_config=annotator,
            generator_name=generator_name,
            output_dir=Path(output_dir) / annotator,
        )
        for annotator in ALPACA_EVAL_ANNOTATORS
    ]
    destination = Path(output_dir).expanduser().resolve() / "scoring-commands.sh"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n\n"
        "# Requires OPENAI_API_KEY in this shell. The first command reproduces DARE;\n"
        "# the second runs the current AlpacaEval 2.0 length-controlled protocol.\n"
        + "\n".join(" ".join(shlex.quote(part) for part in command) for command in commands)
        + "\n",
        encoding="utf-8",
    )
    destination.chmod(0o755)
    return str(destination)


def _read_leaderboard(output_dir: Path, generator_name: str) -> dict[str, str]:
    leaderboard_paths = sorted(output_dir.rglob("leaderboard.csv"))
    if not leaderboard_paths:
        raise AlpacaEvalBridgeError(
            f"official AlpacaEval completed without a leaderboard.csv under {output_dir}"
        )
    with leaderboard_paths[-1].open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise AlpacaEvalBridgeError(f"AlpacaEval leaderboard is empty: {leaderboard_paths[-1]}")
    model_rows = [
        row
        for row in rows
        if generator_name in {str(value) for value in row.values() if value is not None}
    ]
    return model_rows[-1] if model_rows else rows[-1]


def run_official_alpaca_eval(
    model_outputs: str | Path,
    *,
    artifact_root: str | Path,
    annotators_config: str,
    generator_name: str,
    output_dir: str | Path,
    auto_prepare: bool,
) -> dict[str, object]:
    if not os.environ.get("OPENAI_API_KEY"):
        raise AlpacaEvalBridgeError(
            "Official AlpacaEval judging requires OPENAI_API_KEY in the process environment. "
            "Export it in the terminal before running; do not put the key in run.py."
        )
    executable = ensure_alpaca_eval_environment(
        artifact_root,
        auto_prepare=auto_prepare,
    )
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    command = alpaca_eval_command(
        executable,
        model_outputs,
        annotators_config=annotators_config,
        generator_name=generator_name,
        output_dir=destination,
    )
    subprocess.run(
        command,
        check=True,
        env=_online_environment(include_openai_api_key=True),
    )
    result: dict[str, object] = {
        "tool": "tatsu-lab/alpaca_eval",
        "version": ALPACA_EVAL_VERSION,
        "source_revision": ALPACA_EVAL_SOURCE_REVISION,
        "annotators_config": annotators_config,
        "generator": generator_name,
        "leaderboard": _read_leaderboard(destination, generator_name),
    }
    result_path = destination / "sft-conflict-official-result.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result["result_path"] = str(result_path)
    return result
