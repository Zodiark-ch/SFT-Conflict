"""Dataset identities, split contracts, and deterministic text adapters."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from functools import partial
from typing import Any

from sft_conflict.config import DatasetConfig


class DatasetRole(str, Enum):
    TARGET_TRAIN = "target_train"
    TARGET_VALIDATION = "target_validation"
    TARGET_EVALUATION = "target_evaluation"
    ANCHOR_TRAIN = "anchor_train"
    CALIBRATION = "calibration"
    GENERAL_EVALUATION = "general_evaluation"


@dataclass(frozen=True)
class RenderedExample:
    prompt: str
    response: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class DatasetAdapter:
    name: str
    template_version: str
    render: Callable[[Mapping[str, Any]], RenderedExample]


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    dataset_id: str
    revision: str
    adapter: str
    config_name: str | None
    role_splits: Mapping[DatasetRole, str]
    role_data_files: Mapping[DatasetRole, str] = field(default_factory=dict)
    role_selections: Mapping[DatasetRole, str] = field(default_factory=dict)
    stratify_regression: bool = False
    license_name: str | None = None
    lineage: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.key or not self.dataset_id or not self.adapter:
            raise ValueError("dataset key, ID, and adapter must be non-empty")
        if not self.revision or self.revision in {"main", "master", "latest"}:
            raise ValueError(f"dataset '{self.key}' requires an immutable revision")

    def config_for(
        self,
        role: DatasetRole,
        *,
        max_samples: int | None = None,
    ) -> DatasetConfig:
        try:
            split = self.role_splits[role]
        except KeyError as exc:
            raise ValueError(f"dataset '{self.key}' does not support role '{role.value}'") from exc
        return DatasetConfig(
            source="hf",
            path=self.dataset_id,
            split=split,
            revision=self.revision,
            name=self.config_name,
            data_file=self.role_data_files.get(role),
            adapter=self.adapter,
            max_samples=max_samples,
            selection=self.role_selections.get(role, "all"),
            selection_seed=0,
            validation_fraction=0.1,
            stratify_column=(
                "label" if self.role_selections.get(role) in {"train_90", "validation_10"} else None
            ),
            stratify_regression=self.stratify_regression,
        )


class DatasetRegistry:
    def __init__(self) -> None:
        self._adapters: dict[str, DatasetAdapter] = {}
        self._datasets: dict[str, DatasetSpec] = {}

    def register_adapter(self, adapter: DatasetAdapter) -> None:
        if adapter.name in self._adapters:
            raise ValueError(f"dataset adapter already registered: {adapter.name}")
        self._adapters[adapter.name] = adapter

    def register_dataset(self, spec: DatasetSpec) -> None:
        if spec.key in self._datasets:
            raise ValueError(f"dataset already registered: {spec.key}")
        if spec.adapter not in self._adapters and spec.adapter != "tokenized":
            raise ValueError(f"dataset '{spec.key}' references unknown adapter '{spec.adapter}'")
        if not spec.role_splits:
            raise ValueError(f"dataset '{spec.key}' must declare at least one role/split")
        self._datasets[spec.key] = spec

    def adapter(self, name: str) -> DatasetAdapter:
        try:
            return self._adapters[name]
        except KeyError as exc:
            choices = ", ".join(sorted(self._adapters))
            raise ValueError(f"unknown dataset adapter '{name}'; available: {choices}") from exc

    def dataset(self, key: str) -> DatasetSpec:
        try:
            return self._datasets[key]
        except KeyError as exc:
            choices = ", ".join(sorted(self._datasets))
            raise ValueError(f"unknown dataset '{key}'; available: {choices}") from exc

    def render(self, adapter_name: str, row: Mapping[str, Any]) -> RenderedExample:
        return self.adapter(adapter_name).render(row)

    def matching_dataset(
        self,
        *,
        dataset_id: str,
        revision: str | None,
        config_name: str | None,
    ) -> DatasetSpec | None:
        matches = [
            spec
            for spec in self._datasets.values()
            if spec.dataset_id == dataset_id
            and spec.revision == revision
            and spec.config_name == config_name
        ]
        if len(matches) > 1:
            raise ValueError(f"multiple registry entries match {dataset_id}@{revision}")
        return matches[0] if matches else None

    @property
    def dataset_keys(self) -> tuple[str, ...]:
        return tuple(sorted(self._datasets))

    @property
    def adapter_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._adapters))


def _prompt_response(row: Mapping[str, Any]) -> RenderedExample:
    return RenderedExample(
        prompt=str(row["prompt"]),
        response=str(row["response"]),
        metadata={},
    )


def _gsm8k(row: Mapping[str, Any]) -> RenderedExample:
    return RenderedExample(
        prompt=f"Question: {row['question']}\nAnswer:",
        response=f" {row['answer']}",
        metadata={"task_name": "gsm8k"},
    )


def _smart_flan(row: Mapping[str, Any]) -> RenderedExample:
    prompt = row.get("inputs", row.get("input", row.get("prompt")))
    response = row.get("targets", row.get("target", row.get("response")))
    if prompt is None or response is None:
        raise KeyError("SMART FLAN rows require inputs/targets or prompt/response fields")
    prompt_text = str(prompt)
    response_text = str(response)
    if response_text and not response_text[0].isspace() and not prompt_text[-1:].isspace():
        response_text = " " + response_text
    metadata = {
        key: row[key]
        for key in ("task", "task_name", "task_source", "source", "template_type")
        if key in row
    }
    metadata["task_metadata_available"] = any(
        key in row for key in ("task", "task_name", "task_source", "template_type")
    )
    metadata.setdefault("task_name", "smart_flan")
    return RenderedExample(prompt=prompt_text, response=response_text, metadata=metadata)


def _lm_eval_only(row: Mapping[str, Any]) -> RenderedExample:
    del row
    raise ValueError("lm_eval_only datasets must be evaluated through the pinned harness")


def build_default_registry() -> DatasetRegistry:
    from .code import (
        render_code_bagel_hermes,
        render_codealpaca,
        render_codexglue_token_completion,
        render_humaneval,
        render_infinity_instruct_code,
    )
    from .glue import GLUE_TASK_SPECS, render_glue
    from .logic import render_alpaca_eval, render_evol_instruct, render_logiqa2

    registry = DatasetRegistry()
    registry.register_adapter(
        DatasetAdapter("prompt_response", "prompt-response-v1", _prompt_response)
    )
    for task_name in GLUE_TASK_SPECS:
        registry.register_adapter(
            DatasetAdapter(
                task_name,
                f"glue-{task_name}-decoder-v1",
                partial(render_glue, task_name=task_name),
            )
        )
    registry.register_adapter(DatasetAdapter("gsm8k", "gsm8k-cot-v1", _gsm8k))
    registry.register_adapter(
        DatasetAdapter("evol_instruct", "wizardlm-train-freeform-v1", render_evol_instruct)
    )
    registry.register_adapter(
        DatasetAdapter("alpaca_eval", "dare-wizardlm-chat-v1", render_alpaca_eval)
    )
    registry.register_adapter(DatasetAdapter("logiqa2", "logiqa2-zero-shot-mrc-v1", render_logiqa2))
    registry.register_adapter(DatasetAdapter("codealpaca", "lori-codealpaca-v1", render_codealpaca))
    registry.register_adapter(
        DatasetAdapter(
            "code_bagel_hermes",
            "dift-code-bagel-alpaca-v1",
            render_code_bagel_hermes,
        )
    )
    registry.register_adapter(
        DatasetAdapter(
            "infinity_instruct_code",
            "dift-infinity-query-response-v1",
            render_infinity_instruct_code,
        )
    )
    registry.register_adapter(
        DatasetAdapter("humaneval", "lori-humaneval-completion-v1", render_humaneval)
    )
    registry.register_adapter(
        DatasetAdapter(
            "codexglue_token_completion",
            "codexglue-pretokenized-v1",
            render_codexglue_token_completion,
        )
    )
    registry.register_adapter(DatasetAdapter("smart_flan", "smart-flan-v3", _smart_flan))
    registry.register_adapter(DatasetAdapter("lm_eval_only", "lm-eval-0.4.9", _lm_eval_only))
    glue_revision = "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c"
    for task_name, task_spec in GLUE_TASK_SPECS.items():
        evaluation_split = "validation_matched" if task_name == "mnli" else "validation"
        registry.register_dataset(
            DatasetSpec(
                key=task_name,
                dataset_id="nyu-mll/glue",
                revision=glue_revision,
                adapter=task_name,
                config_name=task_name,
                role_splits={
                    DatasetRole.TARGET_TRAIN: "train",
                    DatasetRole.TARGET_VALIDATION: "train",
                    DatasetRole.TARGET_EVALUATION: evaluation_split,
                },
                role_selections={
                    DatasetRole.TARGET_TRAIN: "train_90",
                    DatasetRole.TARGET_VALIDATION: "validation_10",
                },
                stratify_regression=task_spec.regression,
                license_name=f"GLUE/{task_name.upper()}",
            )
        )
    registry.register_dataset(
        DatasetSpec(
            key="gsm8k",
            dataset_id="openai/gsm8k",
            revision="740312add88f781978c0658806c59bc2815b9866",
            adapter="gsm8k",
            config_name="main",
            role_splits={
                DatasetRole.TARGET_TRAIN: "train",
                DatasetRole.TARGET_EVALUATION: "test",
            },
            license_name="MIT",
        )
    )
    registry.register_dataset(
        DatasetSpec(
            key="evol_instruct",
            dataset_id="WizardLMTeam/WizardLM_evol_instruct_70k",
            revision="16b48bd8eecad79d4f42e1ab641db317e1b27443",
            adapter="evol_instruct",
            config_name=None,
            role_splits={DatasetRole.TARGET_TRAIN: "train"},
            license_name="MIT",
            lineage={
                "training_protocol": "WizardLM src/train_freeform.py single-turn prompt",
                "evaluation_protocol": "DARE WizardLM prompt and AlpacaEval",
                "row_count": 70_000,
            },
        )
    )
    registry.register_dataset(
        DatasetSpec(
            key="alpaca_eval",
            dataset_id="tatsu-lab/alpaca_eval",
            revision="2edc6fad8be6b14ea7230aabfd08188da6b8b814",
            adapter="alpaca_eval",
            config_name="alpaca_eval",
            role_splits={DatasetRole.TARGET_EVALUATION: "eval"},
            role_data_files={DatasetRole.TARGET_EVALUATION: "alpaca_eval.json"},
            license_name="CC-BY-NC-4.0",
            lineage={
                "reference_protocol": "DARE chatgpt_fn",
                "row_count": 805,
                "official_tool_revision": "f19c323d8309d0d6f306bd26597db44fc6c62d57",
            },
        )
    )
    registry.register_dataset(
        DatasetSpec(
            key="logiqa2",
            dataset_id="datatune/LogiQA2.0",
            revision="edbf6440d36a03f715c0c9d479f421f2ece87d6a",
            adapter="logiqa2",
            config_name=None,
            role_splits={
                DatasetRole.TARGET_TRAIN: "train",
                DatasetRole.TARGET_VALIDATION: "validation",
                DatasetRole.TARGET_EVALUATION: "test",
            },
            role_data_files={
                DatasetRole.TARGET_TRAIN: "MRC/train.txt",
                DatasetRole.TARGET_VALIDATION: "MRC/dev.txt",
                DatasetRole.TARGET_EVALUATION: "MRC/test.txt",
            },
            license_name="MIT",
            lineage={
                "reference_protocol": "official English MRC; DiFT uses LogiQA2 0-shot Acc",
                "metric": "accuracy",
                "row_counts": {"train": 12_567, "validation": 1_569, "test": 1_572},
            },
        )
    )
    registry.register_dataset(
        DatasetSpec(
            key="codealpaca",
            dataset_id="sahil2801/CodeAlpaca-20k",
            revision="152bb5e9a29651266b018106053980070a0521a1",
            adapter="codealpaca",
            config_name=None,
            role_splits={DatasetRole.TARGET_TRAIN: "train"},
            license_name="CC-BY-4.0",
            lineage={"paper_protocol": "LoRI", "reported_training_epochs": 2},
        )
    )
    registry.register_dataset(
        DatasetSpec(
            key="code_bagel_hermes",
            dataset_id="rombodawg/code_bagel_hermes-2.5",
            revision="01708cfd32ec7fc6292ab374d3d7d5fdef626e59",
            adapter="code_bagel_hermes",
            config_name=None,
            role_splits={DatasetRole.TARGET_TRAIN: "train"},
            role_selections={DatasetRole.TARGET_TRAIN: "sample_n"},
            license_name="Apache-2.0",
            lineage={"paper_protocol": "DiFT", "sample_budget": 20_000},
        )
    )
    registry.register_dataset(
        DatasetSpec(
            key="infinity_instruct_code",
            dataset_id="BAAI/Infinity-Instruct",
            revision="bddc39a8feadbd679c30623197f4e736b7e75b48",
            adapter="infinity_instruct_code",
            config_name="7M",
            role_splits={DatasetRole.TARGET_TRAIN: "train"},
            role_selections={DatasetRole.TARGET_TRAIN: "infinity_code_sample_n"},
            license_name="CC-BY-SA-4.0",
            lineage={
                "paper_protocol": "DiFT",
                "sample_budget": 20_000,
                "access": "gated; Hugging Face terms must be accepted",
            },
        )
    )
    registry.register_dataset(
        DatasetSpec(
            key="humaneval",
            dataset_id="openai/openai_humaneval",
            revision="7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544",
            adapter="humaneval",
            config_name="openai_humaneval",
            role_splits={DatasetRole.TARGET_EVALUATION: "test"},
            license_name="MIT",
            lineage={"paper_protocol": "LoRI", "pass_at_k": [1, 5, 10]},
        )
    )
    for language in ("python", "java"):
        registry.register_dataset(
            DatasetSpec(
                key=f"codexglue_{language}",
                dataset_id="google/code_x_glue_cc_code_completion_token",
                revision="54bdbd7ee7e40c7f3c68ba5a5a20ab0b2d38d5b2",
                adapter="codexglue_token_completion",
                config_name=language,
                role_splits={DatasetRole.TARGET_EVALUATION: "test"},
                license_name="C-UDA",
                lineage={
                    "paper_protocol": "DiFT",
                    "official_metric": "token_accuracy",
                    "language": language,
                },
            )
        )
    registry.register_dataset(
        DatasetSpec(
            key="smart_flan",
            dataset_id=("kowndinya23/flan2022-4096-1840-tasks-logdet-400000-instances-logdet"),
            revision="6ca74ceaa9a6e284ff540aa51b059d5421756f67",
            adapter="smart_flan",
            config_name=None,
            role_splits={DatasetRole.ANCHOR_TRAIN: "train"},
            lineage={
                "source_kind": "external_immutable_mixture",
                "smart_repository_url": "https://github.com/kowndinya-renduchintala/SMART",
                "local_snapshot_hash_algorithm": "sha256(sorted sha256(file bytes))",
                "local_snapshot_sha256": (
                    "7ea5eeaf49419c64276f416d581a6e870f30d947b9eab77e91fc2fb4665b6905"
                ),
                "mixture_rebuilt_by_project": False,
                "task_selection_method": "logdet",
                "selected_task_count": 1840,
                "total_instance_budget": 400000,
                "instance_ordering_method": "logdet",
                "upstream_random_seed": 23,
                "published_schema_fields": ["prompt", "response"],
                "task_metadata_available": False,
                "task_marginal_gain_table": {"status": "unavailable_in_published_artifact"},
                "task_budget_table": {"status": "unavailable_in_published_artifact"},
                "task_template_budget_table": {"status": "unavailable_in_published_artifact"},
                "ordering_artifact_hash": {"status": "unavailable_in_published_artifact"},
                "budget_conservation": {
                    "expected_published_train_examples": 400000,
                    "verified_from_dataset_card": True,
                },
                "transform_rules": [
                    "preserve published order",
                    "prepend response whitespace when absent for token boundary safety",
                    "mask boundary tokens whose response-side fragment is only whitespace",
                    "drop rows with no supervised token after truncation",
                    "do not reconstruct missing task metadata",
                ],
            },
        )
    )
    registry.register_dataset(
        DatasetSpec(
            key="hellaswag",
            dataset_id="Rowan/hellaswag",
            revision="218ec52e09a7e7462a5400043bb9a69a41d06b76",
            adapter="lm_eval_only",
            config_name=None,
            role_splits={DatasetRole.GENERAL_EVALUATION: "validation"},
        )
    )
    return registry


DEFAULT_DATASET_REGISTRY = build_default_registry()
