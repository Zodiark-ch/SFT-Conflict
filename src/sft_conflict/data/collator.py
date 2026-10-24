"""Collators that preserve metadata outside model keyword arguments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


@dataclass(frozen=True)
class CausalLMCollator:
    pad_token_id: int
    pad_to_multiple_of: int | None = 8

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        if not examples:
            raise ValueError("cannot collate an empty token batch")
        maximum_length = max(len(example["input_ids"]) for example in examples)
        if self.pad_to_multiple_of:
            multiple = self.pad_to_multiple_of
            maximum_length = ((maximum_length + multiple - 1) // multiple) * multiple

        input_ids = torch.full((len(examples), maximum_length), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(examples), maximum_length), dtype=torch.long)
        labels = torch.full((len(examples), maximum_length), -100, dtype=torch.long)
        for row, example in enumerate(examples):
            length = len(example["input_ids"])
            input_ids[row, :length] = torch.tensor(example["input_ids"], dtype=torch.long)
            attention_mask[row, :length] = torch.tensor(example["attention_mask"], dtype=torch.long)
            labels[row, :length] = torch.tensor(example["labels"], dtype=torch.long)

        batch: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "sample_ids": [example["sample_id"] for example in examples],
            "source_metadata": [example.get("source_metadata", {}) for example in examples],
        }
        if any("selected_anchor_positions" in example for example in examples):
            positions = [example.get("selected_anchor_positions", []) for example in examples]
            target_ids = [example.get("selected_anchor_target_ids", []) for example in examples]
            width = max(len(row) for row in positions)
            position_tensor = torch.zeros((len(examples), width), dtype=torch.long)
            position_mask = torch.zeros((len(examples), width), dtype=torch.bool)
            target_tensor = torch.full((len(examples), width), -100, dtype=torch.long)
            for row_index, row_positions in enumerate(positions):
                if len(target_ids[row_index]) != len(row_positions):
                    raise ValueError("Anchor positions and target IDs must have equal lengths")
                if row_positions:
                    position_tensor[row_index, : len(row_positions)] = torch.tensor(row_positions)
                    position_mask[row_index, : len(row_positions)] = True
                    target_tensor[row_index, : len(row_positions)] = torch.tensor(
                        target_ids[row_index]
                    )
            batch["selected_anchor_positions"] = position_tensor
            batch["selected_anchor_position_mask"] = position_mask
            batch["selected_anchor_target_ids"] = target_tensor
        regression_examples = [example for example in examples if "regression_label" in example]
        if regression_examples:
            if len(regression_examples) != len(examples):
                raise ValueError("a batch cannot mix regression and non-regression examples")
            integer_token_ids = regression_examples[0]["regression_integer_token_ids"]
            fraction_token_ids = regression_examples[0]["regression_fraction_token_ids"]
            if any(
                example["regression_integer_token_ids"] != integer_token_ids
                or example["regression_fraction_token_ids"] != fraction_token_ids
                for example in regression_examples[1:]
            ):
                raise ValueError("all regression examples must use the same numeric token IDs")
            batch["regression_labels"] = torch.tensor(
                [example["regression_label"] for example in regression_examples],
                dtype=torch.float32,
            )
            batch["regression_positions"] = torch.tensor(
                [example["regression_positions"] for example in regression_examples],
                dtype=torch.long,
            )
            batch["regression_integer_token_ids"] = torch.tensor(
                integer_token_ids,
                dtype=torch.long,
            ).repeat(len(examples), 1)
            batch["regression_fraction_token_ids"] = torch.tensor(
                fraction_token_ids,
                dtype=torch.long,
            ).repeat(len(examples), 1)
        return batch


class DualStreamCollator:
    def __init__(self, token_collator: CausalLMCollator) -> None:
        self.token_collator = token_collator

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        target_examples = [example["target"] for example in examples]
        anchor_examples = [anchor for example in examples for anchor in example.get("anchor", ())]
        return {
            "target": self.token_collator(target_examples),
            "anchor": self.token_collator(anchor_examples) if anchor_examples else None,
        }


class TargetStreamCollator:
    def __init__(self, token_collator: CausalLMCollator) -> None:
        self.token_collator = token_collator

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        return {"target": self.token_collator(examples), "anchor": None}


def model_tensor_inputs(batch: dict[str, Any]) -> dict[str, Tensor]:
    return {
        key: value
        for key, value in batch.items()
        if key in {"input_ids", "attention_mask", "position_ids"} and isinstance(value, Tensor)
    }
