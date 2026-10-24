"""Loss plugin interfaces and shared forward caching."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Protocol

import torch
import torch.distributed as distributed
import torch.nn.functional as functional
from torch import Tensor, nn

from sft_conflict.data.collator import model_tensor_inputs
from sft_conflict.models import UnembeddingInputCapture


@torch.no_grad()
def causal_lm_accuracy_counts(logits: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
    """Teacher-forced next-token counts, without a softmax or a retained graph."""
    shifted_labels = labels[:, 1:]
    valid = shifted_labels.ne(-100)
    predictions = logits[:, :-1, :].argmax(dim=-1)
    return predictions.eq(shifted_labels).logical_and(valid).sum(), valid.sum()


def causal_lm_cross_entropy(logits: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
    shift_logits = logits[:, :-1, :].contiguous().float()
    shift_labels = labels[:, 1:].contiguous()
    valid_mask = shift_labels.ne(-100)
    valid_count = valid_mask.sum()
    if not bool(valid_count):
        return logits.sum() * 0.0, valid_count
    loss = (
        functional.cross_entropy(
            shift_logits.view(-1, shift_logits.shape[-1]),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="sum",
        )
        / valid_count
    )
    return loss, valid_count


def stsb_numeric_mse(
    logits: Tensor,
    labels: Tensor,
    positions: Tensor,
    integer_token_ids: Tensor,
    fraction_token_ids: Tensor,
) -> tuple[Tensor, Tensor]:
    batch_size = logits.shape[0]
    if labels.shape != (batch_size,):
        raise ValueError("STS-B regression labels must have shape [batch]")
    if positions.shape != (batch_size, 2):
        raise ValueError("STS-B regression positions must have shape [batch, 2]")
    if integer_token_ids.shape != (batch_size, 6):
        raise ValueError("STS-B integer token IDs must have shape [batch, 6]")
    if fraction_token_ids.shape != (batch_size, 10):
        raise ValueError("STS-B fractional token IDs must have shape [batch, 10]")
    prediction_positions = positions - 1
    if bool((prediction_positions < 0).any()) or bool(
        (prediction_positions >= logits.shape[1]).any()
    ):
        raise ValueError("STS-B regression prediction positions are outside the logits")

    row_indices = torch.arange(batch_size, device=logits.device)
    integer_logits = (
        logits[row_indices, prediction_positions[:, 0]]
        .float()
        .gather(
            1,
            integer_token_ids,
        )
    )
    fraction_logits = (
        logits[row_indices, prediction_positions[:, 1]]
        .float()
        .gather(
            1,
            fraction_token_ids,
        )
    )
    integer_probabilities = functional.softmax(integer_logits, dim=-1)
    fraction_probabilities = functional.softmax(fraction_logits, dim=-1)
    integer_values = torch.arange(6, device=logits.device, dtype=torch.float32)
    fraction_values = torch.arange(10, device=logits.device, dtype=torch.float32) / 10.0
    expected_integer = (integer_probabilities * integer_values).sum(dim=-1)
    expected_fraction = (fraction_probabilities * fraction_values).sum(dim=-1)
    predictions = expected_integer + (1.0 - integer_probabilities[:, 5]) * expected_fraction
    return functional.mse_loss(predictions, labels.float()), predictions


@dataclass(frozen=True)
class LossResult:
    loss: Tensor
    metrics: dict[str, Tensor | float]
    # These override scalar means for the same keys in metrics at logging time.
    ratio_numerators: dict[str, Tensor | float] = field(default_factory=dict)
    ratio_denominators: dict[str, Tensor | float] = field(default_factory=dict)
    sum_metrics: dict[str, Tensor | float] = field(default_factory=dict)


class CountAwareMetricAccumulator:
    """Accumulate a window locally; align keys and SUM only when logging.

    Ratios divide globally summed numerators by globally summed denominators.
    An empty denominator reports zero (with the caller's explicit count metrics).
    Other metrics retain the legacy mean of rank/microbatch observations, NOT a
    token mean or a pooled quantile. Counts use float64 for the compact reduction.
    """

    def __init__(self) -> None:
        self.totals: defaultdict[str, float] = defaultdict(float)
        self.observations: defaultdict[str, int] = defaultdict(int)
        self.ratio_numerators: defaultdict[str, float] = defaultdict(float)
        self.ratio_denominators: defaultdict[str, float] = defaultdict(float)
        self.sum_totals: defaultdict[str, float] = defaultdict(float)

    @staticmethod
    def _number(value: Tensor | float) -> float:
        return float(value.detach().item()) if isinstance(value, Tensor) else float(value)

    def update(self, result: LossResult) -> None:
        if result.ratio_numerators.keys() != result.ratio_denominators.keys():
            raise ValueError("metric numerator and denominator keys must match")
        counted = result.ratio_numerators.keys() | result.sum_metrics.keys()
        for name, value in result.metrics.items():
            if name not in counted:
                self.totals[name] += self._number(value)
                self.observations[name] += 1
        for source, destination in (
            (result.ratio_numerators, self.ratio_numerators),
            (result.ratio_denominators, self.ratio_denominators),
            (result.sum_metrics, self.sum_totals),
        ):
            for name, value in source.items():
                destination[name] += self._number(value)

    def compute(
        self,
        *,
        device: torch.device | str | None = None,
        process_group: distributed.ProcessGroup | None = None,
    ) -> dict[str, float]:
        """All ranks must call, including ranks with no observations in the window."""
        columns = {
            "mean": (self.totals, self.observations),
            "ratio": (self.ratio_numerators, self.ratio_denominators),
            "sum": (self.sum_totals, {}),
        }
        local_keys = [(name, kind) for kind, (totals, _) in columns.items() for name in totals]
        distributed_active = distributed.is_available() and distributed.is_initialized()
        world_size = distributed.get_world_size(process_group) if distributed_active else 1
        schemas = [local_keys]
        if world_size > 1:
            # Never reduce rank-local dict iteration order or skip empty ranks.
            # The small schema exchange also supports plugin-specific/missing keys.
            schemas = [[] for _ in range(world_size)]
            distributed.all_gather_object(schemas, local_keys, group=process_group)
        keys = sorted({key for schema in schemas for key in schema})
        kinds: dict[str, str] = {}
        for name, kind in keys:
            if name in kinds and kinds[name] != kind:
                raise ValueError(f"inconsistent metric aggregation for {name}")
            kinds[name] = kind
        if not keys:
            return {}
        if device is None:
            device = (
                torch.device("cuda", torch.cuda.current_device())
                if distributed_active and distributed.get_backend(process_group) == "nccl"
                else torch.device("cpu")
            )
        values = torch.tensor(
            [
                [columns[kind][0].get(name, 0.0), columns[kind][1].get(name, 0.0)]
                for name, kind in keys
            ],
            dtype=torch.float64,
            device=device,
        )
        if world_size > 1:
            distributed.all_reduce(values, op=distributed.ReduceOp.SUM, group=process_group)
        return {
            name: (
                numerator if kind == "sum" else numerator / denominator if denominator > 0 else 0.0
            )
            for (name, kind), (numerator, denominator) in zip(
                keys, values.cpu().tolist(), strict=True
            )
        }

    def reset(self) -> None:
        for values in (
            self.totals,
            self.observations,
            self.ratio_numerators,
            self.ratio_denominators,
            self.sum_totals,
        ):
            values.clear()


class LossPlugin(Protocol):
    name: str
    requires_anchor_hidden: bool

    def __call__(self, context: LossContext) -> LossResult: ...


class LossContext:
    def __init__(
        self,
        model: nn.Module,
        batch: dict[str, Any],
        *,
        reference_model: nn.Module | None,
        capture_anchor_hidden: bool,
        global_step: int = 0,
        compensate_distributed_gradient_average: bool = True,
        combine_current_streams: bool = False,
        capture_anchor_label_positions: bool = False,
        anchor_geometry_plugin: Any | None = None,
        anchor_geometry_epoch: int = 0,
    ) -> None:
        self.model = model
        self.batch = batch
        self.reference_model = reference_model
        self.capture_anchor_hidden = capture_anchor_hidden
        self.capture_anchor_label_positions = bool(capture_anchor_label_positions)
        self.global_step = int(global_step)
        self.compensate_distributed_gradient_average = bool(compensate_distributed_gradient_average)
        self.combine_current_streams = bool(combine_current_streams)
        self.anchor_geometry_plugin = anchor_geometry_plugin
        self.anchor_geometry_epoch = int(anchor_geometry_epoch)
        self.anchor_geometry_result: LossResult | None = None
        self.current_outputs: dict[str, Any] = {}
        self.reference_outputs: dict[str, Any] = {}
        self.hidden_states: dict[str, Tensor] = {}
        self._anchor_label_selection_cache: tuple[Tensor, Tensor, Tensor] | None = None
        self._anchor_label_positions_captured = False
        # Joint Anchor CE+LVD shares logits and must report their counts only once.
        self.anchor_accuracy_recorded = False

    def stream_batch(self, stream: str) -> dict[str, Any] | None:
        value = self.batch.get(stream)
        if value is not None and not isinstance(value, dict):
            raise TypeError(f"{stream} batch must be a mapping or None")
        return value

    @torch.no_grad()
    def anchor_label_selection(self) -> tuple[Tensor, Tensor, Tensor]:
        """Cache all supervised causal (predictor position, mask, gold ID) rows.

        Each output is [B,M], with M the largest shifted-label count in this
        batch (possibly zero). Padding is position 0, mask False, gold -100.
        EOS and initially incorrect labels are ordinary observations. Legacy
        selected-position fields never participate in this selection.
        """
        if self._anchor_label_selection_cache is not None:
            return self._anchor_label_selection_cache
        batch = self.stream_batch("anchor")
        if batch is None:
            raise ValueError("Anchor label selection requires an anchor batch")
        labels = batch.get("labels")
        if (
            not isinstance(labels, Tensor)
            or labels.ndim != 2
            or labels.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("Anchor labels must have integer shape [batch, sequence]")
        input_ids = batch.get("input_ids")
        if input_ids is not None and (
            not isinstance(input_ids, Tensor) or input_ids.shape != labels.shape
        ):
            raise ValueError("Anchor input_ids and labels must have matching shapes")
        attention_mask = batch.get("attention_mask")
        if not isinstance(attention_mask, Tensor) or attention_mask.shape != labels.shape:
            raise ValueError("Anchor attention_mask must match the labels shape")
        if bool(((attention_mask != 0) & (attention_mask != 1)).any()):
            raise ValueError("Anchor attention_mask must contain only zero or one")
        attended = attention_mask.to(device=labels.device, dtype=torch.bool)
        shifted_labels = labels[:, 1:]
        valid = shifted_labels.ne(-100)
        if bool((valid & shifted_labels.lt(0)).any()):
            raise ValueError("supervised Anchor labels must be non-negative token IDs or -100")
        if bool((valid & ~(attended[:, 1:] & attended[:, :-1])).any()):
            raise ValueError(
                "supervised Anchor labels and their preceding prediction positions must not "
                "be on attention-mask padding"
            )
        counts = valid.sum(dim=1)
        width = int(counts.max().item()) if counts.numel() else 0
        positions = torch.zeros((labels.shape[0], width), dtype=torch.long, device=labels.device)
        mask = torch.arange(width, device=labels.device).unsqueeze(0) < counts.unsqueeze(1)
        gold_ids = torch.full_like(positions, -100)
        predictors = torch.arange(shifted_labels.shape[1], device=labels.device).expand_as(valid)
        positions[mask] = predictors[valid]
        gold_ids[mask] = shifted_labels[valid].long()
        self._anchor_label_selection_cache = (positions, mask, gold_ids)
        return self._anchor_label_selection_cache

    def forward_current(self, stream: str) -> Any:
        if stream in self.current_outputs:
            if (
                stream == "anchor"
                and self.capture_anchor_hidden
                and self.capture_anchor_label_positions
                and not self._anchor_label_positions_captured
            ):
                raise RuntimeError(
                    "capture_anchor_label_positions=True and capture_anchor_hidden=True must "
                    "be set before the first Anchor or combined forward"
                )
            return self.current_outputs[stream]
        if self.combine_current_streams:
            self._forward_combined_current()
            return self.current_outputs[stream]
        stream_batch = self.stream_batch(stream)
        if stream_batch is None:
            raise ValueError(f"the current dual-stream step has no {stream} batch")
        capture_hidden = stream == "anchor" and self.capture_anchor_hidden
        if capture_hidden:
            positions = (
                self.anchor_label_selection()[0]
                if self.capture_anchor_label_positions
                else stream_batch.get("selected_anchor_positions")
            )
            with UnembeddingInputCapture(
                self.model,
                positions=positions,
                observer=self._geometry_observer(),
            ) as capture:
                outputs = self.model(**model_tensor_inputs(stream_batch))
            if capture.hidden_states is None:
                raise RuntimeError("the anchor forward did not invoke the Unembedding")
            self.hidden_states[stream] = capture.hidden_states
            self._anchor_label_positions_captured = self.capture_anchor_label_positions
        else:
            outputs = self.model(**model_tensor_inputs(stream_batch))
        self.current_outputs[stream] = outputs
        return outputs

    def _forward_combined_current(self) -> None:
        target_batch = self.stream_batch("target")
        anchor_batch = self.stream_batch("anchor")
        if target_batch is None or anchor_batch is None:
            raise ValueError("combined current forward requires target and anchor batches")
        target_inputs = model_tensor_inputs(target_batch)
        anchor_inputs = model_tensor_inputs(anchor_batch)
        target_length = target_inputs["input_ids"].shape[1]
        anchor_length = anchor_inputs["input_ids"].shape[1]
        combined_length = max(target_length, anchor_length)

        def pad(inputs: dict[str, Tensor]) -> dict[str, Tensor]:
            width = combined_length - inputs["input_ids"].shape[1]
            if width <= 0:
                return inputs
            padded = dict(inputs)
            padded["input_ids"] = functional.pad(inputs["input_ids"], (0, width), value=0)
            padded["attention_mask"] = functional.pad(inputs["attention_mask"], (0, width), value=0)
            if "position_ids" in inputs:
                padded["position_ids"] = functional.pad(inputs["position_ids"], (0, width))
            return padded

        target_inputs = pad(target_inputs)
        anchor_inputs = pad(anchor_inputs)
        combined_inputs = {
            key: torch.cat((target_inputs[key], anchor_inputs[key]), dim=0)
            for key in target_inputs.keys() & anchor_inputs.keys()
        }
        target_size = target_inputs["input_ids"].shape[0]
        if self.capture_anchor_hidden:
            anchor_positions = (
                self.anchor_label_selection()[0]
                if self.capture_anchor_label_positions
                else anchor_batch.get("selected_anchor_positions")
            )
            if anchor_positions is None:
                raise ValueError("combined Anchor hidden capture requires selected positions")
            target_positions = torch.zeros(
                (target_size, anchor_positions.shape[1]),
                dtype=anchor_positions.dtype,
                device=anchor_positions.device,
            )
            positions = torch.cat((target_positions, anchor_positions), dim=0)
            with UnembeddingInputCapture(
                self.model,
                positions=positions,
                observer=self._geometry_observer(target_size, anchor_length),
            ) as capture:
                outputs = self.model(**combined_inputs)
            if capture.hidden_states is None:
                raise RuntimeError("the combined forward did not invoke the Unembedding")
            self.hidden_states["anchor"] = capture.hidden_states[target_size:]
            self._anchor_label_positions_captured = self.capture_anchor_label_positions
        else:
            outputs = self.model(**combined_inputs)
        self.current_outputs["target"] = SimpleNamespace(
            logits=outputs.logits[:target_size, :target_length]
        )
        self.current_outputs["anchor"] = SimpleNamespace(
            logits=outputs.logits[target_size:, :anchor_length]
        )

    def _geometry_observer(self, target_size: int = 0, anchor_length: int | None = None):
        if self.anchor_geometry_plugin is None:
            return None

        def observe(weight, bias, selected_hidden, logits):
            self.anchor_geometry_result = self.anchor_geometry_plugin.observe(
                self,
                weight,
                bias,
                selected_hidden[target_size:],
                logits[target_size:, :anchor_length],
            )

        return observe

    def forward_reference(self, stream: str) -> Any:
        if stream in self.reference_outputs:
            return self.reference_outputs[stream]
        if self.reference_model is None:
            raise ValueError("this loss requires an online reference model")
        stream_batch = self.stream_batch(stream)
        if stream_batch is None:
            raise ValueError(f"the current dual-stream step has no {stream} batch")
        reference_inputs = {
            key: value
            for key, value in model_tensor_inputs(stream_batch).items()
            if key != "labels"
        }
        self.reference_model.eval()
        with torch.no_grad():
            outputs = self.reference_model(**reference_inputs)
        self.reference_outputs[stream] = outputs
        return outputs

    def connected_zero(self) -> Tensor:
        return self.forward_current("target").logits.sum() * 0.0
