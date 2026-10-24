"""Independent algebraic oracles for Unembedding Laguerre geometry."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import Tensor


def _fp32(value: Tensor) -> Tensor:
    return value.float() if value.dtype != torch.float32 else value


def linear_logits(hidden_states: Tensor, weight: Tensor, bias: Tensor | None = None) -> Tensor:
    return functional.linear(
        _fp32(hidden_states),
        _fp32(weight),
        _fp32(bias) if bias is not None else None,
    )


def laguerre_power_distances(
    hidden_states: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
) -> Tensor:
    hidden_states_fp32 = _fp32(hidden_states)
    weight_fp32 = _fp32(weight)
    sites = weight_fp32 / 2.0
    site_norms = weight_fp32.square().sum(dim=-1) / 4.0
    weights = site_norms
    if bias is not None:
        weights = weights + _fp32(bias)
    return (hidden_states_fp32.unsqueeze(-2) - sites).square().sum(dim=-1) - weights


def pairwise_signed_distance(
    hidden_states: Tensor,
    weight: Tensor,
    reference_token_ids: Tensor,
    competitor_token_ids: Tensor,
    bias: Tensor | None = None,
    *,
    epsilon: float = 1e-12,
) -> Tensor:
    hidden_states_fp32 = _fp32(hidden_states)
    weight_fp32 = _fp32(weight)
    reference_weight = weight_fp32[reference_token_ids]
    competitor_weight = weight_fp32[competitor_token_ids]
    difference = reference_weight - competitor_weight
    denominator = torch.linalg.vector_norm(difference, dim=-1)
    if bool((denominator <= epsilon).any()):
        raise ValueError("a reference/competitor pair has a zero or near-zero boundary normal")
    numerator = (difference * hidden_states_fp32).sum(dim=-1)
    if bias is not None:
        bias_fp32 = _fp32(bias)
        numerator = numerator + bias_fp32[reference_token_ids] - bias_fp32[competitor_token_ids]
    return numerator / denominator


@dataclass(frozen=True)
class LaguerreOracleAudit:
    argmin_argmax_match: bool
    power_identity_max_error: float
    pairwise_identity_max_error: float
    checked_pairs: int


@dataclass(frozen=True)
class LVDPartitionAudit:
    unembedding_token_ids: tuple[int, ...]
    lvd_cell_token_ids: tuple[int, ...]
    token_ids_match: bool
    all_points_inside_lvd_cells: bool
    mismatched_point_count: int
    violating_halfspace_count: int
    boundary_tie_count: int
    near_boundary_point_count: int
    checked_hidden_state_count: int
    checked_halfspace_count: int
    vocabulary_size: int
    minimum_signed_distance: float
    power_identity_matches: bool
    power_identity_max_error: float


@torch.no_grad()
def audit_lvd_partition(
    hidden_states: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    *,
    vocabulary_chunk_size: int = 1024,
    normal_epsilon: float = 1e-12,
    absolute_tolerance: float = 1e-5,
    relative_tolerance: float = 1e-5,
) -> LVDPartitionAudit:
    if hidden_states.ndim < 1 or weight.ndim != 2:
        raise ValueError(
            "hidden states and Unembedding weight must have shapes [..., H] and [V, H]"
        )
    if hidden_states.shape[-1] != weight.shape[-1]:
        raise ValueError("hidden-state and Unembedding dimensions do not match")
    if weight.shape[0] <= 1:
        raise ValueError("LVD partition audit requires at least two vocabulary tokens")
    if bias is not None and bias.shape != weight.shape[:-1]:
        raise ValueError("Unembedding bias must have one value per vocabulary row")
    if hidden_states.device != weight.device or (bias is not None and bias.device != weight.device):
        raise ValueError("hidden states, Unembedding weight, and bias must be on the same device")
    if vocabulary_chunk_size <= 0:
        raise ValueError("vocabulary_chunk_size must be positive")
    if normal_epsilon <= 0 or absolute_tolerance < 0 or relative_tolerance < 0:
        raise ValueError("LVD audit tolerances are invalid")

    hidden = _fp32(hidden_states).reshape(-1, hidden_states.shape[-1])
    if not hidden.shape[0]:
        raise ValueError("LVD partition audit requires at least one hidden state")
    unembedding_weight = _fp32(weight)
    unembedding_bias = _fp32(bias) if bias is not None else None
    logits = linear_logits(hidden, unembedding_weight, unembedding_bias)
    unembedding_ids = logits.argmax(dim=-1)
    maximum_logits = logits.gather(1, unembedding_ids.unsqueeze(1))
    ties = logits.eq(maximum_logits).sum(dim=-1)

    vocabulary_size = unembedding_weight.shape[0]
    best_power = torch.full(
        (hidden.shape[0],),
        torch.inf,
        dtype=torch.float32,
        device=hidden.device,
    )
    lvd_ids = torch.zeros(hidden.shape[0], dtype=torch.long, device=hidden.device)
    hidden_norm = hidden.square().sum(dim=-1, keepdim=True)
    power_identity_matches = True
    power_identity_max_error = 0.0
    for start in range(0, vocabulary_size, vocabulary_chunk_size):
        end = min(start + vocabulary_chunk_size, vocabulary_size)
        chunk_weight = unembedding_weight[start:end]
        sites = chunk_weight / 2.0
        laguerre_weights = chunk_weight.square().sum(dim=-1) / 4.0
        if unembedding_bias is not None:
            laguerre_weights = laguerre_weights + unembedding_bias[start:end]
        power_distances = (hidden.unsqueeze(1) - sites.unsqueeze(0)).square().sum(
            dim=-1
        ) - laguerre_weights.unsqueeze(0)
        expected_power = hidden_norm - logits[:, start:end]
        power_error = (power_distances - expected_power).abs()
        power_identity_matches = power_identity_matches and not bool(
            (power_error > absolute_tolerance + relative_tolerance * expected_power.abs()).any()
        )
        power_identity_max_error = max(
            power_identity_max_error,
            float(power_error.max().item()),
        )
        chunk_power, chunk_offsets = power_distances.min(dim=-1)
        improves = chunk_power < best_power
        best_power = torch.where(improves, chunk_power, best_power)
        lvd_ids = torch.where(improves, chunk_offsets + start, lvd_ids)

    winner_weight = unembedding_weight[lvd_ids]
    winner_bias = unembedding_bias[lvd_ids] if unembedding_bias is not None else None
    minimum_signed_distance = torch.full(
        (hidden.shape[0],),
        torch.inf,
        dtype=torch.float32,
        device=hidden.device,
    )
    violating_halfspaces = torch.zeros(hidden.shape[0], dtype=torch.long, device=hidden.device)
    for start in range(0, vocabulary_size, vocabulary_chunk_size):
        end = min(start + vocabulary_chunk_size, vocabulary_size)
        chunk_weight = unembedding_weight[start:end]
        normals = winner_weight.unsqueeze(1) - chunk_weight.unsqueeze(0)
        normal_norms = torch.linalg.vector_norm(normals, dim=-1)
        token_ids = torch.arange(start, end, device=hidden.device)
        self_mask = token_ids.unsqueeze(0).eq(lvd_ids.unsqueeze(1))
        if bool((normal_norms.le(normal_epsilon) & ~self_mask).any()):
            raise ValueError("LVD partition contains a zero or near-zero boundary normal")
        numerators = (normals * hidden.unsqueeze(1)).sum(dim=-1)
        if winner_bias is not None and unembedding_bias is not None:
            numerators = numerators + winner_bias.unsqueeze(1) - unembedding_bias[start:end]
        signed_distances = numerators / normal_norms.clamp_min(normal_epsilon)
        signed_distances = signed_distances.masked_fill(self_mask, torch.inf)
        minimum_signed_distance = torch.minimum(
            minimum_signed_distance,
            signed_distances.min(dim=-1).values,
        )
        violating_halfspaces += signed_distances.lt(-absolute_tolerance).sum(dim=-1)

    mismatches = unembedding_ids.ne(lvd_ids)
    return LVDPartitionAudit(
        unembedding_token_ids=tuple(int(value) for value in unembedding_ids.tolist()),
        lvd_cell_token_ids=tuple(int(value) for value in lvd_ids.tolist()),
        token_ids_match=not bool(mismatches.any()),
        all_points_inside_lvd_cells=not bool(violating_halfspaces.any()),
        mismatched_point_count=int(mismatches.sum().item()),
        violating_halfspace_count=int(violating_halfspaces.sum().item()),
        boundary_tie_count=int(ties.gt(1).sum().item()),
        near_boundary_point_count=int(minimum_signed_distance.le(absolute_tolerance).sum().item()),
        checked_hidden_state_count=hidden.shape[0],
        checked_halfspace_count=hidden.shape[0] * (vocabulary_size - 1),
        vocabulary_size=vocabulary_size,
        minimum_signed_distance=float(minimum_signed_distance.min().item()),
        power_identity_matches=power_identity_matches,
        power_identity_max_error=power_identity_max_error,
    )


@torch.no_grad()
def audit_laguerre_oracle(
    hidden_states: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    *,
    absolute_tolerance: float = 1e-5,
    relative_tolerance: float = 1e-5,
) -> LaguerreOracleAudit:
    if hidden_states.shape[-1] != weight.shape[-1]:
        raise ValueError("hidden-state and Unembedding dimensions do not match")
    if bias is not None and bias.shape != weight.shape[:-1]:
        raise ValueError("Unembedding bias must have one value per vocabulary row")

    logits = linear_logits(hidden_states, weight, bias)
    power_distances = laguerre_power_distances(hidden_states, weight, bias)
    hidden_norm = _fp32(hidden_states).square().sum(dim=-1, keepdim=True)
    expected_power = hidden_norm - logits
    if not torch.allclose(
        power_distances,
        expected_power,
        atol=absolute_tolerance,
        rtol=relative_tolerance,
    ):
        raise AssertionError("Laguerre power distances do not equal ||h||^2 - linear logits")

    flattened_hidden = _fp32(hidden_states).reshape(-1, hidden_states.shape[-1])
    flattened_logits = logits.reshape(-1, logits.shape[-1])
    reference_ids = flattened_logits.argmax(dim=-1)
    vocabulary_size = weight.shape[0]
    competitor_offsets = torch.arange(1, vocabulary_size, device=weight.device)
    competitor_ids = (reference_ids.unsqueeze(-1) + competitor_offsets) % vocabulary_size
    expanded_reference_ids = reference_ids.unsqueeze(-1).expand_as(competitor_ids)
    expanded_hidden = flattened_hidden.unsqueeze(-2).expand(-1, vocabulary_size - 1, -1)
    distances = pairwise_signed_distance(
        expanded_hidden,
        weight,
        expanded_reference_ids,
        competitor_ids,
        bias,
    )
    reference_logits = flattened_logits.gather(1, reference_ids.unsqueeze(-1))
    competitor_logits = flattened_logits.gather(1, competitor_ids)
    normals = _fp32(weight)[expanded_reference_ids] - _fp32(weight)[competitor_ids]
    reconstructed_differences = torch.linalg.vector_norm(normals, dim=-1) * distances
    expected_differences = reference_logits - competitor_logits
    if not torch.allclose(
        reconstructed_differences,
        expected_differences,
        atol=absolute_tolerance,
        rtol=relative_tolerance,
    ):
        raise AssertionError("pairwise signed distances do not reconstruct logit differences")

    return LaguerreOracleAudit(
        argmin_argmax_match=bool(
            torch.equal(power_distances.argmin(dim=-1), logits.argmax(dim=-1))
        ),
        power_identity_max_error=float((power_distances - expected_power).abs().max().item()),
        pairwise_identity_max_error=float(
            (reconstructed_differences - expected_differences).abs().max().item()
        ),
        checked_pairs=expanded_reference_ids.numel(),
    )
