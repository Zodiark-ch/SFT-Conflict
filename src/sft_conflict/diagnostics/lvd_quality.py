"""Paired geometric diagnostics; all rates use explicit pooled denominators."""

from __future__ import annotations

import torch
from torch import Tensor

from sft_conflict.config import LVDConfig, LVDPressureMode, LVDResponseMode
from sft_conflict.geometry import chunked_lvd_pressure_loss, second_moment_frobenius_loss


@torch.no_grad()
def tensor_summary(values: Tensor) -> dict[str, float]:
    data = values.detach().double().reshape(-1)
    if not data.numel():
        return {key: 0.0 for key in ("mean", "min", "q05", "q50", "q95", "max")}
    quantiles = torch.quantile(data, data.new_tensor([0.05, 0.5, 0.95])).cpu().tolist()
    return dict(
        zip(
            ("mean", "min", "q05", "q50", "q95", "max"),
            [float(data.mean()), float(data.min()), *quantiles, float(data.max())],
            strict=True,
        )
    )


@torch.no_grad()
def moment_spectrum(rows: Tensor, *, centered: bool) -> dict[str, object]:
    values = rows.detach().double()
    if centered:
        values = values - values.mean(0, keepdim=True)
    n, dimension = values.shape
    matrix = (values @ values.T if n <= dimension else values.T @ values) / n
    eigenvalues = torch.linalg.eigvalsh(matrix).clamp_min(0).flip(0)
    trace = float(eigenvalues.sum())
    probabilities = eigenvalues / eigenvalues.sum().clamp_min(1e-300)
    entropy = -(probabilities * probabilities.clamp_min(1e-300).log()).sum()
    return {
        "centered": centered,
        "sample_count": n,
        "dimension": dimension,
        "trace": trace,
        "frobenius_squared": float(eigenvalues.square().sum()),
        "maximum_eigenvalue": float(eigenvalues[0]),
        "maximum_eigenvalue_share": float(probabilities[0]) if trace else 0.0,
        "effective_rank": float(entropy.exp()) if trace else 0.0,
        "top16_eigenvalues": eigenvalues[:16].cpu().tolist(),
        "coordinate_second_moments": (values.square().mean(0)).cpu().tolist(),
    }


def replay_pressure_diagnostics(
    *,
    hidden: Tensor,
    base: Tensor,
    weight: Tensor,
    references: Tensor,
    competitors: Tensor,
    margins: Tensor,
    config: LVDConfig,
    sample_sizes: list[int],
) -> tuple[dict[str, object], Tensor, Tensor]:
    """Replay effective pressure/gradients, retaining empty logical microbatches.

    Robust-mode raw spectra are offline, whole-pool comparators, not the trained
    effective-pressure objective or a mean of global microbatch objectives.

    Only three-segment reports add an explicit region-index schema and a
    zero_response_gradient_crossing_rate: this cap-saturation diagnostic is
    zero for its linear tail, not a count of core zeros or full-loss gradients.
    """
    if hidden.ndim != 2 or len(hidden) == 0:
        raise ValueError("replay pressure diagnostics require non-empty [N, hidden] states")
    if any(
        not isinstance(size, int) or isinstance(size, bool) or size < 0 for size in sample_sizes
    ):
        raise ValueError("sample_sizes must contain non-negative integers")
    if sum(sample_sizes) != len(hidden):
        raise ValueError("sample_sizes must sum to len(hidden)")
    current = hidden.detach().clone().requires_grad_()
    mask = torch.ones_like(competitors, dtype=torch.bool)
    result = chunked_lvd_pressure_loss(
        current_hidden=current,
        reference_hidden=base,
        unembedding_weight=weight,
        reference_token_ids=references,
        competitor_ids=competitors,
        baseline_margins=margins,
        competitor_mask=mask,
        config=config,
    )
    # Match four one-sample ranks per logical microbatch; never confuse this
    # average of global microbatch objectives with a whole-dataset covariance.
    block_losses = []
    offset = 0
    for start in range(0, len(sample_sizes), 4):
        count = sum(sample_sizes[start : start + 4])
        if count:
            block_losses.append(
                second_moment_frobenius_loss(
                    result.pressure_vectors[offset : offset + count], mode=config.moment_mode
                )
            )
        else:
            # Production includes all-empty global microbatches as connected zeros.
            block_losses.append(current[offset:offset].sum() * 0.0)
        offset += count
    objective = torch.stack(block_losses).mean()
    gradient = torch.autograd.grad(objective, current)[0]
    ratios = result.erosion_ratios.detach()
    regions = result.response.regions.detach()
    mode = LVDResponseMode(config.response_mode)
    boundary = 1.0 / config.gamma0
    active_exponent = torch.zeros_like(ratios)
    if mode is LVDResponseMode.THREE_SEGMENT:
        active_exponent = torch.where(ratios >= boundary, config.mu3 * (ratios - boundary), 0.0)
    elif mode in {LVDResponseMode.FOUR_SEGMENT, LVDResponseMode.FOUR_SEGMENT_LINEAR_TAIL}:
        active_exponent = torch.where(
            ratios >= boundary,
            config.mu4 * (ratios - boundary),
            torch.where(ratios >= 1, config.mu3 * (ratios - 1), 0.0),
        )
    active_capped = active_exponent.gt(config.exponent_clamp)
    response = result.response.values.detach()
    alpha = result.capacity_weights.detach()
    amplitudes = alpha * response
    pressure = result.pressure_vectors.detach()
    scalar_sum = amplitudes.sum(-1)
    norm = pressure.norm(dim=-1)
    loss_weight = 1.0 if config.resolved_lambda is None else config.resolved_lambda
    with torch.no_grad():
        report = {
            "positions": len(hidden),
            "boundaries": competitors.numel(),
            "region_counts": [int(regions.eq(i).sum()) for i in range(mode.region_count)],
            "region_rates": [
                float(regions.eq(i).double().mean()) for i in range(mode.region_count)
            ],
            "crossing_rate": float(result.current_margins.le(0).double().mean()),
            "active_exponent_above_cap_rate": float(active_capped.double().mean()),
            "active_exponent": tensor_summary(active_exponent),
            "d0": tensor_summary(margins),
            "dnew": tensor_summary(result.current_margins),
            "erosion_ratio": tensor_summary(ratios),
            "response": tensor_summary(response),
            "alpha": tensor_summary(alpha),
            "alpha_max_per_position": tensor_summary(alpha.max(-1).values),
            "alpha_effective_neighbor_count": tensor_summary(alpha.square().sum(-1).reciprocal()),
            "pressure_norm": tensor_summary(norm),
            "pressure_energy": tensor_summary(norm.square()),
            "vector_sum_over_sum_of_magnitudes": tensor_summary(norm / scalar_sum.clamp_min(1e-30)),
            "microbatch_moment_losses": tensor_summary(torch.stack(block_losses)),
            "mean_microbatch_loss": float(objective.detach()),
            "microbatch_loss_values": [float(value.detach()) for value in block_losses],
            "hidden_gradient_norm": float(gradient.double().norm()),
            "position_gradient_norm": tensor_summary(gradient.norm(dim=-1)),
            "weighted_hidden_gradient_norm": float(gradient.double().norm()) * loss_weight,
            "weighted_microbatch_mean_loss": float(objective.detach()) * loss_weight,
            "lambda": config.resolved_lambda,
            "use_capacity_weights": config.use_capacity_weights,
            "response_mode": mode.value,
            "pressure_mode": config.pressure_mode.value,
            "uncentered": moment_spectrum(pressure, centered=False),
            "centered": moment_spectrum(pressure, centered=True),
        }
        if mode is LVDResponseMode.THREE_SEGMENT:
            report["response_region_schema"] = {
                str(i): name for i, name in enumerate(mode.region_names)
            }
            # Above-cap exponents enter a nonzero-slope linear continuation.
            # Physical crossings inside a floored zero core are not saturation.
            report["zero_response_gradient_crossing_rate"] = 0.0
        if result.raw_pressure_vectors is not None:
            raw_pressure = result.raw_pressure_vectors.detach()
            raw_norm = raw_pressure.norm(dim=-1)
            report.update(
                {
                    "raw_pressure_mode": LVDPressureMode.WEIGHTED_SUM.value,
                    "pressure_semantics": (
                        "pressure, moments and gradients use effective asinh-weighted-sum P; "
                        "raw_* use original weighted-sum P. Raw spectra pool all replay positions, "
                        "not microbatch losses. The vector-sum ratio uses raw P to measure "
                        "directional cancellation, not radial compression."
                    ),
                    "raw_pressure_norm": tensor_summary(raw_norm),
                    "raw_pressure_energy": tensor_summary(raw_norm.square()),
                    "raw_uncentered": moment_spectrum(raw_pressure, centered=False),
                    "raw_centered": moment_spectrum(raw_pressure, centered=True),
                    "vector_sum_over_sum_of_magnitudes": tensor_summary(
                        raw_norm / scalar_sum.clamp_min(1e-30)
                    ),
                }
            )
    return report, pressure.cpu(), gradient.detach().cpu()


@torch.no_grad()
def full_vocabulary_geometry(
    hidden: Tensor,
    base: Tensor,
    weight: Tensor,
    references: Tensor,
    *,
    gamma: float,
    floor: float,
    response_mode: LVDResponseMode | str = LVDResponseMode.FOUR_SEGMENT,
) -> dict:
    """Count ALL non-reference boundaries using [chunk,V], not [N,V,H].

    Default to the historical four-bin taxonomy, independent of current YAML
    defaults; gamma remains the caller's source-geometry parameter. Three-segment
    regions are 0 (rho<=1), 1 (1<rho<1/gamma), and 2 (rho>=1/gamma), with an added
    response_region_schema mapping indices to names. Floored d0 can physically
    cross before the last ratio region; top-1 agreement is measured separately.
    """
    mode = LVDResponseMode(response_mode)
    counts = torch.zeros(mode.region_count, dtype=torch.int64, device=hidden.device)
    actual_flips = 0
    max_nearest_margin_error = 0.0
    double_weight = weight.double()
    norms = double_weight.square().sum(-1)
    for start in range(0, len(hidden), 16):
        h, h0, ids = (
            hidden[start : start + 16],
            base[start : start + 16],
            references[start : start + 16],
        )
        wr = double_weight[ids]
        distance_norm = (
            (norms[ids, None] + norms[None] - 2 * (wr @ double_weight.T))
            .clamp_min(0)
            .sqrt()
            .float()
        )
        valid = torch.ones_like(distance_norm, dtype=torch.bool)
        valid.scatter_(1, ids[:, None], False)
        h0_logits, h_logits = h0 @ weight.T, h @ weight.T
        d0 = (h0_logits.gather(1, ids[:, None]) - h0_logits) / distance_norm.clamp_min(1e-12)
        dn = (h_logits.gather(1, ids[:, None]) - h_logits) / distance_norm.clamp_min(1e-12)
        rho = (d0 - dn) / (gamma * d0.clamp_min(floor))
        if mode is LVDResponseMode.THREE_SEGMENT:
            region = torch.where(rho <= 1, 0, torch.where(rho < 1 / gamma, 1, 2))
        else:
            region = torch.where(
                rho <= 0, 0, torch.where(rho < 1, 1, torch.where(rho < 1 / gamma, 2, 3))
            )
        for i in range(mode.region_count):
            counts[i] += ((region == i) & valid).sum()
        actual_flips += int(h_logits.argmax(-1).ne(ids).sum())
        max_nearest_margin_error = max(
            max_nearest_margin_error, float((-d0[valid]).clamp_min(0).max())
        )
    report = {
        "boundary_count": int(counts.sum()),
        "region_counts": counts.cpu().tolist(),
        "region_rates": (counts.double() / counts.sum()).cpu().tolist(),
        "base_top1_disagreement_count": actual_flips,
        "base_top1_agreement": 1 - actual_flips / len(hidden),
        "recomputed_d0_max_negative": max_nearest_margin_error,
    }
    if mode is LVDResponseMode.THREE_SEGMENT:
        report["response_region_schema"] = {
            str(i): name for i, name in enumerate(mode.region_names)
        }
    return report
