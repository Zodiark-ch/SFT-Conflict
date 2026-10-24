"""Persistent artifacts for post-SFT diagnostics."""

from __future__ import annotations

import csv
import html
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from .sft_effects import TokenDirectionMovementStatistics


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_interference_artifacts(
    statistics: TokenDirectionMovementStatistics,
    *,
    token_labels: Sequence[str],
    output_dir: str | Path,
    metadata: dict[str, Any],
) -> dict[str, str]:
    vocabulary_size = len(statistics.mean_absolute_movement)
    if len(token_labels) != vocabulary_size:
        raise ValueError("token labels must contain one value per vocabulary direction")
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    ranking_path = destination / "token-direction-movement.csv"
    summary_path = destination / "interference-summary.json"
    plot_path = destination / "token-direction-movement.svg"

    order = torch.argsort(statistics.mean_positive_movement, descending=True)
    ranked_positive = statistics.mean_positive_movement[order]
    positive_total = float(ranked_positive.sum().item())

    def top_share(fraction: float) -> float:
        count = max(1, round(vocabulary_size * fraction))
        return float(ranked_positive[:count].sum().item()) / max(positive_total, 1e-12)

    temporary_ranking = ranking_path.with_suffix(".csv.tmp")
    with temporary_ranking.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "rank",
                "token_id",
                "token",
                "mean_positive_movement",
                "mean_absolute_movement",
                "mean_signed_movement",
                "root_mean_square_movement",
                "maximum_absolute_movement",
            )
        )
        for rank, token_id_tensor in enumerate(order, start=1):
            token_id = int(token_id_tensor.item())
            writer.writerow(
                (
                    rank,
                    token_id,
                    token_labels[token_id],
                    float(statistics.mean_positive_movement[token_id].item()),
                    float(statistics.mean_absolute_movement[token_id].item()),
                    float(statistics.mean_signed_movement[token_id].item()),
                    float(statistics.root_mean_square_movement[token_id].item()),
                    float(statistics.maximum_absolute_movement[token_id].item()),
                )
            )
    temporary_ranking.replace(ranking_path)

    summary = {
        **metadata,
        "statistic": "mean_positive_projection_onto_unit_base_unembedding_row",
        "position_count": statistics.position_count,
        "vocabulary_size": vocabulary_size,
        "total_positive_movement": statistics.total_positive_movement,
        "total_absolute_movement": statistics.total_absolute_movement,
        "mean_positive_movement_across_tokens": float(
            statistics.mean_positive_movement.mean().item()
        ),
        "mean_absolute_movement_across_tokens": float(
            statistics.mean_absolute_movement.mean().item()
        ),
        "maximum_mean_positive_movement": float(statistics.mean_positive_movement.max().item()),
        "top_1_percent_positive_movement_share": top_share(0.01),
        "top_5_percent_positive_movement_share": top_share(0.05),
        "top_10_percent_positive_movement_share": top_share(0.10),
        "mean_hidden_l2_displacement": statistics.mean_hidden_l2_displacement,
        "ranking_csv": str(ranking_path),
        "plot_svg": str(plot_path),
    }
    _atomic_json(summary_path, summary)

    ranked_values = ranked_positive
    width, height = 1600, 800
    left, right, top, bottom = 150, 50, 90, 110
    plot_width = width - left - right
    plot_height = height - top - bottom
    maximum = max(float(ranked_values.max().item()), 1e-12)
    point_count = min(vocabulary_size, plot_width + 1)
    sampled_indices = torch.linspace(0, vocabulary_size - 1, point_count).round().long()
    sampled_values = ranked_values[sampled_indices]
    points = []
    for index, value in enumerate(sampled_values):
        x = left + index * plot_width / max(point_count - 1, 1)
        y = top + plot_height * (1.0 - float(value.item()) / maximum)
        points.append(f"{x:.2f},{y:.2f}")
    baseline = top + plot_height
    area_points = f"{left},{baseline} " + " ".join(points) + f" {left + plot_width},{baseline}"
    grid_lines = []
    y_labels = []
    for tick in range(6):
        fraction = tick / 5
        y = top + plot_height * (1 - fraction)
        grid_lines.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" '
            'stroke="#D7DEE2" stroke-width="1"/>'
        )
        y_labels.append(
            f'<text x="{left - 18}" y="{y + 6:.2f}" text-anchor="end">'
            f"{maximum * fraction:.4g}</text>"
        )
    trained_method = html.escape(str(metadata.get("trained_method", "SFT")))
    center = width / 2
    vertical_center = top + plot_height / 2
    svg_lines = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">'
        ),
        '<rect width="100%" height="100%" fill="#FFFFFF"/>',
        (
            "<style>text { font-family: 'DejaVu Sans', sans-serif; fill: #18323D; "
            "font-size: 20px; }</style>"
        ),
        (
            f'<text x="{center}" y="42" text-anchor="middle" font-size="28" '
            f'font-weight="600">Token-direction activation interference: {trained_method}</text>'
        ),
        *grid_lines,
        *y_labels,
        f'<polygon points="{area_points}" fill="#94D2BD" fill-opacity="0.38"/>',
        (
            f'<polyline points="{" ".join(points)}" fill="none" stroke="#005F73" '
            'stroke-width="2.2"/>'
        ),
        (
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{baseline}" '
            'stroke="#18323D" stroke-width="2"/>'
        ),
        (
            f'<line x1="{left}" y1="{baseline}" x2="{left + plot_width}" y2="{baseline}" '
            'stroke="#18323D" stroke-width="2"/>'
        ),
        f'<text x="{left}" y="{baseline + 35}" text-anchor="middle">1</text>',
        (
            f'<text x="{left + plot_width}" y="{baseline + 35}" text-anchor="middle">'
            f"{vocabulary_size}</text>"
        ),
        (
            f'<text x="{center}" y="{height - 32}" text-anchor="middle">'
            "Vocabulary tokens ranked by movement</text>"
        ),
        (
            f'<text x="35" y="{vertical_center}" text-anchor="middle" '
            f'transform="rotate(-90 35 {vertical_center})">'
            "Mean positive movement toward token direction</text>"
        ),
        "</svg>",
    ]
    svg = "\n".join(svg_lines) + "\n"
    temporary_plot = plot_path.with_suffix(".svg.tmp")
    temporary_plot.write_text(svg, encoding="utf-8")
    temporary_plot.replace(plot_path)
    return {
        "ranking_csv": str(ranking_path),
        "summary_json": str(summary_path),
        "plot_svg": str(plot_path),
    }
