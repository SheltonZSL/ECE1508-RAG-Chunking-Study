from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def _to_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "1", "yes"}:
            return True
        if text in {"false", "0", "no"}:
            return False
    return default


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Matrix summary not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Matrix summary must be a JSON list")
    return [row for row in payload if isinstance(row, dict)]


def _label_for_row(row: dict[str, Any]) -> str:
    backend = str(row.get("backend", "unknown"))
    strategy = str(row.get("strategy", "unknown"))
    reranker = "rerank" if _to_bool(row.get("reranker_enabled")) else "base"
    return f"{backend}/{strategy}/{reranker}"


def _prepare_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["top_k"] = _to_float(item.get("top_k"))
        item["chunk_size"] = _to_float(item.get("chunk_size"))
        item["overlap"] = _to_float(item.get("overlap"))
        item["recall_at_k"] = _to_float(item.get("recall_at_k"))
        item["mrr"] = _to_float(item.get("mrr"))
        item["f1"] = _to_float(item.get("f1"), float("nan"))
        item["avg_query_latency_ms"] = _to_float(item.get("avg_query_latency_ms"))
        item["reranker_enabled"] = _to_bool(item.get("reranker_enabled"))
        item["label"] = _label_for_row(item)
        prepared.append(item)
    return prepared


def _mean_by(
    rows: list[dict[str, Any]],
    group_keys: list[str],
    value_key: str,
) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in rows:
        value = _to_float(row.get(value_key), float("nan"))
        if math.isnan(value):
            continue
        bucket = tuple(row.get(key) for key in group_keys)
        buckets[bucket].append(value)

    points: list[dict[str, Any]] = []
    for bucket, values in buckets.items():
        if not values:
            continue
        item = {key: val for key, val in zip(group_keys, bucket)}
        item[value_key] = sum(values) / len(values)
        points.append(item)
    return points


def _plot_quality_latency(rows: list[dict[str, Any]], out_dir: Path, quality_key: str) -> Path:
    import matplotlib.pyplot as plt

    backends = sorted({str(row.get("backend", "unknown")) for row in rows})
    backend_colors = {backend: f"C{idx % 10}" for idx, backend in enumerate(backends)}
    marker_map = {"fixed": "o", "structure": "s", "adaptive": "^"}

    fig, ax = plt.subplots(figsize=(10, 6))
    for row in rows:
        quality = _to_float(row.get(quality_key), float("nan"))
        latency = _to_float(row.get("avg_query_latency_ms"), float("nan"))
        if math.isnan(quality) or math.isnan(latency):
            continue
        backend = str(row.get("backend", "unknown"))
        strategy = str(row.get("strategy", "unknown"))
        ax.scatter(
            latency,
            quality,
            s=48,
            c=backend_colors.get(backend, "C0"),
            marker=marker_map.get(strategy, "o"),
            alpha=0.8,
            edgecolors="black",
            linewidths=0.3,
        )

    ax.set_title(f"Quality-Latency Tradeoff ({quality_key})")
    ax.set_xlabel("Average Query Latency (ms)")
    ax.set_ylabel(quality_key)
    ax.grid(alpha=0.25)

    handles = []
    labels = []
    for backend in backends:
        handles.append(plt.Line2D([], [], color=backend_colors[backend], marker="o", linestyle="None"))
        labels.append(f"backend={backend}")
    for strategy, marker in marker_map.items():
        handles.append(
            plt.Line2D([], [], color="black", marker=marker, linestyle="None", markerfacecolor="white")
        )
        labels.append(f"strategy={strategy}")
    if handles:
        ax.legend(handles, labels, fontsize=8, loc="best")

    out_path = out_dir / f"quality_latency_{quality_key}.png"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return out_path


def _plot_topk_sensitivity(rows: list[dict[str, Any]], out_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    output_paths: list[Path] = []
    metrics = ["recall_at_k", "mrr"]
    if any(not math.isnan(_to_float(row.get("f1"), float("nan"))) for row in rows):
        metrics.append("f1")

    labels = sorted({str(row["label"]) for row in rows})
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(10, 6))
        for label in labels:
            points = _mean_by(
                [row for row in rows if str(row.get("label")) == label],
                group_keys=["top_k"],
                value_key=metric,
            )
            points = sorted(points, key=lambda item: _to_float(item.get("top_k")))
            if not points:
                continue
            xs = [_to_float(item.get("top_k")) for item in points]
            ys = [_to_float(item.get(metric)) for item in points]
            ax.plot(xs, ys, marker="o", linewidth=1.8, label=label)

        ax.set_title(f"{metric} Sensitivity to Top-k")
        ax.set_xlabel("top_k")
        ax.set_ylabel(metric)
        ax.grid(alpha=0.3)
        if labels:
            ax.legend(fontsize=7, ncol=2)

        out_path = out_dir / f"topk_sensitivity_{metric}.png"
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out_path, dpi=170)
        plt.close(fig)
        output_paths.append(out_path)
    return output_paths


def _plot_chunk_overlap_heatmaps(rows: list[dict[str, Any]], out_dir: Path, metric: str) -> list[Path]:
    import matplotlib.pyplot as plt
    import numpy as np

    output_paths: list[Path] = []
    groups = defaultdict(list)
    for row in rows:
        key = (
            str(row.get("backend", "unknown")),
            str(row.get("strategy", "unknown")),
            bool(row.get("reranker_enabled", False)),
        )
        groups[key].append(row)

    for (backend, strategy, reranker_enabled), group_rows in sorted(groups.items()):
        points = _mean_by(group_rows, ["chunk_size", "overlap"], metric)
        if not points:
            continue
        xs = sorted({int(_to_float(item.get("chunk_size"))) for item in points})
        ys = sorted({int(_to_float(item.get("overlap"))) for item in points})
        if not xs or not ys:
            continue
        matrix = np.full((len(ys), len(xs)), np.nan, dtype=float)
        for item in points:
            x = int(_to_float(item.get("chunk_size")))
            y = int(_to_float(item.get("overlap")))
            value = _to_float(item.get(metric), float("nan"))
            if math.isnan(value):
                continue
            xi = xs.index(x)
            yi = ys.index(y)
            matrix[yi, xi] = value

        fig, ax = plt.subplots(figsize=(8, 5))
        image = ax.imshow(matrix, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(xs)), [str(v) for v in xs])
        ax.set_yticks(range(len(ys)), [str(v) for v in ys])
        ax.set_xlabel("chunk_size")
        ax.set_ylabel("overlap")
        rerank_label = "rerank" if reranker_enabled else "base"
        ax.set_title(f"{metric} Heatmap ({backend}/{strategy}/{rerank_label})")

        for yi in range(len(ys)):
            for xi in range(len(xs)):
                value = matrix[yi, xi]
                if math.isnan(value):
                    continue
                ax.text(xi, yi, f"{value:.3f}", ha="center", va="center", fontsize=7, color="white")

        fig.colorbar(image, ax=ax, shrink=0.9)
        out_path = out_dir / f"heatmap_{metric}_{backend}_{strategy}_{rerank_label}.png"
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out_path, dpi=170)
        plt.close(fig)
        output_paths.append(out_path)
    return output_paths


def _write_summary(rows: list[dict[str, Any]], out_dir: Path, quality_key: str, plots: list[Path]) -> Path:
    if not rows:
        raise RuntimeError("No rows available for summary")

    quality_rows = [
        row for row in rows if not math.isnan(_to_float(row.get(quality_key), float("nan")))
    ]
    latency_rows = [
        row for row in rows if not math.isnan(_to_float(row.get("avg_query_latency_ms"), float("nan")))
    ]
    best_quality = max(quality_rows, key=lambda row: _to_float(row.get(quality_key))) if quality_rows else None
    best_latency = min(latency_rows, key=lambda row: _to_float(row.get("avg_query_latency_ms"))) if latency_rows else None

    lines = ["# Sensitivity Analysis Summary", ""]
    lines.append(f"- Total matrix rows: `{len(rows)}`")
    lines.append(f"- Quality metric used for tradeoff ranking: `{quality_key}`")
    lines.append("")
    if best_quality is not None:
        lines.append("## Best Quality Setting")
        lines.append(
            "- "
            + ", ".join(
                [
                    f"backend={best_quality.get('backend')}",
                    f"strategy={best_quality.get('strategy')}",
                    f"chunk_size={int(_to_float(best_quality.get('chunk_size')))}",
                    f"overlap={int(_to_float(best_quality.get('overlap')))}",
                    f"top_k={int(_to_float(best_quality.get('top_k')))}",
                    f"reranker={bool(best_quality.get('reranker_enabled', False))}",
                    f"{quality_key}={_to_float(best_quality.get(quality_key)):.4f}",
                ]
            )
        )
        lines.append("")
    if best_latency is not None:
        lines.append("## Fastest Setting")
        lines.append(
            "- "
            + ", ".join(
                [
                    f"backend={best_latency.get('backend')}",
                    f"strategy={best_latency.get('strategy')}",
                    f"chunk_size={int(_to_float(best_latency.get('chunk_size')))}",
                    f"overlap={int(_to_float(best_latency.get('overlap')))}",
                    f"top_k={int(_to_float(best_latency.get('top_k')))}",
                    f"reranker={bool(best_latency.get('reranker_enabled', False))}",
                    f"avg_query_latency_ms={_to_float(best_latency.get('avg_query_latency_ms')):.2f}",
                ]
            )
        )
        lines.append("")

    lines.append("## Generated Plots")
    for plot in sorted(plots):
        lines.append(f"- `{plot.as_posix()}`")
    lines.append("")

    out_path = out_dir / "sensitivity_summary.md"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deeper sensitivity analysis plots.")
    parser.add_argument(
        "--matrix-summary",
        type=str,
        nargs="+",
        default=["results/baseline_lite_matrix_summary.json"],
        help="One or more matrix summary JSON files",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/analysis/sensitivity",
        help="Directory to save sensitivity analysis outputs",
    )
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as _  # noqa: F401
    except Exception as exc:
        raise RuntimeError("matplotlib is required. Install with: pip install matplotlib") from exc

    merged_rows: list[dict[str, Any]] = []
    for summary_path in args.matrix_summary:
        merged_rows.extend(_load_rows(Path(summary_path)))
    rows = _prepare_rows(merged_rows)
    if not rows:
        raise RuntimeError("Matrix summary is empty")
    out_dir = Path(args.out_dir)

    quality_key = "f1" if any(not math.isnan(_to_float(row.get("f1"), float("nan"))) for row in rows) else "mrr"
    generated_plots: list[Path] = []
    generated_plots.append(_plot_quality_latency(rows, out_dir, quality_key))
    generated_plots.extend(_plot_topk_sensitivity(rows, out_dir))
    generated_plots.extend(_plot_chunk_overlap_heatmaps(rows, out_dir, "mrr"))
    if quality_key == "f1":
        generated_plots.extend(_plot_chunk_overlap_heatmaps(rows, out_dir, "f1"))

    summary_path = _write_summary(rows, out_dir, quality_key, generated_plots)
    print(f"Saved sensitivity analysis to: {out_dir}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
