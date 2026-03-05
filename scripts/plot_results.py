from __future__ import annotations

import argparse
import json
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


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Matrix summary not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Matrix summary must be a JSON list")
    return [row for row in payload if isinstance(row, dict)]


def _mean_by_keys(rows: list[dict[str, Any]], keys: list[str], value: str) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in rows:
        bucket = tuple(row.get(k) for k in keys)
        buckets[bucket].append(_to_float(row.get(value)))
    out: list[dict[str, Any]] = []
    for bucket, vals in buckets.items():
        item = {k: v for k, v in zip(keys, bucket)}
        item[value] = sum(vals) / len(vals) if vals else 0.0
        out.append(item)
    return out


def _line_plot(
    *,
    title: str,
    x_label: str,
    y_label: str,
    points: list[dict[str, Any]],
    x_key: str,
    y_key: str,
    series_keys: list[str],
    save_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    series: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in points:
        label = "/".join(str(row.get(k)) for k in series_keys)
        series[label].append((_to_float(row.get(x_key)), _to_float(row.get(y_key))))

    plt.figure(figsize=(10, 6))
    for label, vals in sorted(series.items()):
        vals = sorted(vals, key=lambda x: x[0])
        xs = [v[0] for v in vals]
        ys = [v[1] for v in vals]
        plt.plot(xs, ys, marker="o", label=label)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate plots from matrix summary JSON.")
    parser.add_argument(
        "--matrix-summary",
        type=str,
        default="results/baseline_lite_matrix_summary.json",
        help="Path to matrix summary JSON",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/analysis/plots",
        help="Directory to save plot images",
    )
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as _  # noqa: F401
    except Exception as exc:
        raise RuntimeError("matplotlib is required. Install with: pip install matplotlib") from exc

    rows = _load_rows(Path(args.matrix_summary))
    if not rows:
        raise RuntimeError("Matrix summary is empty")

    out_dir = Path(args.out_dir)

    recall_by_topk = _mean_by_keys(rows, ["backend", "strategy", "top_k"], "recall_at_k")
    _line_plot(
        title="Recall@k vs Top-k",
        x_label="top_k",
        y_label="recall_at_k",
        points=recall_by_topk,
        x_key="top_k",
        y_key="recall_at_k",
        series_keys=["backend", "strategy"],
        save_path=out_dir / "recall_vs_topk.png",
    )

    mrr_by_chunk = _mean_by_keys(rows, ["backend", "strategy", "chunk_size"], "mrr")
    _line_plot(
        title="MRR vs Chunk Size",
        x_label="chunk_size",
        y_label="mrr",
        points=mrr_by_chunk,
        x_key="chunk_size",
        y_key="mrr",
        series_keys=["backend", "strategy"],
        save_path=out_dir / "mrr_vs_chunk_size.png",
    )

    if any("f1" in row for row in rows):
        f1_by_chunk = _mean_by_keys(rows, ["backend", "strategy", "chunk_size"], "f1")
        _line_plot(
            title="F1 vs Chunk Size",
            x_label="chunk_size",
            y_label="f1",
            points=f1_by_chunk,
            x_key="chunk_size",
            y_key="f1",
            series_keys=["backend", "strategy"],
            save_path=out_dir / "f1_vs_chunk_size.png",
        )

    latency_by_topk = _mean_by_keys(rows, ["backend", "strategy", "top_k"], "avg_query_latency_ms")
    _line_plot(
        title="Latency vs Top-k",
        x_label="top_k",
        y_label="avg_query_latency_ms",
        points=latency_by_topk,
        x_key="top_k",
        y_key="avg_query_latency_ms",
        series_keys=["backend", "strategy"],
        save_path=out_dir / "latency_vs_topk.png",
    )

    print(f"Saved plots to: {out_dir}")


if __name__ == "__main__":
    main()

