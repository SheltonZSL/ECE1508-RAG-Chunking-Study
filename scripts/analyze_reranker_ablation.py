from __future__ import annotations

import argparse
import csv
import json
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


def _key(row: dict[str, Any]) -> tuple[str, str, int, int, int]:
    return (
        str(row.get("backend", "")),
        str(row.get("strategy", "")),
        int(_to_float(row.get("chunk_size"))),
        int(_to_float(row.get("overlap"))),
        int(_to_float(row.get("top_k"))),
    )


def _delta(current: dict[str, Any], base: dict[str, Any], metric: str) -> float:
    return _to_float(current.get(metric)) - _to_float(base.get(metric))


def _quality_gain(row: dict[str, Any]) -> float:
    delta_f1 = _to_float(row.get("delta_f1"))
    if abs(delta_f1) > 1e-12:
        return delta_f1
    return _to_float(row.get("delta_mrr"))


def _tradeoff_score(row: dict[str, Any]) -> float:
    """
    Stable quality-latency tradeoff score.
    - If latency increases, use gain per added ms.
    - If latency is unchanged or decreases, use raw quality gain (avoid exploding ratios).
    """
    gain = _quality_gain(row)
    delta_latency = _to_float(row.get("delta_latency_ms"))
    if delta_latency > 0:
        return gain / delta_latency
    return gain


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "backend",
        "strategy",
        "chunk_size",
        "overlap",
        "top_k",
        "reranker_type",
        "reranker_candidate_k",
        "reranker_alpha",
        "recall_at_k",
        "mrr",
        "em",
        "f1",
        "avg_query_latency_ms",
        "delta_recall_at_k",
        "delta_mrr",
        "delta_em",
        "delta_f1",
        "delta_latency_ms",
        "quality_gain_per_ms",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


def _plot_ablation_scatter(rows: list[dict[str, Any]], out_path: Path) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    xs = [_to_float(row.get("delta_latency_ms")) for row in rows]
    ys = [_to_float(row.get("delta_f1")) for row in rows]
    if all(abs(y) < 1e-12 for y in ys):
        ys = [_to_float(row.get("delta_mrr")) for row in rows]
        ylabel = "Delta MRR"
    else:
        ylabel = "Delta F1"

    labels = [
        f"c{int(_to_float(row.get('reranker_candidate_k')))} a{_to_float(row.get('reranker_alpha')):.1f}"
        for row in rows
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(0.0, color="gray", linewidth=1.0, alpha=0.6)
    ax.axvline(0.0, color="gray", linewidth=1.0, alpha=0.6)
    ax.scatter(xs, ys, c="C0", alpha=0.85)
    for x, y, label in zip(xs, ys, labels):
        ax.annotate(label, (x, y), fontsize=7, alpha=0.8)

    ax.set_title("Reranker Ablation: Quality Gain vs Latency Cost")
    ax.set_xlabel("Delta Latency (ms)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return True


def _build_markdown(rows: list[dict[str, Any]]) -> str:
    lines = ["# Reranker Ablation Summary", ""]
    lines.append(f"- Compared reranker runs: `{len(rows)}`")
    lines.append(
        "- Tradeoff score rule: gain/ms when latency increases; otherwise use raw gain (stable, no exploding ratios)."
    )
    lines.append("")
    if not rows:
        lines.append("No reranker rows with matching no-reranker baseline were found.")
        return "\n".join(lines)

    has_f1 = any(abs(_to_float(row.get("delta_f1"))) > 1e-12 for row in rows)
    quality_key = "delta_f1" if has_f1 else "delta_mrr"
    best_quality = max(rows, key=lambda row: _to_float(row.get(quality_key)))
    best_tradeoff = max(rows, key=_tradeoff_score)

    lines.append("## Best Quality Gain")
    lines.append(
        "- "
        + ", ".join(
            [
                f"candidate_k={int(_to_float(best_quality.get('reranker_candidate_k')))}",
                f"alpha={_to_float(best_quality.get('reranker_alpha')):.2f}",
                f"{quality_key}={_to_float(best_quality.get(quality_key)):.4f}",
                f"delta_latency_ms={_to_float(best_quality.get('delta_latency_ms')):.3f}",
            ]
        )
    )
    lines.append("")

    lines.append("## Best Quality/Latency Tradeoff")
    lines.append(
        "- "
        + ", ".join(
            [
                f"candidate_k={int(_to_float(best_tradeoff.get('reranker_candidate_k')))}",
                f"alpha={_to_float(best_tradeoff.get('reranker_alpha')):.2f}",
                f"tradeoff_score={_to_float(best_tradeoff.get('quality_gain_per_ms')):.6f}",
                f"delta_f1={_to_float(best_tradeoff.get('delta_f1')):.4f}",
                f"delta_mrr={_to_float(best_tradeoff.get('delta_mrr')):.4f}",
                f"delta_latency_ms={_to_float(best_tradeoff.get('delta_latency_ms')):.3f}",
            ]
        )
    )
    lines.append("")

    lines.append("## Top 5 Settings")
    lines.append("")
    lines.append("| rank | candidate_k | alpha | delta_f1 | delta_mrr | delta_recall@k | delta_latency_ms |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    sorted_rows = sorted(rows, key=lambda row: _to_float(row.get(quality_key)), reverse=True)
    for idx, row in enumerate(sorted_rows[:5], start=1):
        lines.append(
            "| "
            + f"{idx} | {int(_to_float(row.get('reranker_candidate_k')))} | "
            + f"{_to_float(row.get('reranker_alpha')):.2f} | "
            + f"{_to_float(row.get('delta_f1')):.4f} | "
            + f"{_to_float(row.get('delta_mrr')):.4f} | "
            + f"{_to_float(row.get('delta_recall_at_k')):.4f} | "
            + f"{_to_float(row.get('delta_latency_ms')):.3f} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze reranker ablation from matrix summary JSON.")
    parser.add_argument(
        "--matrix-summary",
        type=str,
        default="results/reranker_ablation_lite_matrix_summary.json",
        help="Path to matrix summary JSON",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/analysis/reranker_ablation",
        help="Output directory for reranker ablation artifacts",
    )
    args = parser.parse_args()

    rows = _load_rows(Path(args.matrix_summary))
    base_rows = { _key(row): row for row in rows if not _to_bool(row.get("reranker_enabled")) }
    compared: list[dict[str, Any]] = []

    for row in rows:
        if not _to_bool(row.get("reranker_enabled")):
            continue
        key = _key(row)
        base = base_rows.get(key)
        if base is None:
            continue
        enriched = dict(row)
        enriched["delta_recall_at_k"] = _delta(row, base, "recall_at_k")
        enriched["delta_mrr"] = _delta(row, base, "mrr")
        enriched["delta_em"] = _delta(row, base, "em")
        enriched["delta_f1"] = _delta(row, base, "f1")
        enriched["delta_latency_ms"] = _delta(row, base, "avg_query_latency_ms")
        enriched["quality_gain_per_ms"] = _tradeoff_score(enriched)
        compared.append(enriched)

    compared.sort(key=lambda row: (_to_float(row.get("delta_f1")), _to_float(row.get("delta_mrr"))), reverse=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "reranker_ablation_table.csv"
    md_path = out_dir / "reranker_ablation_summary.md"
    plot_path = out_dir / "reranker_ablation_scatter.png"

    _write_csv(csv_path, compared)
    md_path.write_text(_build_markdown(compared), encoding="utf-8")
    plotted = _plot_ablation_scatter(compared, plot_path)

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {md_path}")
    if plotted:
        print(f"Wrote: {plot_path}")
    else:
        print("Skipped scatter plot (matplotlib not available).")


if __name__ == "__main__":
    main()
