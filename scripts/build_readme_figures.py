from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_rows(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected list payload: {path}")
    return [row for row in payload if isinstance(row, dict)]


def _f(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _best_by_backend_topk(rows: list[dict]) -> list[dict]:
    best: dict[tuple[str, int], dict] = {}
    for row in rows:
        backend = str(row.get("backend", "unknown"))
        top_k = int(_f(row.get("top_k")))
        key = (backend, top_k)
        score = (
            _f(row.get("recall_at_k")) * 0.45
            + _f(row.get("mrr")) * 0.25
            + _f(row.get("f1")) * 0.25
            - _f(row.get("avg_query_latency_ms")) * 0.01
        )
        current = best.get(key)
        if current is None or score > current["_score"]:
            item = dict(row)
            item["_score"] = score
            best[key] = item
    return sorted(best.values(), key=lambda row: (str(row["backend"]), int(_f(row["top_k"]))))


def _best_by_strategy(rows: list[dict], top_k: int) -> list[dict]:
    best: dict[str, dict] = {}
    for row in rows:
        if int(_f(row.get("top_k"))) != top_k:
            continue
        strategy = str(row.get("strategy", "unknown"))
        score = (
            _f(row.get("f1")) * 0.6
            + _f(row.get("mrr")) * 0.25
            + _f(row.get("recall_at_k")) * 0.15
            - _f(row.get("avg_query_latency_ms")) * 0.002
        )
        current = best.get(strategy)
        if current is None or score > current["_score"]:
            item = dict(row)
            item["_score"] = score
            best[strategy] = item
    return [best[key] for key in sorted(best)]


def _reranker_deltas(rows: list[dict]) -> list[dict]:
    baseline = next((row for row in rows if not bool(row.get("reranker_enabled"))), None)
    if baseline is None:
        raise RuntimeError("Could not find reranker baseline row.")
    out = []
    for row in rows:
        if not bool(row.get("reranker_enabled")):
            continue
        item = dict(row)
        item["delta_f1"] = _f(row.get("f1")) - _f(baseline.get("f1"))
        item["delta_latency_ms"] = _f(row.get("avg_query_latency_ms")) - _f(
            baseline.get("avg_query_latency_ms")
        )
        out.append(item)
    return out


def _style():
    import matplotlib.pyplot as plt

    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "axes.edgecolor": "#CBD5E1",
            "axes.linewidth": 1.0,
            "axes.facecolor": "#FFFFFF",
            "figure.facecolor": "#F8FAFC",
            "savefig.facecolor": "#F8FAFC",
            "grid.color": "#E2E8F0",
            "grid.alpha": 0.9,
            "grid.linestyle": "-",
        }
    )


def _plot_backend_scaling(rows: list[dict], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    _style()
    colors = {"dense": "#2563EB", "bm25": "#F97316"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    by_backend: dict[str, list[dict]] = {}
    for row in rows:
        by_backend.setdefault(str(row["backend"]), []).append(row)

    for backend, backend_rows in by_backend.items():
        backend_rows = sorted(backend_rows, key=lambda row: int(_f(row["top_k"])))
        xs = [int(_f(row["top_k"])) for row in backend_rows]
        recall = [_f(row["recall_at_k"]) for row in backend_rows]
        latency = [_f(row["avg_query_latency_ms"]) for row in backend_rows]
        axes[0].plot(xs, recall, marker="o", linewidth=2.6, color=colors.get(backend, "#334155"), label=backend)
        axes[1].plot(xs, latency, marker="o", linewidth=2.6, color=colors.get(backend, "#334155"), label=backend)

        for x, y in zip(xs, recall):
            axes[0].annotate(f"{y:.02f}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center")
        for x, y in zip(xs, latency):
            axes[1].annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center")

    axes[0].set_title("Retrieval Coverage vs Top-k")
    axes[0].set_xlabel("Top-k")
    axes[0].set_ylabel("Recall@k")
    axes[0].grid(True)

    axes[1].set_title("Latency vs Top-k")
    axes[1].set_xlabel("Top-k")
    axes[1].set_ylabel("Avg Query Latency (ms)")
    axes[1].grid(True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Backend Scaling Under Increasing Retrieval Depth", fontsize=16, fontweight="bold", y=1.06)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_strategy_comparison(rows: list[dict], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    _style()
    labels = [str(row["strategy"]).title() for row in rows]
    f1_vals = [_f(row["f1"]) for row in rows]
    latency_vals = [_f(row["avg_query_latency_ms"]) for row in rows]
    recall_vals = [_f(row["recall_at_k"]) for row in rows]
    colors = ["#0F766E", "#7C3AED", "#EA580C"]
    xs = list(range(len(rows)))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    bars = axes[0].bar(xs, f1_vals, color=colors, width=0.62)
    axes[0].plot(xs, recall_vals, color="#1E293B", marker="o", linewidth=2.2, label="Recall@10")
    axes[0].set_xticks(xs, labels)
    axes[0].set_title("Chunking Strategy Quality at Top-k = 10 (BM25)")
    axes[0].set_ylabel("F1")
    axes[0].grid(True, axis="y")
    axes[0].legend(frameon=False)
    for bar, value in zip(bars, f1_vals):
        axes[0].annotate(f"{value:.4f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()), textcoords="offset points", xytext=(0, 6), ha="center")

    bars2 = axes[1].bar(xs, latency_vals, color=colors, width=0.62)
    axes[1].set_xticks(xs, labels)
    axes[1].set_title("Chunking Strategy Latency at Top-k = 10 (BM25)")
    axes[1].set_ylabel("Avg Query Latency (ms)")
    axes[1].grid(True, axis="y")
    for bar, value in zip(bars2, latency_vals):
        axes[1].annotate(f"{value:.3f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()), textcoords="offset points", xytext=(0, 6), ha="center")

    fig.suptitle("Chunking Strategy Comparison Under a Fixed Retrieval Depth", fontsize=16, fontweight="bold", y=1.04)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_reranker_tradeoff(rows: list[dict], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    _style()
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    best = None
    best_score = None

    for row in rows:
        x = _f(row["delta_latency_ms"])
        y = _f(row["delta_f1"])
        label = f"k={int(_f(row['reranker_candidate_k']))}, a={_f(row['reranker_alpha']):.1f}"
        score = y - max(x, 0.0) * 0.02
        if best is None or score > best_score:
            best = row
            best_score = score

        ax.scatter(
            x,
            y,
            s=120,
            color="#DC2626" if row is best else "#2563EB",
            alpha=0.88,
            edgecolors="#0F172A",
            linewidths=0.5,
        )
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(6, 6), fontsize=9)

    ax.axhline(0.0, color="#94A3B8", linewidth=1.2)
    ax.axvline(0.0, color="#94A3B8", linewidth=1.2)
    ax.set_title("Reranker Quality-Latency Tradeoff", fontsize=16, fontweight="bold")
    ax.set_xlabel("Latency Change vs Baseline (ms)")
    ax.set_ylabel("F1 Change vs Baseline")
    ax.grid(True)

    if best is not None:
        ax.text(
            0.02,
            0.98,
            (
                "Best observed tradeoff: "
                f"k={int(_f(best['reranker_candidate_k']))}, "
                f"alpha={_f(best['reranker_alpha']):.1f}, "
                f"delta_f1={_f(best['delta_f1']):+.4f}, "
                f"delta_latency={_f(best['delta_latency_ms']):+.3f} ms"
            ),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"facecolor": "#FFFFFF", "edgecolor": "#CBD5E1", "boxstyle": "round,pad=0.35"},
        )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    dense_rows = _load_rows(ROOT / "results" / "summaries" / "baseline_lite_matrix_summary.json")
    bm25_rows = _load_rows(ROOT / "results" / "summaries" / "baseline_lite_bm25_matrix_summary.json")
    reranker_rows = _load_rows(ROOT / "results" / "reranker_ablation_lite_matrix_summary.json")

    backend_rows = _best_by_backend_topk(dense_rows + bm25_rows)
    strategy_rows = _best_by_strategy(bm25_rows, top_k=10)
    rerank_rows = _reranker_deltas(reranker_rows)

    out_dir = ROOT / "docs" / "assets"
    _plot_backend_scaling(backend_rows, out_dir / "backend_topk_scaling.png")
    _plot_strategy_comparison(strategy_rows, out_dir / "bm25_strategy_comparison.png")
    _plot_reranker_tradeoff(rerank_rows, out_dir / "reranker_tradeoff.png")
    print(f"Saved README figures to: {out_dir}")


if __name__ == "__main__":
    main()
