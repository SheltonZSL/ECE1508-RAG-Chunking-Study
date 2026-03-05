from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _load_matrix_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Matrix summary not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Matrix summary must be a JSON list")
    return [row for row in payload if isinstance(row, dict)]


def _to_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def _aggregate_mean(rows: list[dict[str, Any]], group_keys: list[str]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(row.get(k) for k in group_keys)
        buckets[key].append(row)

    aggregated: list[dict[str, Any]] = []
    for key, items in buckets.items():
        out = {k: v for k, v in zip(group_keys, key)}
        out["count"] = len(items)
        out["recall_at_k"] = sum(_to_float(x.get("recall_at_k")) for x in items) / len(items)
        out["mrr"] = sum(_to_float(x.get("mrr")) for x in items) / len(items)
        out["avg_query_latency_ms"] = (
            sum(_to_float(x.get("avg_query_latency_ms")) for x in items) / len(items)
        )
        if any("em" in x for x in items):
            out["em"] = sum(_to_float(x.get("em")) for x in items) / len(items)
        if any("f1" in x for x in items):
            out["f1"] = sum(_to_float(x.get("f1")) for x in items) / len(items)
        aggregated.append(out)
    return aggregated


def _score_row(row: dict[str, Any]) -> float:
    # Retrieval-first ranking score for candidate selection.
    recall = _to_float(row.get("recall_at_k"))
    mrr = _to_float(row.get("mrr"))
    latency = _to_float(row.get("avg_query_latency_ms"))
    return recall + 0.5 * mrr - 0.001 * latency


def _top_candidates(rows: list[dict[str, Any]], top_n: int) -> list[dict[str, Any]]:
    ranked = []
    for row in rows:
        copied = dict(row)
        copied["selection_score"] = _score_row(row)
        ranked.append(copied)
    ranked.sort(key=lambda x: _to_float(x.get("selection_score")), reverse=True)
    return ranked[:top_n]


def _best_by_backend(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for row in rows:
        backend = str(row.get("backend", "unknown"))
        if backend not in best or _score_row(row) > _score_row(best[backend]):
            best[backend] = row
    return best


def _render_summary_markdown(
    *,
    matrix_rows: list[dict[str, Any]],
    best_backend: dict[str, dict[str, Any]],
    candidates: list[dict[str, Any]],
) -> str:
    lines: list[str] = []
    lines.append("# Experiment Summary")
    lines.append("")
    lines.append(f"- Total matrix runs: {len(matrix_rows)}")
    lines.append("")
    lines.append("## Best Retrieval Config By Backend")
    lines.append("")
    lines.append("| backend | strategy | chunk_size | overlap | top_k | recall@k | mrr | latency_ms |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for backend, row in best_backend.items():
        lines.append(
            "| "
            + f"{backend} | {row.get('strategy')} | {row.get('chunk_size')} | {row.get('overlap')} | "
            + f"{row.get('top_k')} | {round(_to_float(row.get('recall_at_k')), 4)} | "
            + f"{round(_to_float(row.get('mrr')), 4)} | {round(_to_float(row.get('avg_query_latency_ms')), 4)} |"
        )
    lines.append("")
    lines.append("## Recommended QA Candidate Runs")
    lines.append("")
    lines.append("| rank | backend | strategy | chunk_size | overlap | top_k | score |")
    lines.append("|---:|---|---|---:|---:|---:|---:|")
    for idx, row in enumerate(candidates, start=1):
        lines.append(
            "| "
            + f"{idx} | {row.get('backend')} | {row.get('strategy')} | {row.get('chunk_size')} | "
            + f"{row.get('overlap')} | {row.get('top_k')} | {round(_to_float(row.get('selection_score')), 4)} |"
        )
    lines.append("")
    lines.append("Run QA for these candidates by creating small config variants or by extending run_experiments filtering.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize matrix experiment outputs into report-friendly tables.")
    parser.add_argument(
        "--matrix-summary",
        type=str,
        default="results/baseline_lite_matrix_summary.json",
        help="Path to matrix summary JSON generated by run_experiments.py",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/analysis",
        help="Directory for generated CSV/Markdown summaries",
    )
    parser.add_argument("--top-n", type=int, default=8, help="Number of recommended QA candidates")
    args = parser.parse_args()

    matrix_path = Path(args.matrix_summary)
    out_dir = Path(args.out_dir)
    rows = _load_matrix_rows(matrix_path)
    if not rows:
        raise RuntimeError("Matrix summary is empty")

    runs_sorted = sorted(
        rows,
        key=lambda x: (_to_float(x.get("recall_at_k")), _to_float(x.get("mrr")), -_to_float(x.get("avg_query_latency_ms"))),
        reverse=True,
    )
    _write_csv(
        out_dir / "retrieval_runs_sorted.csv",
        runs_sorted,
        [
            "backend",
            "strategy",
            "chunk_size",
            "overlap",
            "top_k",
            "recall_at_k",
            "mrr",
            "avg_query_latency_ms",
            "em",
            "f1",
            "avg_context_char_len",
        ],
    )

    by_strategy = _aggregate_mean(rows, ["backend", "strategy"])
    by_strategy.sort(key=lambda x: (_to_float(x.get("recall_at_k")), _to_float(x.get("mrr"))), reverse=True)
    _write_csv(
        out_dir / "retrieval_by_strategy.csv",
        by_strategy,
        ["backend", "strategy", "count", "recall_at_k", "mrr", "avg_query_latency_ms", "em", "f1"],
    )

    by_topk = _aggregate_mean(rows, ["backend", "strategy", "top_k"])
    by_topk.sort(
        key=lambda x: (
            str(x.get("backend")),
            str(x.get("strategy")),
            _to_float(x.get("top_k")),
        )
    )
    _write_csv(
        out_dir / "retrieval_by_topk.csv",
        by_topk,
        ["backend", "strategy", "top_k", "count", "recall_at_k", "mrr", "avg_query_latency_ms", "em", "f1"],
    )

    by_chunk_size = _aggregate_mean(rows, ["backend", "strategy", "chunk_size"])
    by_chunk_size.sort(
        key=lambda x: (
            str(x.get("backend")),
            str(x.get("strategy")),
            _to_float(x.get("chunk_size")),
        )
    )
    _write_csv(
        out_dir / "retrieval_by_chunk_size.csv",
        by_chunk_size,
        [
            "backend",
            "strategy",
            "chunk_size",
            "count",
            "recall_at_k",
            "mrr",
            "avg_query_latency_ms",
            "em",
            "f1",
        ],
    )

    candidates = _top_candidates(rows, args.top_n)
    _write_csv(
        out_dir / "qa_candidates.csv",
        candidates,
        [
            "backend",
            "strategy",
            "chunk_size",
            "overlap",
            "top_k",
            "selection_score",
            "recall_at_k",
            "mrr",
            "avg_query_latency_ms",
            "em",
            "f1",
        ],
    )

    best_backend = _best_by_backend(rows)
    summary_md = _render_summary_markdown(
        matrix_rows=rows,
        best_backend=best_backend,
        candidates=candidates,
    )
    summary_path = out_dir / "analysis_summary.md"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(summary_md, encoding="utf-8")

    print(f"Wrote: {out_dir / 'retrieval_runs_sorted.csv'}")
    print(f"Wrote: {out_dir / 'retrieval_by_strategy.csv'}")
    print(f"Wrote: {out_dir / 'retrieval_by_topk.csv'}")
    print(f"Wrote: {out_dir / 'retrieval_by_chunk_size.csv'}")
    print(f"Wrote: {out_dir / 'qa_candidates.csv'}")
    print(f"Wrote: {summary_path}")


if __name__ == "__main__":
    main()

