from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _to_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_round(value: Any, digits: int = 4) -> float:
    return round(_to_float(value), digits)


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


def _score(row: dict[str, Any]) -> float:
    recall = _to_float(row.get("recall_at_k"))
    mrr = _to_float(row.get("mrr"))
    f1 = _to_float(row.get("f1"))
    latency = _to_float(row.get("avg_query_latency_ms"))
    return recall * 0.5 + mrr * 0.3 + f1 * 0.2 - latency * 0.001


def _load_matrix_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Matrix summary must be a list: {path}")
    rows: list[dict[str, Any]] = []
    for raw in payload:
        if not isinstance(raw, dict):
            continue
        row = dict(raw)
        row["source_file"] = str(path.as_posix())
        row["backend"] = str(row.get("backend", "")).strip().lower()
        row["strategy"] = str(row.get("strategy", "")).strip().lower()
        row["reranker_enabled"] = _to_bool(row.get("reranker_enabled"), False)
        row["reranker_type"] = (
            str(row.get("reranker_type", "none")).strip().lower()
            if row["reranker_enabled"]
            else "none"
        )
        row["reranker_candidate_k"] = int(_to_float(row.get("reranker_candidate_k"), 0.0))
        row["reranker_alpha"] = _to_float(row.get("reranker_alpha"), 0.0)
        row["chunk_size"] = int(_to_float(row.get("chunk_size")))
        row["overlap"] = int(_to_float(row.get("overlap")))
        row["top_k"] = int(_to_float(row.get("top_k")))
        row["composite_score"] = _score(row)
        rows.append(row)
    return rows


def discover_matrix_files(results_dir: Path) -> list[Path]:
    summary_dir = results_dir / "summaries"
    candidates = list(summary_dir.glob("*_matrix_summary.json")) if summary_dir.exists() else []
    if not candidates:
        candidates = list(results_dir.glob("*_matrix_summary.json"))
    return sorted({path.resolve() for path in candidates})


def collect_rows(paths: list[Path]) -> list[dict[str, Any]]:
    merged: dict[tuple[Any, ...], dict[str, Any]] = {}
    for path in paths:
        for row in _load_matrix_rows(path):
            if not row.get("backend") or not row.get("strategy"):
                continue
            key = (
                row.get("backend"),
                row.get("strategy"),
                row.get("reranker_enabled"),
                row.get("reranker_type"),
                row.get("reranker_candidate_k"),
                row.get("reranker_alpha"),
                row.get("chunk_size"),
                row.get("overlap"),
                row.get("top_k"),
            )
            existing = merged.get(key)
            if existing is None:
                merged[key] = row
                continue
            existing_has_qa = _to_float(existing.get("f1"), -1.0) >= 0.0 and _to_float(
                existing.get("em"), -1.0
            ) >= 0.0
            current_has_qa = _to_float(row.get("f1"), -1.0) >= 0.0 and _to_float(row.get("em"), -1.0) >= 0.0
            if current_has_qa and not existing_has_qa:
                merged[key] = row
    return list(merged.values())


def _best_by_group(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        group_key = tuple(row.get(key) for key in keys)
        current = grouped.get(group_key)
        if current is None or _score(row) > _score(current):
            grouped[group_key] = row
    return sorted(grouped.values(), key=lambda item: _score(item), reverse=True)


def _mean_by_group(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[tuple(row.get(key) for key in keys)].append(row)

    summary: list[dict[str, Any]] = []
    for group_key, items in buckets.items():
        base = {k: v for k, v in zip(keys, group_key)}
        base["count"] = len(items)
        base["recall_at_k"] = sum(_to_float(item.get("recall_at_k")) for item in items) / len(items)
        base["mrr"] = sum(_to_float(item.get("mrr")) for item in items) / len(items)
        base["f1"] = sum(_to_float(item.get("f1")) for item in items) / len(items)
        base["avg_query_latency_ms"] = (
            sum(_to_float(item.get("avg_query_latency_ms")) for item in items) / len(items)
        )
        base["composite_score"] = _score(base)
        summary.append(base)
    summary.sort(key=lambda item: _score(item), reverse=True)
    return summary


def write_comparison_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "backend",
        "strategy",
        "reranker_enabled",
        "reranker_type",
        "reranker_candidate_k",
        "reranker_alpha",
        "chunk_size",
        "overlap",
        "top_k",
        "recall_at_k",
        "mrr",
        "em",
        "f1",
        "avg_query_latency_ms",
        "composite_score",
        "source_file",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: _score(item), reverse=True):
            writer.writerow(
                {
                    "backend": row.get("backend"),
                    "strategy": row.get("strategy"),
                    "reranker_enabled": row.get("reranker_enabled", False),
                    "reranker_type": row.get("reranker_type", "none"),
                    "reranker_candidate_k": int(_to_float(row.get("reranker_candidate_k"))),
                    "reranker_alpha": _safe_round(row.get("reranker_alpha"), 3),
                    "chunk_size": int(_to_float(row.get("chunk_size"))),
                    "overlap": int(_to_float(row.get("overlap"))),
                    "top_k": int(_to_float(row.get("top_k"))),
                    "recall_at_k": _safe_round(row.get("recall_at_k"), 4),
                    "mrr": _safe_round(row.get("mrr"), 4),
                    "em": _safe_round(row.get("em"), 4),
                    "f1": _safe_round(row.get("f1"), 4),
                    "avg_query_latency_ms": _safe_round(row.get("avg_query_latency_ms"), 3),
                    "composite_score": _safe_round(row.get("composite_score"), 4),
                    "source_file": row.get("source_file"),
                }
            )


def _render_markdown_table(rows: list[dict[str, Any]], include_source: bool = False, limit: int = 10) -> list[str]:
    header = [
        "| rank | backend | strategy | reranker | chunk | overlap | top_k | recall@k | mrr | f1 | latency(ms) | score |"
    ]
    if include_source:
        header = [
            "| rank | backend | strategy | reranker | chunk | overlap | top_k | recall@k | mrr | f1 | latency(ms) | score | source |"
        ]
    sep = [
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    ]
    if include_source:
        sep = [
            "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"
        ]

    lines = header + sep
    for idx, row in enumerate(rows[:limit], start=1):
        base = (
            f"| {idx} | {row.get('backend')} | {row.get('strategy')} | "
            f"{row.get('reranker_enabled', False)} | "
            f"{int(_to_float(row.get('chunk_size')))} | {int(_to_float(row.get('overlap')))} | "
            f"{int(_to_float(row.get('top_k')))} | {_safe_round(row.get('recall_at_k'), 4)} | "
            f"{_safe_round(row.get('mrr'), 4)} | {_safe_round(row.get('f1'), 4)} | "
            f"{_safe_round(row.get('avg_query_latency_ms'), 3)} | {_safe_round(_score(row), 4)}"
        )
        if include_source:
            base += f" | `{Path(str(row.get('source_file', ''))).name}` |"
        else:
            base += " |"
        lines.append(base)
    return lines


def build_final_report(
    *,
    source_files: list[Path],
    rows: list[dict[str, Any]],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ranked = sorted(rows, key=lambda item: _score(item), reverse=True)
    best_overall = ranked[0] if ranked else None
    best_backend = _best_by_group(rows, ["backend"])
    best_strategy = _best_by_group(rows, ["strategy"])
    strategy_means = _mean_by_group(rows, ["backend", "strategy"])

    has_ci = any(
        any(key in row for key in ("recall_at_k_ci_low", "mrr_ci_low", "em_ci_low", "f1_ci_low"))
        for row in rows
    )
    has_qa = any(_to_float(row.get("f1"), -1.0) >= 0.0 for row in rows)
    timestamp = datetime.now(timezone.utc).isoformat()

    lines: list[str] = []
    lines.append("# Final Report")
    lines.append("")
    lines.append(f"- Generated at (UTC): `{timestamp}`")
    lines.append(f"- Matrix summary files: `{len(source_files)}`")
    lines.append(f"- Unique experiment rows: `{len(rows)}`")
    lines.append(f"- QA metrics available: `{has_qa}`")
    lines.append(f"- CI fields detected in source rows: `{has_ci}`")
    lines.append("")

    if best_overall is not None:
        lines.append("## Best Overall Run")
        lines.append("")
        lines.append(
            f"- `{best_overall.get('backend')}/{best_overall.get('strategy')}`"
            f" rerank={best_overall.get('reranker_enabled', False)}"
            f" c{int(_to_float(best_overall.get('chunk_size')))}"
            f" o{int(_to_float(best_overall.get('overlap')))}"
            f" k{int(_to_float(best_overall.get('top_k')))}"
            f" | recall@k={_safe_round(best_overall.get('recall_at_k'))}"
            f" | mrr={_safe_round(best_overall.get('mrr'))}"
            f" | f1={_safe_round(best_overall.get('f1'))}"
            f" | latency={_safe_round(best_overall.get('avg_query_latency_ms'), 3)} ms"
            f" | score={_safe_round(_score(best_overall))}"
        )
        lines.append("")

    lines.append("## Top 10 Runs")
    lines.append("")
    lines.extend(_render_markdown_table(ranked, include_source=True, limit=10))
    lines.append("")

    lines.append("## Best Per Backend")
    lines.append("")
    lines.extend(_render_markdown_table(best_backend, include_source=False, limit=20))
    lines.append("")

    lines.append("## Best Per Strategy")
    lines.append("")
    lines.extend(_render_markdown_table(best_strategy, include_source=False, limit=20))
    lines.append("")

    lines.append("## Mean Performance By Backend/Strategy")
    lines.append("")
    lines.extend(_render_markdown_table(strategy_means, include_source=False, limit=40))
    lines.append("")

    lines.append("## Data Sources")
    lines.append("")
    for path in source_files:
        lines.append(f"- `{path.as_posix()}`")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build final comparison artifacts from matrix summary files.")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Root results directory containing summaries/",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/analysis",
        help="Output directory for final report artifacts.",
    )
    parser.add_argument(
        "--matrix-summary",
        type=str,
        default=None,
        help="Optional single matrix summary file. If omitted, auto-discover in results/summaries.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    if args.matrix_summary:
        source_files = [Path(args.matrix_summary).resolve()]
    else:
        source_files = discover_matrix_files(results_dir)
    if not source_files:
        raise FileNotFoundError("No matrix summary files found. Run scripts/run_experiments.py first.")

    rows = collect_rows(source_files)
    if not rows:
        raise RuntimeError("No valid rows found in matrix summary files.")

    comparison_path = out_dir / "comparison_table.csv"
    report_path = out_dir / "final_report.md"
    write_comparison_csv(comparison_path, rows)
    build_final_report(source_files=source_files, rows=rows, out_path=report_path)

    print(f"Wrote comparison table: {comparison_path}")
    print(f"Wrote final report: {report_path}")
    print(f"Loaded matrix files: {len(source_files)}")
    print(f"Unique rows: {len(rows)}")


if __name__ == "__main__":
    main()
