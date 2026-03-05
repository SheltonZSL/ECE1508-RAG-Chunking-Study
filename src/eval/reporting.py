from __future__ import annotations

from pathlib import Path
from typing import Any

from src.utils.io import write_json, write_jsonl


def save_eval_outputs(
    *,
    out_dir: str | Path,
    metrics: dict[str, Any],
    predictions: list[dict[str, Any]],
    retrieval_hits: list[dict[str, Any]],
    error_analysis: str,
) -> None:
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "metrics.json", metrics)
    write_jsonl(output / "predictions.jsonl", predictions)
    write_jsonl(output / "retrieval_hits.jsonl", retrieval_hits)
    (output / "error_analysis.md").write_text(error_analysis, encoding="utf-8")


def build_error_analysis(predictions: list[dict[str, Any]], limit: int = 20) -> str:
    wrong = [row for row in predictions if row.get("em", 0.0) < 1.0]
    header = [
        "# Error Analysis",
        "",
        f"Total predictions: {len(predictions)}",
        f"Exact-match failures: {len(wrong)}",
        "",
    ]
    if not wrong:
        header.append("No EM errors found.")
        return "\n".join(header)

    header.append("## Representative Errors")
    header.append("")

    lines: list[str] = list(header)
    for idx, row in enumerate(wrong[:limit], start=1):
        lines.append(f"### {idx}. Query {row['query_id']}")
        lines.append(f"- Question: {row['question']}")
        lines.append(f"- Prediction: {row['prediction']}")
        lines.append(f"- Gold answers: {row['gold_answers']}")
        lines.append(f"- Retrieved chunk ids: {row['retrieved_chunk_ids']}")
        lines.append("")
    return "\n".join(lines)

