from __future__ import annotations

import json
from pathlib import Path

from scripts.build_final_report import build_final_report, collect_rows, write_comparison_csv


def test_final_report_artifacts(tmp_path: Path) -> None:
    summary = tmp_path / "sample_matrix_summary.json"
    summary.write_text(
        json.dumps(
            [
                {
                    "backend": "dense",
                    "strategy": "fixed",
                    "chunk_size": 256,
                    "overlap": 32,
                    "top_k": 5,
                    "recall_at_k": 0.6,
                    "mrr": 0.42,
                    "f1": 0.31,
                    "em": 0.2,
                    "avg_query_latency_ms": 12.4,
                },
                {
                    "backend": "bm25",
                    "strategy": "adaptive",
                    "chunk_size": 128,
                    "overlap": 0,
                    "top_k": 10,
                    "recall_at_k": 0.55,
                    "mrr": 0.33,
                    "f1": 0.25,
                    "em": 0.18,
                    "avg_query_latency_ms": 6.8,
                },
            ]
        ),
        encoding="utf-8",
    )

    rows = collect_rows([summary])
    assert len(rows) == 2
    assert all("composite_score" in row for row in rows)

    out_dir = tmp_path / "analysis"
    csv_path = out_dir / "comparison_table.csv"
    md_path = out_dir / "final_report.md"
    write_comparison_csv(csv_path, rows)
    build_final_report(source_files=[summary], rows=rows, out_path=md_path)

    assert csv_path.exists()
    assert md_path.exists()
    text = md_path.read_text(encoding="utf-8")
    assert "Final Report" in text
    assert "Top 10 Runs" in text
