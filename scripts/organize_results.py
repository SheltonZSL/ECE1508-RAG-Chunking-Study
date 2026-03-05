from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def _move_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        # Merge directory contents if both are directories.
        if src.is_dir() and dst.is_dir():
            for item in src.iterdir():
                _move_if_exists(item, dst / item.name)
            src.rmdir()
            return
        dst.unlink()
    shutil.move(str(src), str(dst))


def _write_results_readme(results_root: Path) -> None:
    content = """# Results Folder Guide

This folder is organized for presentation and quick navigation:

- `runs/`: per-experiment output bundles (`metrics.json`, `predictions.jsonl`, etc.)
- `summaries/`: matrix summary JSON files
- `analysis/`: plots/tables/notes for comparison analysis
  - `analysis/dense/`: dense-focused analysis artifacts
  - `analysis/bm25/`: bm25-focused analysis artifacts

Typical files:
- `runs/<exp_name>/metrics.json`
- `runs/<exp_name>/predictions.jsonl`
- `runs/<exp_name>/retrieval_hits.jsonl`
- `runs/<exp_name>/error_analysis.md`
- `summaries/*_matrix_summary.json`
"""
    (results_root / "README.md").write_text(content, encoding="utf-8")


def organize_results(results_root: Path) -> None:
    if not results_root.exists():
        results_root.mkdir(parents=True, exist_ok=True)

    runs_dir = results_root / "runs"
    summaries_dir = results_root / "summaries"
    analysis_root = results_root / "analysis"
    dense_analysis_dir = analysis_root / "dense"
    bm25_analysis_dir = analysis_root / "bm25"

    runs_dir.mkdir(parents=True, exist_ok=True)
    summaries_dir.mkdir(parents=True, exist_ok=True)
    dense_analysis_dir.mkdir(parents=True, exist_ok=True)
    bm25_analysis_dir.mkdir(parents=True, exist_ok=True)

    # Known run folder currently used in this project state.
    _move_if_exists(results_root / "baseline_lite", runs_dir / "baseline_lite")

    # Move matrix summary files under summaries/.
    _move_if_exists(
        results_root / "baseline_lite_matrix_summary.json",
        summaries_dir / "baseline_lite_matrix_summary.json",
    )
    _move_if_exists(
        results_root / "baseline_lite_bm25_matrix_summary.json",
        summaries_dir / "baseline_lite_bm25_matrix_summary.json",
    )

    # Normalize analysis layout.
    legacy_bm25 = results_root / "analysis_bm25"
    _move_if_exists(legacy_bm25, bm25_analysis_dir)

    # If legacy dense files are directly under analysis/, move them to analysis/dense/.
    if analysis_root.exists():
        for item in list(analysis_root.iterdir()):
            if item.name in {"dense", "bm25"}:
                continue
            _move_if_exists(item, dense_analysis_dir / item.name)

    _write_results_readme(results_root)

    # Keep .gitkeep for empty-case compatibility.
    gitkeep = results_root / ".gitkeep"
    if not gitkeep.exists():
        gitkeep.write_text("", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Organize results into a clean folder structure.")
    parser.add_argument("--results-dir", type=str, default="results")
    args = parser.parse_args()
    organize_results(Path(args.results_dir))
    print(f"Organized results under: {Path(args.results_dir)}")


if __name__ == "__main__":
    main()

