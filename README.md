# ECE1508 RAG Chunking Study

A reproducible RAG experiment framework for one core question:

**How do chunking choices change retrieval quality and final QA quality in a frozen-model pipeline?**

## 1) What this project includes
- Dense retrieval: `intfloat/e5-base-v2` + FAISS (`IndexFlatIP`)
- Sparse retrieval baseline: BM25 (`rank-bm25`)
- Generator: `google/flan-t5-base` (fallback to `google/flan-t5-small`)
- Chunking strategies: `fixed`, `structure`, `adaptive`
- Metrics: `EM`, `F1`, `Recall@k`, `MRR`, latency
- Frontend demo dashboard + live QA page

No training or fine-tuning is used.

## 2) Quick start (recommended)
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Environment requirements:
- Python `3.10+` (recommended: `3.10` or `3.11`)
- OS: Windows/macOS/Linux
- For instant interactive mode: CPU is enough
- For full dense + generation experiments: GPU is recommended but optional

### Step A (Instant Interactive, zero data prep)
```bash
python scripts/serve_dashboard.py --config configs/portable_interactive.yaml --open
```

This mode uses bundled demo data (`data/demo`) + BM25 retrieval.
No dataset download, no index prebuild required.

### Step B (Full experiment mode): Prepare data (lite mode)
```bash
python scripts/prepare_data.py --config configs/baseline_lite.yaml
```

### Step C (Full experiment mode): Build index
```bash
python scripts/build_index.py --config configs/baseline_lite.yaml
```

### Step D (Full experiment mode): Run baseline QA eval
```bash
python scripts/run_qa_eval.py --config configs/baseline_lite.yaml
```

### Step E: Open frontend
```bash
python scripts/serve_dashboard.py --config configs/baseline_lite.yaml --open
```

Open:
- `http://127.0.0.1:8000/dashboard/`
- `http://127.0.0.1:8000/dashboard/demo.html`

## 3) Config choices
- `configs/baseline_lite.yaml`:
  - Smaller, disk-friendly corpus setup
  - Best for demos and quick iteration
- `configs/baseline_dense.yaml`:
  - Heavier setup (`wiki_dpr`)
  - Better for full-size study runs
- `configs/baseline_bm25.yaml`:
  - BM25 baseline config
- `configs/baseline_lite_bm25_only.yaml`:
  - BM25 matrix on lite corpus
- `configs/portable_interactive.yaml`:
  - Clone-and-run interactive demo mode
  - Uses bundled local demo corpus, no prep step

## 4) Main commands
- Prepare data:
```bash
python scripts/prepare_data.py --config <config_path>
```
- Build index:
```bash
python scripts/build_index.py --config <config_path> [--force-rebuild]
```
- Retrieval-only eval:
```bash
python scripts/run_retrieval_eval.py --config <config_path> [--force-rebuild]
```
- End-to-end QA eval:
```bash
python scripts/run_qa_eval.py --config <config_path> [--force-rebuild]
```
- Matrix experiments:
```bash
python scripts/run_experiments.py --config <config_path> [--limit N] [--skip-qa] [--force-rebuild]
```

## 5) Output contract
Each run writes:
- `results/{exp_name}/metrics.json`
- `results/{exp_name}/predictions.jsonl`
- `results/{exp_name}/retrieval_hits.jsonl`
- `results/{exp_name}/error_analysis.md`

For cleaner presentation, you can reorganize outputs into:
- `results/runs/`
- `results/summaries/`
- `results/analysis/{dense,bm25}/`

Command:
```bash
python scripts/organize_results.py
```

Indexes/chunks are stored under:
- `data/indexes/{index_name}/...`

## 6) Project structure map
```text
ECE1508-RAG-Chunking-Study/
|- configs/                 # experiment YAMLs (7 fixed sections)
|- dashboard/               # frontend dashboard + interactive demo
|- data/
|  |- processed/            # prepared queries/corpus jsonl (ignored by git)
|  |- indexes/              # FAISS/BM25 artifacts + chunks (ignored by git)
|- results/                 # experiment outputs and analysis (ignored by git)
|- scripts/                 # runnable CLI entrypoints
|- src/
|  |- config/               # config dataclasses + loader
|  |- data/                 # NQ + corpus preparation
|  |- chunking/             # fixed / structure / adaptive chunkers
|  |- retrieval/            # dense + bm25 retrievers
|  |- generation/           # HF generator wrapper
|  |- pipeline/             # orchestration + shared types
|  |- eval/                 # QA/retrieval metrics + reporting
|  |- utils/                # io, seed, text helpers
|- tests/                   # unit + smoke tests
|- requirements.txt
|- README.md
```

Detailed structure notes:
- `docs/PROJECT_STRUCTURE.md`

## 7) Common confusion (important)
- Matrix mode can be slow because it runs many combinations.
- Dense index files are large (tens of MB each). This is expected.
- `data/` and `results/` are intentionally not uploaded to GitHub.
- Teammates should run the same scripts locally to regenerate data/index/results.
- If you have old matrix runs, you may have duplicated legacy index folders with `_k*` suffix.
  - Dry run:
  ```bash
  python scripts/cleanup_legacy_indexes.py
  ```
  - Apply deletion:
  ```bash
  python scripts/cleanup_legacy_indexes.py --apply
  ```

## 8) Recent framework improvements
- Dashboard now merges multiple matrix summary files (Dense + BM25 can be shown together).
- Matrix runner now reuses one index for multiple `top_k` settings, reducing duplicated index files and runtime.
- Latency metrics now reflect real retrieval/end-to-end timing instead of metric-computation overhead.

## 9) Post-v1: planned modifications and additions
The current version is submission-ready for an initial release. The following items are planned next:

- Add one-click setup scripts:
  - `scripts/setup_portable.py` for instant interactive mode
  - `scripts/setup_full.py` for full experiment mode
- Add run metadata logging:
  - output `run_manifest.json` with config snapshot, timestamp, git commit, dependency versions, and device info
- Add frontend side-by-side comparison mode:
  - compare Config A vs Config B answer, evidence hits, and latency in one screen
- Add final report auto-builder:
  - `scripts/build_final_report.py` to export `results/analysis/final_report.md` and `comparison_table.csv`
- Add API smoke tests:
  - verify `/api/defaults`, `/api/examples`, `/api/ask` for basic reliability
- Add statistical confidence reporting:
  - bootstrap confidence intervals for EM/F1/Recall@k/MRR
- Add optional reranker switch:
  - keep default off, support controlled reranker vs no-reranker comparison
- Add dashboard export feature:
  - export current filtered runs as CSV/JSON for presentation and report use
