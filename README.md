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
python scripts/setup_portable.py --open
```

This mode uses bundled demo data (`data/demo`) + BM25 retrieval.
No dataset download, no index prebuild required.
In this mode, `run_qa_eval.py` also works without loading an HF generator model (fallback answer from retrieved evidence).

### Step B (Full experiment mode): Prepare data (lite mode)
```bash
python scripts/setup_full.py
```

### Step C (Full experiment mode): Run baseline QA eval (optional)
```bash
python scripts/setup_full.py --run-qa
```

### Step D: Open frontend in full mode
```bash
python scripts/setup_full.py --serve --open
```

Open:
- `http://127.0.0.1:8000/dashboard/`
- `http://127.0.0.1:8000/dashboard/demo.html`

## 3) Config choices
- `configs/baseline_lite.yaml`:
  - Smaller, disk-friendly corpus setup
  - Best for demos and quick iteration
- `configs/baseline_lite_reranker.yaml`:
  - Lite setup with reranker enabled for controlled reranker-vs-baseline comparison
- `configs/reranker_ablation_lite.yaml`:
  - Focused reranker ablation grid (`candidate_k` and `alpha`) on lite setup
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
- One-click portable setup + serve:
```bash
python scripts/setup_portable.py [--install-deps] [--open]
```
- One-click full setup (+ optional serve/eval):
```bash
python scripts/setup_full.py [--install-deps] [--run-qa] [--serve --open]
```
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
- Reranker extension comparison (same matrix, reranker on):
```bash
python scripts/run_experiments.py --config configs/baseline_lite_reranker.yaml
```
- Reranker ablation experiment:
```bash
python scripts/run_experiments.py --config configs/reranker_ablation_lite.yaml
python scripts/analyze_reranker_ablation.py --matrix-summary results/reranker_ablation_lite_matrix_summary.json
```
- Build final report artifacts from matrix summaries:
```bash
python scripts/build_final_report.py [--results-dir results] [--out-dir results/analysis]
```
- Generate deeper sensitivity-analysis plots:
```bash
python scripts/plot_sensitivity.py --matrix-summary <summary_json_or_list...> [--out-dir results/analysis/sensitivity]
```
Example (baseline + reranker summary together):
```bash
python scripts/plot_sensitivity.py --matrix-summary results/baseline_lite_matrix_summary.json results/baseline_lite_reranker_matrix_summary.json
```
- API smoke tests:
```bash
pytest -q tests/test_api_smoke.py
```

## 5) Output contract
Each run writes:
- `results/{exp_name}/metrics.json`
- `results/{exp_name}/predictions.jsonl`
- `results/{exp_name}/retrieval_hits.jsonl`
- `results/{exp_name}/error_analysis.md`
- `results/{exp_name}/run_manifest.json`

For matrix suite runs, an extra suite-level manifest is written:
- `results/{base_exp_name}_run_manifest.json`

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

### Optional: remove `TRANSFORMERS_CACHE` deprecation warning
The project now auto-maps deprecated `TRANSFORMERS_CACHE` to `HF_HOME` at runtime.
If you want to clean your Windows user environment permanently, run:
```powershell
[Environment]::SetEnvironmentVariable("HF_HOME", "$env:USERPROFILE\\.cache\\huggingface", "User")
[Environment]::SetEnvironmentVariable("TRANSFORMERS_CACHE", $null, "User")
```
Then restart your terminal.

## 8) Recent framework improvements
- Dashboard now merges multiple matrix summary files (Dense + BM25 can be shown together).
- Matrix runner now reuses one index for multiple `top_k` settings, reducing duplicated index files and runtime.
- Latency metrics now reflect real retrieval/end-to-end timing instead of metric-computation overhead.
- Every eval run now writes `run_manifest.json` (config snapshot, script args, env, git info, timing).
- Interactive demo now supports A/B comparison (Run A vs Run B side-by-side with latency and evidence).
- Added API smoke test coverage for `/api/health`, `/api/defaults`, `/api/examples`, `/api/ask`.
- Added bootstrap CI fields for EM/F1/Recall@k/MRR in evaluation outputs.
- Added dashboard export for filtered runs (CSV/JSON buttons in the Experiment Lab panel).
- Added final report auto-builder (`scripts/build_final_report.py`) for:
  - `results/analysis/comparison_table.csv`
  - `results/analysis/final_report.md`
- Added optional reranker extension (default off):
  - config keys: `retrieval.reranker_enabled`, `reranker_type`, `reranker_candidate_k`, `reranker_alpha`
  - supports controlled reranker/no-reranker comparison under the same retrieval backend
- Added deeper sensitivity-analysis visualization script (`scripts/plot_sensitivity.py`) for:
  - quality-latency tradeoff scatter
  - top-k sensitivity curves
  - chunk-size/overlap heatmaps
  - markdown summary (`sensitivity_summary.md`)
- Hardened interactive API:
  - config path confinement to repo root
  - backend/strategy option validation
  - request body size limit
  - no traceback leakage in API error responses
- Replaced BM25 pickle deserialization with JSON-only metadata restore path.

## 9) Post-v1: planned modifications and additions
The current version is submission-ready for an initial release. The following items are planned next:

- Add optional API auth mode for non-local sharing scenarios:
  - token gate for `/api/ask` when serving beyond localhost
