# ECE1508 RAG Chunking Study

## Overview
This repository implements a reproducible Retrieval-Augmented Generation (RAG) experiment framework for the following research question:

**How do chunking design choices affect retrieval quality and downstream QA quality in a frozen-model pipeline?**

The project is designed for controlled comparison, not model training. All models are used in frozen (pretrained) form.

## Implemented Scope
- Dense retrieval: `intfloat/e5-base-v2` + FAISS (`IndexFlatIP`)
- Sparse baseline: BM25 (`rank-bm25`)
- Generation model: `google/flan-t5-base` (fallback: `google/flan-t5-small`)
- Chunking strategies: `fixed`, `structure`, `adaptive`
- Metrics:
  - QA: `EM`, `F1`
  - Retrieval: `Recall@k`, `MRR`
  - Efficiency: retrieval/total latency
- Frontend:
  - dashboard (`/dashboard/`)
  - interactive QA demo (`/dashboard/demo.html`)

Out of scope:
- training, fine-tuning, distillation
- production deployment and distributed serving

## Repository Structure
```text
ECE1508-RAG-Chunking-Study/
|- configs/                 # experiment YAML configs
|- dashboard/               # frontend dashboard and interactive demo
|- data/
|  |- demo/                 # bundled lightweight demo corpus
|  |- processed/            # generated prepared corpus/query files (gitignored)
|  |- indexes/              # generated index and chunk artifacts (gitignored)
|- results/                 # generated run outputs and analysis (gitignored except .gitkeep)
|- scripts/                 # executable experiment/evaluation scripts
|- src/                     # core implementation
|- tests/                   # unit and smoke tests
|- requirements.txt
|- README.md
```

Additional structure notes: `docs/PROJECT_STRUCTURE.md`

## Environment
- Python `3.10+` (recommended `3.10` or `3.11`)
- Windows/macOS/Linux
- CPU is sufficient for portable interactive mode
- GPU is recommended for larger dense/generation runs

Setup:
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Quick Start

### A) Portable Interactive Mode (fastest, no data prep)
Runs directly on bundled demo corpus. No dataset download, no index prebuild.

```bash
python scripts/setup_portable.py --open
```

Open:
- `http://127.0.0.1:8000/dashboard/`
- `http://127.0.0.1:8000/dashboard/demo.html`

### B) Full Experiment Mode (lite baseline recommended)
```bash
python scripts/setup_full.py
```

Optional:
```bash
python scripts/setup_full.py --run-qa
python scripts/setup_full.py --serve --open
```

## Configuration Files
- `configs/portable_interactive.yaml`  
  Lightweight interactive mode using bundled local demo data.

- `configs/baseline_lite.yaml`  
  Recommended reproducibility baseline with moderate disk/runtime cost.

- `configs/baseline_lite_bm25_only.yaml`  
  BM25-only matrix on lite corpus.

- `configs/baseline_lite_reranker.yaml`  
  Lite matrix with reranker enabled for controlled comparison.

- `configs/reranker_ablation_lite.yaml`  
  Reranker ablation grid (`candidate_k`, `alpha`) on lite setup.

- `configs/baseline_dense.yaml`  
  Heavier dense setting (larger corpus), intended for extended runs.

- `configs/baseline_bm25.yaml`  
  BM25 baseline on full setting.

## Main Commands

### Data and Index
```bash
python scripts/prepare_data.py --config <config_path>
python scripts/build_index.py --config <config_path> [--force-rebuild]
```

### Evaluation
```bash
python scripts/run_retrieval_eval.py --config <config_path> [--force-rebuild]
python scripts/run_qa_eval.py --config <config_path> [--force-rebuild]
```

### Matrix Experiments
```bash
python scripts/run_experiments.py --config <config_path> [--limit N] [--skip-qa] [--force-rebuild]
```

Reranker-focused runs:
```bash
python scripts/run_experiments.py --config configs/baseline_lite_reranker.yaml
python scripts/run_experiments.py --config configs/reranker_ablation_lite.yaml
python scripts/analyze_reranker_ablation.py --matrix-summary results/reranker_ablation_lite_matrix_summary.json
```

### Reporting and Visualization
```bash
python scripts/build_final_report.py [--results-dir results] [--out-dir results/analysis]
python scripts/plot_sensitivity.py --matrix-summary <summary_json_or_list...> [--out-dir results/analysis/sensitivity]
python scripts/organize_results.py
```

Example:
```bash
python scripts/plot_sensitivity.py --matrix-summary results/baseline_lite_matrix_summary.json results/baseline_lite_reranker_matrix_summary.json
```

### API Smoke Test
```bash
pytest -q tests/test_api_smoke.py
```

## Output Contract
Each run writes:
- `results/{exp_name}/metrics.json`
- `results/{exp_name}/predictions.jsonl`
- `results/{exp_name}/retrieval_hits.jsonl`
- `results/{exp_name}/error_analysis.md`
- `results/{exp_name}/run_manifest.json`

Matrix suite run also writes:
- `results/{base_exp_name}_run_manifest.json`

Index and chunk artifacts are stored under:
- `data/indexes/{index_name}/...`

## Reproducibility Checklist (Recommended Final Run)
```bash
python scripts/prepare_data.py --config configs/baseline_lite.yaml
python scripts/build_index.py --config configs/baseline_lite.yaml
python scripts/run_retrieval_eval.py --config configs/baseline_lite.yaml
python scripts/run_qa_eval.py --config configs/baseline_lite.yaml
python scripts/run_experiments.py --config configs/reranker_ablation_lite.yaml
python scripts/analyze_reranker_ablation.py --matrix-summary results/reranker_ablation_lite_matrix_summary.json
```

## What Is Tracked in Git
Included in clone:
- source code (`src/`, `scripts/`, `dashboard/`, `tests/`)
- configs (`configs/`)
- demo corpus (`data/demo/corpus.jsonl`, `data/demo/queries.jsonl`)

Generated locally (not tracked):
- `data/processed/*`
- `data/indexes/*`
- `results/*` (except `results/.gitkeep`)

## Troubleshooting
- `ModuleNotFoundError: No module named 'yaml'`
  - Run:
  ```bash
  pip install -r requirements.txt
  ```

- Hugging Face downloads are slow or rate-limited
  - Authenticate once:
  ```bash
  hf auth login
  ```
  - Or set `HF_TOKEN`.

- Cache fills `C:` drive
  - Move cache to `D:` (Windows):
  ```powershell
  [Environment]::SetEnvironmentVariable("HF_HOME", "D:\\hf_home", "User")
  [Environment]::SetEnvironmentVariable("TRANSFORMERS_CACHE", $null, "User")
  ```
  - Restart terminal.

- `Repo id ... ''` during QA eval
  - Ensure `generator.model_name` is set in config.

- `pytest` temp directory permission issue (`WinError 5`)
  - Use a writable temp directory or set a custom writable `--basetemp`.

