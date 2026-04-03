# ECE1508 RAG Chunking Study

## 1) Project Purpose
This project provides a reproducible Retrieval-Augmented Generation (RAG) experiment framework to evaluate how chunking strategy impacts retrieval quality and downstream QA quality in a frozen-model setup.

Primary study question:
- when retriever and generator models are fixed, which chunking strategy yields the best retrieval and answer quality tradeoff?

## 2) Project Value
This repository can be used as:
- a research framework for chunking strategy comparison
- a benchmark template for Dense vs BM25 retrieval baselines
- an engineering reference for end-to-end RAG evaluation workflows

It provides:
- standardized configurations for fair comparison
- consistent output protocol for reproducibility
- unified metrics for quality and efficiency analysis

## 3) Project Results
The framework produces structured and reproducible outputs for each run, including:
- QA metrics: `EM`, `F1`
- retrieval metrics: `Recall@k`, `MRR`
- efficiency metrics: retrieval latency and total latency
- detailed prediction/evidence artifacts for error analysis

It also supports matrix studies and reranker ablation analysis to produce comparison tables, summaries, and plots.

Current lite-run highlights from repository artifacts:
- Dense baseline (`fixed`, chunk size `128`, overlap `0`, `top_k=10`) reached `Recall@10 = 0.12` and `MRR = 0.0427`.
- Increasing `top_k` from `3 -> 5 -> 10` improved recall (`0.06 -> 0.09 -> 0.12`) and MRR (`0.0317 -> 0.0387 -> 0.0427`) with latency increase (`0.176 ms -> 0.258 ms -> 0.489 ms`).
- Best reranker quality gain in the lite ablation was at `candidate_k=10`, `alpha=0.20`, with `delta_f1=+0.0194` and `delta_latency=+0.143 ms`.

## 4) Visualization and Interactive Demo
This project includes built-in visualization and interactive inspection:
- dashboard overview: `/dashboard/`
- live QA console: `/dashboard/demo.html`
- plotting/report scripts: `scripts/plot_results.py`, `scripts/plot_sensitivity.py`, `scripts/build_final_report.py`

### Visualization Preview
#### Retrieval Trend (Dense)
![Recall vs Top-k (Dense)](docs/assets/recall_vs_topk_dense.png)

#### Reranker Ablation
![Reranker Ablation Scatter](docs/assets/reranker_ablation_scatter.png)

### What the charts indicate
- **Recall vs Top-k (Dense):** expanding top-k improves coverage, but latency also grows, showing a clear retrieval quality vs efficiency tradeoff.
- **Reranker ablation scatter:** reranker settings can improve `F1` at moderate latency cost, but gains are configuration-sensitive and may not improve every retrieval metric simultaneously.

## 5) Why This Fits Applied Deep Learning
This project aligns with an Applied Deep Learning course because it applies pretrained deep models in a real evaluation workflow and analyzes practical design tradeoffs:
- uses foundation models directly in an application pipeline (`e5-base-v2`, `flan-t5`)
- studies model behavior under system-level design choices (chunking, top-k, reranking) rather than training-only metrics
- evaluates with deployment-relevant criteria: answer quality, retrieval ranking quality, and latency
- includes reproducibility, ablation, and visualization, which are core expectations for applied ML engineering

The focus is on **applied model utilization and decision-making**, which is central to Applied Deep Learning projects.

## 6) Implemented Scope
- Dense retrieval: `intfloat/e5-base-v2` + FAISS (`IndexFlatIP`)
- Sparse baseline: BM25 (`rank-bm25`)
- Generator: `google/flan-t5-base` (fallback: `google/flan-t5-small`)
- Chunking strategies: `fixed`, `structure`, `adaptive`
- Metrics: `EM`, `F1`, `Recall@k`, `MRR`, latency
- Frontend: dashboard + interactive demo

Out of scope:
- model training, fine-tuning, distillation
- production deployment and distributed serving

## 7) Repository Structure
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

## 8) Environment Setup
Requirements:
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

## 9) Quick Start
### A) Portable Interactive Mode (fastest)
Runs directly on bundled demo corpus. No dataset download and no index prebuild required.

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

## 10) Configuration Files
- `configs/portable_interactive.yaml`: lightweight interactive mode with bundled local demo corpus
- `configs/baseline_lite.yaml`: recommended reproducibility baseline
- `configs/baseline_lite_bm25_only.yaml`: BM25-only matrix on lite corpus
- `configs/baseline_lite_reranker.yaml`: lite matrix with reranker enabled
- `configs/reranker_ablation_lite.yaml`: reranker ablation grid (`candidate_k`, `alpha`)
- `configs/baseline_dense.yaml`: heavier dense setting for extended runs
- `configs/baseline_bm25.yaml`: BM25 baseline on full setting

## 11) Main Commands
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

## 12) Output Contract
Each run writes:
- `results/{exp_name}/metrics.json`
- `results/{exp_name}/predictions.jsonl`
- `results/{exp_name}/retrieval_hits.jsonl`
- `results/{exp_name}/error_analysis.md`
- `results/{exp_name}/run_manifest.json`

Matrix suite runs also write:
- `results/{base_exp_name}_run_manifest.json`

Index/chunk artifacts are stored under:
- `data/indexes/{index_name}/...`

## 13) Reproducibility Checklist (Recommended Final Run)
```bash
python scripts/prepare_data.py --config configs/baseline_lite.yaml
python scripts/build_index.py --config configs/baseline_lite.yaml
python scripts/run_retrieval_eval.py --config configs/baseline_lite.yaml
python scripts/run_qa_eval.py --config configs/baseline_lite.yaml
python scripts/run_experiments.py --config configs/reranker_ablation_lite.yaml
python scripts/analyze_reranker_ablation.py --matrix-summary results/reranker_ablation_lite_matrix_summary.json
```

## 14) What Is Tracked in Git
Included in clone:
- source code (`src/`, `scripts/`, `dashboard/`, `tests/`)
- configs (`configs/`)
- demo corpus (`data/demo/corpus.jsonl`, `data/demo/queries.jsonl`)

Generated locally (not tracked):
- `data/processed/*`
- `data/indexes/*`
- `results/*` (except `results/.gitkeep`)

## 15) Troubleshooting
- `ModuleNotFoundError: No module named 'yaml'`
  - run `pip install -r requirements.txt`

- Hugging Face downloads are slow or rate-limited
  - run `hf auth login`
  - or set `HF_TOKEN`

- Cache fills `C:` drive
  - move Hugging Face cache to `D:`:
  ```powershell
  [Environment]::SetEnvironmentVariable("HF_HOME", "D:\\hf_home", "User")
  [Environment]::SetEnvironmentVariable("TRANSFORMERS_CACHE", $null, "User")
  ```

- `Repo id ... ''` during QA eval
  - ensure `generator.model_name` is set in config

- `pytest` temp directory permission issue (`WinError 5`)
  - use a writable temp directory or set a writable `--basetemp`
