# Project Structure Guide (Plain-English)

This file explains where things are and what to touch first.

## Core flow

The runtime path is:

`prepare_data -> build_index -> retrieve -> generate -> evaluate -> visualize`

## Folder-by-folder

- `configs/`
  - YAML experiment configs.
  - Keep exactly 7 top-level sections:
    - `dataset`, `retriever`, `generator`, `chunking`, `retrieval`, `eval`, `run`

- `scripts/`
  - CLI entrypoints. Start here when running the project.
  - Key scripts:
    - `prepare_data.py`
    - `build_index.py`
    - `run_retrieval_eval.py`
    - `run_qa_eval.py`
    - `run_experiments.py`
    - `serve_dashboard.py`

- `src/`
  - Actual implementation code.
  - `src/pipeline/workflows.py` is the main orchestration layer.

- `dashboard/`
  - Frontend showcase and interactive QA demo.
  - `index.html` = experiment dashboard.
  - `demo.html` = live query page.

- `data/` (git-ignored)
  - `processed/` contains prepared corpus/query JSONL.
  - `indexes/` contains chunk files and retrieval indexes.

- `results/` (git-ignored)
  - Per-run outputs and analysis artifacts.

- `tests/`
  - Unit tests and smoke tests for chunking, FAISS, metrics.

## What to edit for common tasks

- Change experiment scale:
  - edit `configs/*.yaml`
- Add chunking logic:
  - edit `src/chunking/*`
- Change retrieval backend behavior:
  - edit `src/retrieval/*`
- Change evaluation metrics:
  - edit `src/eval/*`
- Improve demo UI:
  - edit `dashboard/*`

## Disk usage notes

- Dense matrix experiments can produce many index files.
- Current matrix runner reuses indexes across `top_k` values to reduce duplication.
- If disk is tight, use `configs/baseline_lite.yaml` and limit matrix runs:
  - `python scripts/run_experiments.py --config configs/baseline_lite.yaml --limit 6 --skip-qa`

