from __future__ import annotations

import argparse
import copy
import itertools
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.eval.qa_metrics import best_qa_scores
from src.eval.reporting import build_error_analysis, save_eval_outputs
from src.pipeline.workflows import (
    build_chunks,
    build_or_load_retriever,
    evaluate_qa,
    evaluate_retrieval,
    load_prepared_documents,
    load_prepared_queries,
)
from src.utils.seed import seed_everything


def _iter_matrix(config):
    matrix = config.run.matrix
    for backend, strategy, chunk_size, overlap, top_k in itertools.product(
        matrix.backends,
        matrix.strategies,
        matrix.chunk_sizes,
        matrix.overlaps,
        matrix.top_ks,
    ):
        if overlap >= chunk_size:
            continue
        yield backend, strategy, chunk_size, overlap, top_k


def _run_name(base_name: str, backend: str, strategy: str, chunk_size: int, overlap: int, top_k: int) -> str:
    return f"{base_name}_{backend}_{strategy}_c{chunk_size}_o{overlap}_k{top_k}"


def _index_name(base_name: str, backend: str, strategy: str, chunk_size: int, overlap: int) -> str:
    return f"{base_name}_{backend}_{strategy}_c{chunk_size}_o{overlap}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run experiment matrix for chunking study.")
    parser.add_argument("--config", type=str, default="configs/baseline_dense.yaml")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of matrix runs")
    parser.add_argument("--skip-qa", action="store_true", help="Only run retrieval metrics")
    parser.add_argument("--force-rebuild", action="store_true")
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    seed_everything(base_cfg.run.seed)
    documents = load_prepared_documents(base_cfg)
    queries = load_prepared_queries(base_cfg)

    rows: list[dict[str, float | str]] = []
    retriever_cache: dict[tuple[str, str, int, int], object] = {}
    count = 0

    for backend, strategy, chunk_size, overlap, top_k in _iter_matrix(base_cfg):
        if args.limit is not None and count >= args.limit:
            break

        cfg = copy.deepcopy(base_cfg)
        cfg.retriever.backend = backend
        cfg.chunking.strategy = strategy
        cfg.chunking.chunk_size = chunk_size
        cfg.chunking.overlap = overlap
        cfg.retrieval.top_k = top_k
        cfg.run.experiment_name = _run_name(
            base_name=base_cfg.run.experiment_name,
            backend=backend,
            strategy=strategy,
            chunk_size=chunk_size,
            overlap=overlap,
            top_k=top_k,
        )
        cfg.validate()

        print(f"Running {cfg.run.experiment_name}")
        index_key = (backend, strategy, chunk_size, overlap)
        if index_key not in retriever_cache:
            index_cfg = copy.deepcopy(cfg)
            index_cfg.run.experiment_name = _index_name(
                base_name=base_cfg.run.experiment_name,
                backend=backend,
                strategy=strategy,
                chunk_size=chunk_size,
                overlap=overlap,
            )
            index_cfg.validate()

            print(f"  Building/loading index once for key={index_key}")
            index_root = Path(index_cfg.retriever.index_dir) / index_cfg.run.experiment_name
            index_root.mkdir(parents=True, exist_ok=True)
            chunks = build_chunks(index_cfg, documents, save_dir=index_root)
            retriever_cache[index_key] = build_or_load_retriever(
                index_cfg, chunks, force_rebuild=args.force_rebuild
            )

        retriever = retriever_cache[index_key]

        retrieval_hits, retrieval_metrics = evaluate_retrieval(
            queries=queries, retriever=retriever, top_k=cfg.retrieval.top_k
        )

        metrics: dict[str, float | str] = {
            "backend": backend,
            "strategy": strategy,
            "chunk_size": float(chunk_size),
            "overlap": float(overlap),
            "top_k": float(top_k),
            "recall_at_k": retrieval_metrics["recall_at_k"],
            "mrr": retrieval_metrics["mrr"],
            "avg_query_latency_ms": retrieval_metrics["avg_query_latency_ms"],
        }

        prediction_rows: list[dict[str, object]] = []
        if not args.skip_qa:
            predictions, qa_hits, qa_metrics = evaluate_qa(
                config=cfg,
                queries=queries,
                retriever=retriever,
                top_k=cfg.retrieval.top_k,
            )
            for pred in predictions:
                em, f1 = best_qa_scores(pred.prediction, pred.gold_answers)
                row = asdict(pred)
                row["em"] = em
                row["f1"] = f1
                prediction_rows.append(row)

            metrics["em"] = qa_metrics["em"]
            metrics["f1"] = qa_metrics["f1"]
            metrics["avg_context_char_len"] = qa_metrics["avg_context_char_len"]
            retrieval_hits = qa_hits

        results_dir = Path(cfg.run.results_dir) / cfg.run.experiment_name
        save_eval_outputs(
            out_dir=results_dir,
            metrics={"task": "matrix_run", **metrics},
            predictions=prediction_rows,
            retrieval_hits=[asdict(hit) for hit in retrieval_hits],
            error_analysis=build_error_analysis(prediction_rows),
        )

        rows.append(metrics)
        count += 1

    summary_path = Path(base_cfg.run.results_dir) / f"{base_cfg.run.experiment_name}_matrix_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved matrix summary to: {summary_path}")
    print(f"Completed runs: {count}")


if __name__ == "__main__":
    main()
