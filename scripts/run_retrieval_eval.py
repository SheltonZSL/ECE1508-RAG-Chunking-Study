from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.eval.reporting import (
    build_error_analysis,
    build_run_manifest,
    save_eval_outputs,
    save_run_manifest,
)
from src.pipeline.workflows import (
    build_chunks,
    build_or_load_retriever,
    evaluate_retrieval,
    load_prepared_documents,
    load_prepared_queries,
)
from src.utils.seed import seed_everything


def main() -> None:
    start_time_utc = datetime.now(timezone.utc)

    parser = argparse.ArgumentParser(description="Run retrieval-only evaluation.")
    parser.add_argument("--config", type=str, default="configs/baseline_dense.yaml")
    parser.add_argument("--force-rebuild", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(config.run.seed)

    t0 = time.perf_counter()
    documents = load_prepared_documents(config)
    t_docs = time.perf_counter() - t0
    t1 = time.perf_counter()
    queries = load_prepared_queries(config)
    t_queries = time.perf_counter() - t1
    index_root = Path(config.retriever.index_dir) / config.run.experiment_name
    index_root.mkdir(parents=True, exist_ok=True)
    t2 = time.perf_counter()
    chunks = build_chunks(config, documents, save_dir=index_root)
    t_chunks = time.perf_counter() - t2

    t3 = time.perf_counter()
    retriever = build_or_load_retriever(config, chunks, force_rebuild=args.force_rebuild)
    t_index = time.perf_counter() - t3
    t4 = time.perf_counter()
    hits, metrics = evaluate_retrieval(
        config=config,
        queries=queries,
        retriever=retriever,
        top_k=config.retrieval.top_k,
    )
    t_eval = time.perf_counter() - t4

    results_dir = Path(config.run.results_dir) / config.run.experiment_name
    retrieval_rows = [asdict(hit) for hit in hits]
    metrics_payload = {
        "task": "retrieval_eval",
        "reranker_enabled": config.retrieval.reranker_enabled,
        "reranker_type": config.retrieval.reranker_type if config.retrieval.reranker_enabled else "none",
        "reranker_candidate_k": config.retrieval.reranker_candidate_k,
        "reranker_alpha": config.retrieval.reranker_alpha,
        **metrics,
    }
    save_eval_outputs(
        out_dir=results_dir,
        metrics=metrics_payload,
        predictions=[],
        retrieval_hits=retrieval_rows,
        error_analysis=build_error_analysis([]),
    )
    manifest = build_run_manifest(
        script_name="scripts/run_retrieval_eval.py",
        script_args=vars(args),
        config=config,
        config_path=args.config,
        stage="retrieval_eval",
        start_time_utc=start_time_utc,
        end_time_utc=datetime.now(timezone.utc),
        extra={
            "results_dir": str(results_dir),
            "metrics_snapshot": {
                "num_queries": metrics.get("num_queries"),
                "recall_at_k": metrics.get("recall_at_k"),
                "mrr": metrics.get("mrr"),
                "avg_query_latency_ms": metrics.get("avg_query_latency_ms"),
            },
            "stage_timings_seconds": {
                "load_documents": round(t_docs, 4),
                "load_queries": round(t_queries, 4),
                "build_chunks": round(t_chunks, 4),
                "build_or_load_index": round(t_index, 4),
                "evaluate": round(t_eval, 4),
            },
        },
    )
    save_run_manifest(results_dir, manifest)
    print(f"Saved retrieval evaluation outputs to: {results_dir}")


if __name__ == "__main__":
    main()
