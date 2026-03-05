from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.eval.reporting import build_error_analysis, save_eval_outputs
from src.pipeline.workflows import (
    build_chunks,
    build_or_load_retriever,
    evaluate_retrieval,
    load_prepared_documents,
    load_prepared_queries,
)
from src.utils.seed import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Run retrieval-only evaluation.")
    parser.add_argument("--config", type=str, default="configs/baseline_dense.yaml")
    parser.add_argument("--force-rebuild", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(config.run.seed)

    documents = load_prepared_documents(config)
    queries = load_prepared_queries(config)
    index_root = Path(config.retriever.index_dir) / config.run.experiment_name
    index_root.mkdir(parents=True, exist_ok=True)
    chunks = build_chunks(config, documents, save_dir=index_root)

    retriever = build_or_load_retriever(config, chunks, force_rebuild=args.force_rebuild)
    hits, metrics = evaluate_retrieval(queries=queries, retriever=retriever, top_k=config.retrieval.top_k)

    results_dir = Path(config.run.results_dir) / config.run.experiment_name
    retrieval_rows = [asdict(hit) for hit in hits]
    save_eval_outputs(
        out_dir=results_dir,
        metrics={"task": "retrieval_eval", **metrics},
        predictions=[],
        retrieval_hits=retrieval_rows,
        error_analysis=build_error_analysis([]),
    )
    print(f"Saved retrieval evaluation outputs to: {results_dir}")


if __name__ == "__main__":
    main()
