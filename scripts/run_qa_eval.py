from __future__ import annotations

import argparse
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
    load_prepared_documents,
    load_prepared_queries,
)
from src.utils.seed import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Run end-to-end QA evaluation.")
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

    predictions, hits, metrics = evaluate_qa(
        config=config,
        queries=queries,
        retriever=retriever,
        top_k=config.retrieval.top_k,
    )

    prediction_rows = []
    for pred in predictions:
        em, f1 = best_qa_scores(pred.prediction, pred.gold_answers)
        row = asdict(pred)
        row["em"] = em
        row["f1"] = f1
        prediction_rows.append(row)

    results_dir = Path(config.run.results_dir) / config.run.experiment_name
    save_eval_outputs(
        out_dir=results_dir,
        metrics={"task": "qa_eval", **metrics},
        predictions=prediction_rows,
        retrieval_hits=[asdict(hit) for hit in hits],
        error_analysis=build_error_analysis(prediction_rows),
    )
    print(f"Saved QA evaluation outputs to: {results_dir}")


if __name__ == "__main__":
    main()
