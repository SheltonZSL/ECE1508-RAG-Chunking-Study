from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.chunking import create_chunker
from src.config.types import PipelineConfig
from src.eval.qa_metrics import best_qa_scores, compute_qa_aggregate
from src.eval.retrieval_metrics import aggregate_retrieval_metrics, compute_retrieval_for_query
from src.pipeline.types import Chunk, Document, QAPrediction, Query, RetrievalHit
from src.utils.io import read_jsonl, write_jsonl


def load_prepared_documents(config: PipelineConfig) -> list[Document]:
    path = Path(config.dataset.data_dir) / "corpus.jsonl"
    rows = read_jsonl(path)
    return [
        Document(
            document_id=row["document_id"],
            title=row.get("title"),
            text=row["text"],
            metadata=row.get("metadata", {}),
        )
        for row in rows
    ]


def load_prepared_queries(config: PipelineConfig) -> list[Query]:
    path = Path(config.dataset.data_dir) / "queries.jsonl"
    rows = read_jsonl(path)
    if config.eval.max_eval_queries is not None:
        rows = rows[: config.eval.max_eval_queries]
    return [
        Query(
            query_id=row["query_id"],
            question=row["question"],
            answers=list(row.get("answers", [])),
            metadata=row.get("metadata", {}),
        )
        for row in rows
    ]


def build_chunks(config: PipelineConfig, documents: list[Document], save_dir: str | Path) -> list[Chunk]:
    chunker = create_chunker(config.chunking)
    chunks = chunker.chunk_documents(documents)
    out_path = Path(save_dir) / "chunks.jsonl"
    write_jsonl(out_path, [asdict(chunk) for chunk in chunks])
    return chunks


def _index_root(config: PipelineConfig) -> Path:
    return Path(config.retriever.index_dir) / config.run.experiment_name


def _build_retriever(config: PipelineConfig):
    backend = config.retriever.backend.lower().strip()
    if backend == "dense":
        from src.retrieval.dense import DenseRetriever

        return DenseRetriever(config.retriever)
    if backend == "bm25":
        from src.retrieval.bm25 import BM25Retriever

        return BM25Retriever(config.retriever)
    raise ValueError(f"Unsupported retriever backend: {config.retriever.backend}")


def build_or_load_retriever(
    config: PipelineConfig,
    chunks: list[Chunk],
    force_rebuild: bool = False,
):
    root = _index_root(config)
    backend = config.retriever.backend.lower().strip()
    retriever = _build_retriever(config)

    dense_ready = (root / "dense.index.faiss").exists() and (root / "dense.meta.json").exists()
    bm25_ready = (root / "bm25.model.pkl").exists() and (root / "bm25.meta.json").exists()
    can_load = dense_ready if backend == "dense" else bm25_ready

    if can_load and not force_rebuild:
        retriever.load(root)
    else:
        retriever.fit(chunks)
        retriever.save(root)
    return retriever


def evaluate_retrieval(
    *,
    queries: list[Query],
    retriever,
    top_k: int,
) -> tuple[list[RetrievalHit], dict[str, float]]:
    retrieval_start = time.perf_counter()
    all_hits = retriever.retrieve(queries, top_k=top_k)
    retrieval_total_ms = (time.perf_counter() - retrieval_start) * 1000.0
    flattened: list[RetrievalHit] = []
    rows: list[dict[str, float]] = []

    for query, hits in zip(queries, all_hits):
        row_metrics = compute_retrieval_for_query(hits, query.answers)
        rows.append(row_metrics)
        flattened.extend(hits)

    aggregate = aggregate_retrieval_metrics(rows)
    aggregate["avg_query_latency_ms"] = retrieval_total_ms / len(queries) if queries else 0.0
    aggregate["total_retrieval_latency_ms"] = retrieval_total_ms
    aggregate["num_queries"] = float(len(queries))
    return flattened, aggregate


def evaluate_qa(
    *,
    config: PipelineConfig,
    queries: list[Query],
    retriever,
    top_k: int,
) -> tuple[list[QAPrediction], list[RetrievalHit], dict[str, float]]:
    from src.generation.hf_generator import HFGenerator

    generator = HFGenerator(config.generator)
    retrieval_start = time.perf_counter()
    all_hits = retriever.retrieve(queries, top_k=top_k)
    retrieval_total_ms = (time.perf_counter() - retrieval_start) * 1000.0

    qa_rows: list[dict[str, float]] = []
    retrieval_rows: list[dict[str, float]] = []
    predictions: list[QAPrediction] = []
    flattened_hits: list[RetrievalHit] = []
    generation_latencies: list[float] = []
    context_lengths: list[int] = []

    for query, hits in zip(queries, all_hits):
        contexts = [hit.chunk_text for hit in hits]
        start = time.perf_counter()
        prediction_text = generator.generate(query.question, contexts)
        elapsed = (time.perf_counter() - start) * 1000.0
        generation_latencies.append(elapsed)
        context_len = sum(len(c) for c in contexts)
        context_lengths.append(context_len)

        em, f1 = best_qa_scores(prediction_text, query.answers)
        qa_rows.append({"em": em, "f1": f1})
        retrieval_row = compute_retrieval_for_query(hits, query.answers)
        retrieval_rows.append(retrieval_row)
        flattened_hits.extend(hits)

        predictions.append(
            QAPrediction(
                query_id=query.query_id,
                question=query.question,
                prediction=prediction_text,
                gold_answers=query.answers,
                retrieved_chunk_ids=[hit.chunk_id for hit in hits],
                retrieved_texts=contexts,
                latency_ms=elapsed,
                context_char_len=context_len,
            )
        )

    qa_agg = compute_qa_aggregate(qa_rows)
    retrieval_agg = aggregate_retrieval_metrics(retrieval_rows)
    avg_retrieval_latency_ms = retrieval_total_ms / len(queries) if queries else 0.0
    avg_generation_latency_ms = (
        sum(generation_latencies) / len(generation_latencies) if generation_latencies else 0.0
    )
    metrics = {
        "em": qa_agg["em"],
        "f1": qa_agg["f1"],
        "recall_at_k": retrieval_agg["recall_at_k"],
        "mrr": retrieval_agg["mrr"],
        "avg_query_latency_ms": avg_retrieval_latency_ms + avg_generation_latency_ms,
        "avg_retrieval_latency_ms": avg_retrieval_latency_ms,
        "avg_generation_latency_ms": avg_generation_latency_ms,
        "total_retrieval_latency_ms": retrieval_total_ms,
        "avg_context_char_len": sum(context_lengths) / len(context_lengths) if context_lengths else 0.0,
        "num_queries": float(len(queries)),
    }
    return predictions, flattened_hits, metrics


def to_dict_rows(items: list[Any]) -> list[dict[str, Any]]:
    return [asdict(item) for item in items]
