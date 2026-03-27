from __future__ import annotations

import pytest

from src.config.types import (
    ChunkingConfig,
    DatasetConfig,
    EvalConfig,
    GeneratorConfig,
    PipelineConfig,
    RetrievalConfig,
    RetrieverConfig,
    RunConfig,
)
from src.pipeline.types import Query, RetrievalHit
from src.retrieval.reranker import rerank_hits_by_overlap


def _build_pipeline(retrieval_cfg: RetrievalConfig) -> PipelineConfig:
    return PipelineConfig(
        dataset=DatasetConfig(),
        retriever=RetrieverConfig(),
        generator=GeneratorConfig(),
        chunking=ChunkingConfig(),
        retrieval=retrieval_cfg,
        eval=EvalConfig(),
        run=RunConfig(),
    )


def test_overlap_reranker_changes_rank_and_truncates() -> None:
    query = Query(query_id="q1", question="Who discovered penicillin?", answers=["Alexander Fleming"])
    hits = [
        RetrievalHit(
            query_id="q1",
            chunk_id="c1",
            document_id="d1",
            score=12.0,
            rank=1,
            chunk_text="Tokyo is the capital city of Japan.",
        ),
        RetrievalHit(
            query_id="q1",
            chunk_id="c2",
            document_id="d2",
            score=10.0,
            rank=2,
            chunk_text="Penicillin was discovered by Alexander Fleming in 1928.",
        ),
        RetrievalHit(
            query_id="q1",
            chunk_id="c3",
            document_id="d3",
            score=8.0,
            rank=3,
            chunk_text="Python is a popular programming language.",
        ),
    ]

    reranked = rerank_hits_by_overlap(query=query, hits=hits, top_k=2, alpha=0.25)

    assert len(reranked) == 2
    assert reranked[0].chunk_id == "c2"
    assert reranked[0].rank == 1
    assert reranked[1].rank == 2


def test_pipeline_validate_rejects_invalid_reranker_type() -> None:
    cfg = _build_pipeline(
        RetrievalConfig(
            top_k=5,
            reranker_enabled=True,
            reranker_type="unsupported",
            reranker_candidate_k=20,
            reranker_alpha=0.5,
        )
    )
    with pytest.raises(ValueError):
        cfg.validate()
