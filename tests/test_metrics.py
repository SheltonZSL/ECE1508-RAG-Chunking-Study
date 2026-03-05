from __future__ import annotations

from src.eval.qa_metrics import exact_match_score, f1_score, normalize_answer
from src.eval.retrieval_metrics import compute_retrieval_for_query
from src.pipeline.types import RetrievalHit


def test_normalize_answer() -> None:
    assert normalize_answer("The Eiffel Tower!") == "eiffel tower"


def test_qa_metrics() -> None:
    assert exact_match_score("Paris", "paris") == 1.0
    assert f1_score("city of paris", "paris") > 0.0


def test_retrieval_metrics() -> None:
    hits = [
        RetrievalHit(
            query_id="q1",
            chunk_id="c1",
            document_id="d1",
            score=1.0,
            rank=1,
            chunk_text="This chunk contains the answer: Paris.",
        ),
        RetrievalHit(
            query_id="q1",
            chunk_id="c2",
            document_id="d2",
            score=0.5,
            rank=2,
            chunk_text="No signal here.",
        ),
    ]
    metrics = compute_retrieval_for_query(hits, ["Paris"])
    assert metrics["recall_at_k"] == 1.0
    assert metrics["reciprocal_rank"] == 1.0

