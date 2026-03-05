from __future__ import annotations

from src.eval.qa_metrics import normalize_answer
from src.pipeline.types import RetrievalHit


def _contains_answer(text: str, answers: list[str]) -> bool:
    normalized_text = normalize_answer(text)
    for answer in answers:
        if not answer:
            continue
        if normalize_answer(answer) in normalized_text:
            return True
    return False


def compute_retrieval_for_query(hits: list[RetrievalHit], answers: list[str]) -> dict[str, float]:
    if not hits:
        return {"recall_at_k": 0.0, "reciprocal_rank": 0.0}

    first_relevant_rank = 0
    for hit in hits:
        if _contains_answer(hit.chunk_text, answers):
            first_relevant_rank = hit.rank
            break

    recall_at_k = 1.0 if first_relevant_rank > 0 else 0.0
    rr = 1.0 / first_relevant_rank if first_relevant_rank > 0 else 0.0
    return {"recall_at_k": recall_at_k, "reciprocal_rank": rr}


def aggregate_retrieval_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {"recall_at_k": 0.0, "mrr": 0.0}
    recall = sum(row["recall_at_k"] for row in rows) / len(rows)
    mrr = sum(row["reciprocal_rank"] for row in rows) / len(rows)
    return {"recall_at_k": recall, "mrr": mrr}

