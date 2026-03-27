from __future__ import annotations

import re

from src.pipeline.types import Query, RetrievalHit

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokenize_to_set(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def _normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []
    lo = min(scores)
    hi = max(scores)
    if hi - lo < 1e-12:
        return [0.0 for _ in scores]
    return [(score - lo) / (hi - lo) for score in scores]


def _overlap_score(query_tokens: set[str], passage_text: str) -> float:
    if not query_tokens:
        return 0.0
    passage_tokens = _tokenize_to_set(passage_text)
    if not passage_tokens:
        return 0.0
    common = query_tokens.intersection(passage_tokens)
    return float(len(common)) / float(len(query_tokens))


def rerank_hits_by_overlap(
    *,
    query: Query,
    hits: list[RetrievalHit],
    top_k: int,
    alpha: float,
) -> list[RetrievalHit]:
    if top_k <= 0:
        return []
    if not hits:
        return []

    query_tokens = _tokenize_to_set(query.question)
    base_scores = [float(hit.score) for hit in hits]
    base_scores_norm = _normalize_scores(base_scores)

    rescored: list[tuple[float, int, RetrievalHit]] = []
    for idx, hit in enumerate(hits):
        lexical_score = _overlap_score(query_tokens, hit.chunk_text)
        fused_score = (alpha * base_scores_norm[idx]) + ((1.0 - alpha) * lexical_score)
        rescored.append((fused_score, idx, hit))

    rescored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    reranked_hits: list[RetrievalHit] = []
    for rank, (score, _, hit) in enumerate(rescored[:top_k], start=1):
        reranked_hits.append(
            RetrievalHit(
                query_id=hit.query_id,
                chunk_id=hit.chunk_id,
                document_id=hit.document_id,
                score=float(score),
                rank=rank,
                chunk_text=hit.chunk_text,
            )
        )
    return reranked_hits


def rerank_hits(
    *,
    queries: list[Query],
    all_hits: list[list[RetrievalHit]],
    reranker_type: str,
    top_k: int,
    alpha: float,
) -> list[list[RetrievalHit]]:
    mode = reranker_type.lower().strip()
    if mode != "overlap":
        raise ValueError(f"Unsupported reranker type: {reranker_type}")

    reranked: list[list[RetrievalHit]] = []
    for query, hits in zip(queries, all_hits):
        reranked.append(
            rerank_hits_by_overlap(
                query=query,
                hits=hits,
                top_k=top_k,
                alpha=alpha,
            )
        )
    return reranked
