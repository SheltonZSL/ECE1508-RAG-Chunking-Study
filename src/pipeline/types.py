from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    document_id: str
    text: str
    title: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    text: str
    start_token: int
    end_token: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Query:
    query_id: str
    question: str
    answers: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalHit:
    query_id: str
    chunk_id: str
    document_id: str
    score: float
    rank: int
    chunk_text: str


@dataclass
class QAPrediction:
    query_id: str
    question: str
    prediction: str
    gold_answers: list[str]
    retrieved_chunk_ids: list[str]
    retrieved_texts: list[str]
    latency_ms: float
    context_char_len: int


@dataclass
class EvalRecord:
    query_id: str
    em: float
    f1: float
    recall_at_k: float
    reciprocal_rank: float
    metadata: dict[str, Any] = field(default_factory=dict)

