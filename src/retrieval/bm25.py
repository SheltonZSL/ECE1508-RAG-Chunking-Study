from __future__ import annotations

import json
import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from src.config.types import RetrieverConfig
from src.pipeline.types import Chunk, Query, RetrievalHit

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Retriever:
    def __init__(self, config: RetrieverConfig) -> None:
        self.config = config
        self.model: BM25Okapi | None = None
        self.chunk_ids: list[str] = []
        self.chunk_texts: list[str] = []
        self.document_ids: list[str] = []

    def fit(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise ValueError("Cannot build BM25 index from empty chunks")
        self.chunk_ids = [chunk.chunk_id for chunk in chunks]
        self.chunk_texts = [chunk.text for chunk in chunks]
        self.document_ids = [chunk.document_id for chunk in chunks]
        tokenized = [_tokenize(text) for text in self.chunk_texts]
        self.model = BM25Okapi(tokenized)

    def retrieve(self, queries: list[Query], top_k: int) -> list[list[RetrievalHit]]:
        if self.model is None:
            raise RuntimeError("BM25Retriever is not fitted or loaded")
        all_hits: list[list[RetrievalHit]] = []
        for query in queries:
            tokens = _tokenize(query.question)
            scores = self.model.get_scores(tokens)
            ranked = sorted(enumerate(scores), key=lambda x: float(x[1]), reverse=True)[:top_k]
            hits: list[RetrievalHit] = []
            for rank, (idx, score) in enumerate(ranked, start=1):
                hits.append(
                    RetrievalHit(
                        query_id=query.query_id,
                        chunk_id=self.chunk_ids[idx],
                        document_id=self.document_ids[idx],
                        score=float(score),
                        rank=rank,
                        chunk_text=self.chunk_texts[idx],
                    )
                )
            all_hits.append(hits)
        return all_hits

    def save(self, index_dir: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("BM25Retriever has no index to save")
        root = Path(index_dir)
        root.mkdir(parents=True, exist_ok=True)
        with (root / "bm25.model.pkl").open("wb") as handle:
            pickle.dump(self.model, handle)
        meta = {
            "chunk_ids": self.chunk_ids,
            "chunk_texts": self.chunk_texts,
            "document_ids": self.document_ids,
        }
        with (root / "bm25.meta.json").open("w", encoding="utf-8") as handle:
            json.dump(meta, handle, ensure_ascii=False, indent=2)

    def load(self, index_dir: str | Path) -> None:
        root = Path(index_dir)
        with (root / "bm25.model.pkl").open("rb") as handle:
            self.model = pickle.load(handle)
        with (root / "bm25.meta.json").open("r", encoding="utf-8") as handle:
            meta = json.load(handle)
        self.chunk_ids = list(meta["chunk_ids"])
        self.chunk_texts = list(meta["chunk_texts"])
        self.document_ids = list(meta["document_ids"])

