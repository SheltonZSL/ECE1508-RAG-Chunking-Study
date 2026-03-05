from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from src.config.types import RetrieverConfig
from src.pipeline.types import Chunk, Query, RetrievalHit

from .faiss_store import FaissStore


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.shape).float()
    masked = last_hidden_state * mask
    summed = masked.sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-9)
    return summed / denom


class DenseRetriever:
    def __init__(self, config: RetrieverConfig) -> None:
        self.config = config
        self.device = _resolve_device(config.device)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModel.from_pretrained(config.model_name)
        self.model.to(self.device)
        self.model.eval()

        self.store: FaissStore | None = None
        self.chunk_ids: list[str] = []
        self.chunk_texts: list[str] = []
        self.document_ids: list[str] = []

    @property
    def _use_e5_prefix(self) -> bool:
        return "e5" in self.config.model_name.lower()

    def _encode(self, texts: list[str], prefix: str) -> np.ndarray:
        vectors: list[np.ndarray] = []
        batch_size = self.config.batch_size
        with torch.no_grad():
            for start in tqdm(range(0, len(texts), batch_size), desc=f"Encoding {prefix.strip()}"):
                batch = texts[start : start + batch_size]
                if self._use_e5_prefix:
                    batch = [f"{prefix}{x}" for x in batch]
                tokens = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=512,
                )
                tokens = {k: v.to(self.device) for k, v in tokens.items()}
                outputs = self.model(**tokens)
                pooled = _mean_pool(outputs.last_hidden_state, tokens["attention_mask"])
                emb = pooled.detach().cpu().numpy().astype(np.float32)
                vectors.append(emb)
        merged = np.concatenate(vectors, axis=0)
        if self.config.normalize_embeddings:
            norms = np.linalg.norm(merged, axis=1, keepdims=True) + 1e-12
            merged = merged / norms
        return merged

    def fit(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise ValueError("Cannot build index from empty chunks")
        self.chunk_ids = [chunk.chunk_id for chunk in chunks]
        self.chunk_texts = [chunk.text for chunk in chunks]
        self.document_ids = [chunk.document_id for chunk in chunks]
        embeddings = self._encode(self.chunk_texts, prefix="passage: ")
        self.store = FaissStore(dimension=embeddings.shape[1])
        self.store.add(embeddings)

    def retrieve(self, queries: list[Query], top_k: int) -> list[list[RetrievalHit]]:
        if self.store is None:
            raise RuntimeError("DenseRetriever is not fitted or loaded")
        query_texts = [query.question for query in queries]
        query_emb = self._encode(query_texts, prefix="query: ")
        scores, indices = self.store.search(query_emb, top_k)

        all_hits: list[list[RetrievalHit]] = []
        for q_idx, query in enumerate(queries):
            query_hits: list[RetrievalHit] = []
            for rank, (score, idx) in enumerate(zip(scores[q_idx], indices[q_idx]), start=1):
                if idx < 0:
                    continue
                query_hits.append(
                    RetrievalHit(
                        query_id=query.query_id,
                        chunk_id=self.chunk_ids[idx],
                        document_id=self.document_ids[idx],
                        score=float(score),
                        rank=rank,
                        chunk_text=self.chunk_texts[idx],
                    )
                )
            all_hits.append(query_hits)
        return all_hits

    def save(self, index_dir: str | Path) -> None:
        if self.store is None:
            raise RuntimeError("DenseRetriever has no index to save")
        root = Path(index_dir)
        root.mkdir(parents=True, exist_ok=True)
        self.store.save(root / "dense.index.faiss")
        meta = {
            "chunk_ids": self.chunk_ids,
            "chunk_texts": self.chunk_texts,
            "document_ids": self.document_ids,
            "model_name": self.config.model_name,
            "normalize_embeddings": self.config.normalize_embeddings,
        }
        with (root / "dense.meta.json").open("w", encoding="utf-8") as handle:
            json.dump(meta, handle, ensure_ascii=False, indent=2)

    def load(self, index_dir: str | Path) -> None:
        root = Path(index_dir)
        with (root / "dense.meta.json").open("r", encoding="utf-8") as handle:
            meta = json.load(handle)
        self.chunk_ids = list(meta["chunk_ids"])
        self.chunk_texts = list(meta["chunk_texts"])
        self.document_ids = list(meta["document_ids"])
        self.store = FaissStore.load(root / "dense.index.faiss")

