from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import faiss
except ImportError as exc:  # pragma: no cover
    faiss = None  # type: ignore[assignment]
    _FAISS_IMPORT_ERROR = exc
else:
    _FAISS_IMPORT_ERROR = None


def _require_faiss() -> None:
    if faiss is None:
        raise RuntimeError(
            "faiss is required for dense retrieval. Install faiss-cpu."
        ) from _FAISS_IMPORT_ERROR


class FaissStore:
    def __init__(self, dimension: int) -> None:
        _require_faiss()
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)

    def add(self, embeddings: np.ndarray) -> None:
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        self.index.add(embeddings)

    def search(self, query_embeddings: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        if query_embeddings.dtype != np.float32:
            query_embeddings = query_embeddings.astype(np.float32)
        scores, indices = self.index.search(query_embeddings, top_k)
        return scores, indices

    def save(self, index_path: str | Path) -> None:
        _require_faiss()
        path = Path(index_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))

    @classmethod
    def load(cls, index_path: str | Path) -> "FaissStore":
        _require_faiss()
        idx = faiss.read_index(str(index_path))
        store = cls(dimension=idx.d)
        store.index = idx
        return store

