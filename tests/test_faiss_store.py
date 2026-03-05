from __future__ import annotations

import numpy as np
import pytest

from src.retrieval.faiss_store import FaissStore


def test_faiss_save_load_search(tmp_path) -> None:
    try:
        store = FaissStore(dimension=4)
    except RuntimeError:
        pytest.skip("faiss is not available")

    vectors = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    store.add(vectors)
    index_path = tmp_path / "test.index.faiss"
    store.save(index_path)

    loaded = FaissStore.load(index_path)
    query = np.array([[0.9, 0.1, 0.0, 0.0]], dtype=np.float32)
    scores, indices = loaded.search(query, top_k=1)
    assert indices.shape == (1, 1)
    assert int(indices[0][0]) == 0
    assert float(scores[0][0]) > 0.0

