from __future__ import annotations

from src.chunking import create_chunker
from src.config.types import ChunkingConfig, RetrieverConfig
from src.eval.qa_metrics import best_qa_scores
from src.eval.retrieval_metrics import compute_retrieval_for_query
from src.pipeline.types import Document, Query
from src.retrieval.bm25 import BM25Retriever


def test_bm25_smoke_pipeline() -> None:
    documents = [
        Document(document_id="d1", text="Paris is the capital of France."),
        Document(document_id="d2", text="Berlin is the capital of Germany."),
    ]
    query = Query(query_id="q1", question="What is the capital of France?", answers=["Paris"])

    chunk_cfg = ChunkingConfig(
        strategy="fixed",
        tokenizer_name="",
        chunk_size=16,
        overlap=0,
        min_chunk_size=4,
        max_chunk_size=12,
    )
    chunks = create_chunker(chunk_cfg).chunk_documents(documents)

    retriever = BM25Retriever(
        RetrieverConfig(backend="bm25", model_name="", batch_size=1, index_dir="data/indexes")
    )
    retriever.fit(chunks)
    hits = retriever.retrieve([query], top_k=2)[0]

    assert hits
    retrieval_metrics = compute_retrieval_for_query(hits, query.answers)
    assert retrieval_metrics["recall_at_k"] in (0.0, 1.0)

    prediction = hits[0].chunk_text
    em, f1 = best_qa_scores(prediction, query.answers)
    assert em in (0.0, 1.0)
    assert 0.0 <= f1 <= 1.0


def test_bm25_save_load_without_pickle(tmp_path) -> None:
    documents = [
        Document(document_id="d1", text="Penicillin was discovered by Alexander Fleming."),
        Document(document_id="d2", text="The capital of Canada is Ottawa."),
    ]
    query = Query(query_id="q1", question="Who discovered penicillin?", answers=["Alexander Fleming"])

    chunk_cfg = ChunkingConfig(
        strategy="fixed",
        tokenizer_name="",
        chunk_size=16,
        overlap=0,
        min_chunk_size=4,
        max_chunk_size=12,
    )
    chunks = create_chunker(chunk_cfg).chunk_documents(documents)
    retriever = BM25Retriever(
        RetrieverConfig(backend="bm25", model_name="", batch_size=1, index_dir=str(tmp_path))
    )
    retriever.fit(chunks)

    index_dir = tmp_path / "bm25_index"
    retriever.save(index_dir)
    assert (index_dir / "bm25.meta.json").exists()
    assert not (index_dir / "bm25.model.pkl").exists()

    loaded = BM25Retriever(
        RetrieverConfig(backend="bm25", model_name="", batch_size=1, index_dir=str(tmp_path))
    )
    loaded.load(index_dir)
    hits = loaded.retrieve([query], top_k=2)[0]
    assert hits
