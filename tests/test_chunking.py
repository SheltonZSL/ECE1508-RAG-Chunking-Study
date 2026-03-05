from __future__ import annotations

from src.chunking import create_chunker
from src.config.types import ChunkingConfig
from src.pipeline.types import Document


def test_fixed_chunking_with_overlap() -> None:
    cfg = ChunkingConfig(
        strategy="fixed",
        tokenizer_name="",
        chunk_size=5,
        overlap=2,
        min_chunk_size=2,
        max_chunk_size=8,
    )
    chunker = create_chunker(cfg)
    doc = Document(document_id="d1", text="a b c d e f g h i j")
    chunks = chunker.chunk_document(doc)

    assert len(chunks) >= 2
    assert chunks[0].text == "a b c d e"
    assert chunks[1].text.startswith("d e")
    assert chunks[0].start_token == 0
    assert chunks[0].end_token == 5


def test_structure_chunking_splits_paragraphs() -> None:
    cfg = ChunkingConfig(
        strategy="structure",
        tokenizer_name="",
        chunk_size=100,
        overlap=0,
        min_chunk_size=2,
        max_chunk_size=20,
    )
    chunker = create_chunker(cfg)
    doc = Document(document_id="d2", text="para one\n\npara two\n\npara three")
    chunks = chunker.chunk_document(doc)
    assert len(chunks) == 3
    assert chunks[0].text == "para one"


def test_adaptive_chunking_respects_max() -> None:
    cfg = ChunkingConfig(
        strategy="adaptive",
        tokenizer_name="",
        chunk_size=10,
        overlap=0,
        min_chunk_size=3,
        max_chunk_size=7,
    )
    chunker = create_chunker(cfg)
    doc = Document(
        document_id="d3",
        text="One short sentence. Another short sentence. Third one here.",
    )
    chunks = chunker.chunk_document(doc)
    assert chunks
    assert all(len(chunk.text.split()) <= 7 for chunk in chunks)

