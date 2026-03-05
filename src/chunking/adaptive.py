from __future__ import annotations

from src.pipeline.types import Chunk, Document
from src.utils.text import split_sentences

from .base import BaseChunker
from .fixed import FixedChunker


class AdaptiveChunker(BaseChunker):
    def __init__(self, config) -> None:
        super().__init__(config)
        self._fallback = FixedChunker(config)

    def chunk_document(self, document: Document) -> list[Chunk]:
        sentences = split_sentences(document.text)
        if not sentences:
            return []

        min_size = self.config.min_chunk_size
        max_size = self.config.max_chunk_size

        chunks: list[Chunk] = []
        chunk_idx = 0
        cursor = 0
        current_parts: list[str] = []
        current_tokens = 0

        def flush_current() -> None:
            nonlocal chunk_idx, cursor, current_parts, current_tokens
            if not current_parts:
                return
            text = " ".join(current_parts).strip()
            token_count = self.tokenizer.token_count(text)
            chunks.append(
                self._make_chunk(
                    document=document,
                    chunk_idx=chunk_idx,
                    text=text,
                    start_token=cursor,
                    end_token=cursor + token_count,
                    strategy="adaptive",
                )
            )
            chunk_idx += 1
            cursor += token_count
            current_parts = []
            current_tokens = 0

        for sentence in sentences:
            sentence_tokens = self.tokenizer.token_count(sentence)
            if sentence_tokens > max_size:
                flush_current()
                temp_doc = Document(
                    document_id=document.document_id,
                    text=sentence,
                    title=document.title,
                    metadata=document.metadata,
                )
                for sub in self._fallback.chunk_document(temp_doc):
                    sub.chunk_id = f"{document.document_id}::c{chunk_idx}"
                    sub.start_token += cursor
                    sub.end_token += cursor
                    sub.metadata["strategy"] = "adaptive"
                    chunks.append(sub)
                    chunk_idx += 1
                cursor += sentence_tokens
                continue

            projected = current_tokens + sentence_tokens
            if projected > max_size and current_tokens >= min_size:
                flush_current()

            current_parts.append(sentence)
            current_tokens += sentence_tokens

        flush_current()
        return chunks

