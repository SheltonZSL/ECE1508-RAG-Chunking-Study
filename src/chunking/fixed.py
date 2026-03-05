from __future__ import annotations

from src.pipeline.types import Chunk, Document

from .base import BaseChunker


class FixedChunker(BaseChunker):
    def chunk_document(self, document: Document) -> list[Chunk]:
        token_ids = self.tokenizer.encode(document.text)
        if not token_ids:
            return []

        size = self.config.chunk_size
        step = size - self.config.overlap
        chunks: list[Chunk] = []
        chunk_idx = 0

        for start in range(0, len(token_ids), step):
            piece = token_ids[start : start + size]
            if not piece:
                continue
            text = self.tokenizer.decode(piece)
            if not text:
                continue
            end = start + len(piece)
            chunks.append(
                self._make_chunk(
                    document=document,
                    chunk_idx=chunk_idx,
                    text=text,
                    start_token=start,
                    end_token=end,
                    strategy="fixed",
                )
            )
            chunk_idx += 1
            if end >= len(token_ids):
                break
        return chunks

