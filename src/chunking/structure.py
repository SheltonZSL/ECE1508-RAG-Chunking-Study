from __future__ import annotations

import re

from src.pipeline.types import Chunk, Document

from .base import BaseChunker
from .fixed import FixedChunker

_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n+")


class StructureChunker(BaseChunker):
    def __init__(self, config) -> None:
        super().__init__(config)
        self._fallback = FixedChunker(config)

    def chunk_document(self, document: Document) -> list[Chunk]:
        parts = [p.strip() for p in _PARAGRAPH_SPLIT_RE.split(document.text) if p.strip()]
        if not parts:
            return []

        chunks: list[Chunk] = []
        chunk_idx = 0
        cursor = 0

        for part in parts:
            token_count = self.tokenizer.token_count(part)
            if token_count <= self.config.chunk_size:
                chunks.append(
                    self._make_chunk(
                        document=document,
                        chunk_idx=chunk_idx,
                        text=part,
                        start_token=cursor,
                        end_token=cursor + token_count,
                        strategy="structure",
                    )
                )
                chunk_idx += 1
                cursor += token_count
                continue

            temp_doc = Document(
                document_id=document.document_id,
                text=part,
                title=document.title,
                metadata=document.metadata,
            )
            for sub in self._fallback.chunk_document(temp_doc):
                sub.chunk_id = f"{document.document_id}::c{chunk_idx}"
                sub.metadata["strategy"] = "structure"
                sub.start_token += cursor
                sub.end_token += cursor
                chunks.append(sub)
                chunk_idx += 1
            cursor += token_count

        return chunks

