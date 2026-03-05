from __future__ import annotations

from abc import ABC, abstractmethod

from src.config.types import ChunkingConfig
from src.pipeline.types import Chunk, Document

from .tokenizer import TokenizerAdapter


class BaseChunker(ABC):
    def __init__(self, config: ChunkingConfig) -> None:
        self.config = config
        self.tokenizer = TokenizerAdapter(config.tokenizer_name)

    @abstractmethod
    def chunk_document(self, document: Document) -> list[Chunk]:
        raise NotImplementedError

    def chunk_documents(self, documents: list[Document]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for document in documents:
            chunks.extend(self.chunk_document(document))
        return chunks

    def _make_chunk(
        self,
        *,
        document: Document,
        chunk_idx: int,
        text: str,
        start_token: int,
        end_token: int,
        strategy: str,
    ) -> Chunk:
        return Chunk(
            chunk_id=f"{document.document_id}::c{chunk_idx}",
            document_id=document.document_id,
            text=text.strip(),
            start_token=start_token,
            end_token=end_token,
            metadata={"strategy": strategy, **document.metadata},
        )

