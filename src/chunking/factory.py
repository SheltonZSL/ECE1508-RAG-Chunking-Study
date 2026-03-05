from __future__ import annotations

from src.config.types import ChunkingConfig

from .adaptive import AdaptiveChunker
from .base import BaseChunker
from .fixed import FixedChunker
from .structure import StructureChunker


def create_chunker(config: ChunkingConfig) -> BaseChunker:
    strategy = config.strategy.lower().strip()
    if strategy == "fixed":
        return FixedChunker(config)
    if strategy == "structure":
        return StructureChunker(config)
    if strategy == "adaptive":
        return AdaptiveChunker(config)
    raise ValueError(f"Unsupported chunking strategy: {config.strategy}")

