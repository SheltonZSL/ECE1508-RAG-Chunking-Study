from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DatasetConfig:
    query_dataset: str = "nq_open"
    query_split: str = "validation"
    corpus_dataset: str = "wiki_dpr"
    corpus_config: str | None = "psgs_w100.nq.exact"
    corpus_split: str = "train"
    corpus_streaming: bool = False
    max_queries: int | None = 500
    max_corpus_docs: int | None = 10000
    data_dir: str = "data/processed"


@dataclass
class ChunkingConfig:
    strategy: str = "fixed"
    tokenizer_name: str = "intfloat/e5-base-v2"
    chunk_size: int = 256
    overlap: int = 32
    min_chunk_size: int = 80
    max_chunk_size: int = 220

    def validate(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if self.overlap < 0:
            raise ValueError("overlap must be >= 0")
        if self.overlap >= self.chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        if self.min_chunk_size <= 0 or self.max_chunk_size <= 0:
            raise ValueError("adaptive chunk sizes must be > 0")
        if self.min_chunk_size > self.max_chunk_size:
            raise ValueError("min_chunk_size must be <= max_chunk_size")


@dataclass
class RetrieverConfig:
    backend: str = "dense"
    model_name: str = "intfloat/e5-base-v2"
    batch_size: int = 16
    normalize_embeddings: bool = True
    device: str = "auto"
    index_dir: str = "data/indexes"


@dataclass
class GeneratorConfig:
    model_name: str = "google/flan-t5-base"
    fallback_model_name: str = "google/flan-t5-small"
    max_new_tokens: int = 64
    temperature: float = 0.0
    device: str = "auto"


@dataclass
class RetrievalConfig:
    top_k: int = 5


@dataclass
class EvalConfig:
    max_eval_queries: int | None = 200
    compute_efficiency: bool = True


@dataclass
class MatrixConfig:
    strategies: list[str] = field(default_factory=lambda: ["fixed", "structure", "adaptive"])
    chunk_sizes: list[int] = field(default_factory=lambda: [128, 256, 384])
    overlaps: list[int] = field(default_factory=lambda: [0, 32, 64])
    top_ks: list[int] = field(default_factory=lambda: [3, 5, 10])
    backends: list[str] = field(default_factory=lambda: ["dense", "bm25"])


@dataclass
class RunConfig:
    experiment_name: str = "baseline_dense"
    seed: int = 42
    save_predictions: bool = True
    results_dir: str = "results"
    matrix: MatrixConfig = field(default_factory=MatrixConfig)


@dataclass
class PipelineConfig:
    dataset: DatasetConfig
    retriever: RetrieverConfig
    generator: GeneratorConfig
    chunking: ChunkingConfig
    retrieval: RetrievalConfig
    eval: EvalConfig
    run: RunConfig

    def validate(self) -> None:
        self.chunking.validate()
        if self.retrieval.top_k <= 0:
            raise ValueError("retrieval.top_k must be > 0")
        if self.retriever.batch_size <= 0:
            raise ValueError("retriever.batch_size must be > 0")
        if self.generator.max_new_tokens <= 0:
            raise ValueError("generator.max_new_tokens must be > 0")
        if self.eval.max_eval_queries is not None and self.eval.max_eval_queries <= 0:
            raise ValueError("eval.max_eval_queries must be > 0 when provided")


def _read_section(raw: dict[str, Any], key: str) -> dict[str, Any]:
    section = raw.get(key)
    if section is None:
        raise ValueError(f"Missing required config section: '{key}'")
    if not isinstance(section, dict):
        raise ValueError(f"Config section '{key}' must be a mapping")
    return section


def build_pipeline_config(raw: dict[str, Any]) -> PipelineConfig:
    dataset = DatasetConfig(**_read_section(raw, "dataset"))
    retriever = RetrieverConfig(**_read_section(raw, "retriever"))
    generator = GeneratorConfig(**_read_section(raw, "generator"))
    chunking = ChunkingConfig(**_read_section(raw, "chunking"))
    retrieval = RetrievalConfig(**_read_section(raw, "retrieval"))
    eval_cfg = EvalConfig(**_read_section(raw, "eval"))

    run_raw = _read_section(raw, "run")
    matrix_raw = run_raw.get("matrix", {})
    if matrix_raw is None:
        matrix_raw = {}
    if not isinstance(matrix_raw, dict):
        raise ValueError("run.matrix must be a mapping")
    run_payload = {k: v for k, v in run_raw.items() if k != "matrix"}
    run = RunConfig(matrix=MatrixConfig(**matrix_raw), **run_payload)

    cfg = PipelineConfig(
        dataset=dataset,
        retriever=retriever,
        generator=generator,
        chunking=chunking,
        retrieval=retrieval,
        eval=eval_cfg,
        run=run,
    )
    cfg.validate()
    return cfg
