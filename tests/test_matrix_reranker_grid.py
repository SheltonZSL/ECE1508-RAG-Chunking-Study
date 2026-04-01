from __future__ import annotations

from scripts.run_experiments import _iter_matrix, _run_name
from src.config.types import (
    ChunkingConfig,
    DatasetConfig,
    EvalConfig,
    GeneratorConfig,
    MatrixConfig,
    PipelineConfig,
    RetrievalConfig,
    RetrieverConfig,
    RunConfig,
)


def _build_cfg(matrix: MatrixConfig) -> PipelineConfig:
    return PipelineConfig(
        dataset=DatasetConfig(),
        retriever=RetrieverConfig(),
        generator=GeneratorConfig(),
        chunking=ChunkingConfig(),
        retrieval=RetrievalConfig(),
        eval=EvalConfig(),
        run=RunConfig(matrix=matrix),
    )


def test_iter_matrix_with_reranker_ablation_grid() -> None:
    cfg = _build_cfg(
        MatrixConfig(
            backends=["dense"],
            strategies=["fixed"],
            chunk_sizes=[256],
            overlaps=[32],
            top_ks=[5],
            reranker_enableds=[False, True],
            reranker_types=["overlap"],
            reranker_candidate_ks=[10, 20],
            reranker_alphas=[0.2, 0.8],
        )
    )
    runs = list(_iter_matrix(cfg))
    assert len(runs) == 5

    enabled_runs = [run for run in runs if run[5] is True]
    disabled_runs = [run for run in runs if run[5] is False]
    assert len(disabled_runs) == 1
    assert len(enabled_runs) == 4


def test_run_name_encodes_reranker_setting() -> None:
    base = _run_name(
        base_name="exp",
        backend="dense",
        strategy="fixed",
        chunk_size=256,
        overlap=32,
        top_k=5,
        reranker_enabled=False,
        reranker_type="none",
        reranker_candidate_k=20,
        reranker_alpha=0.5,
    )
    rr = _run_name(
        base_name="exp",
        backend="dense",
        strategy="fixed",
        chunk_size=256,
        overlap=32,
        top_k=5,
        reranker_enabled=True,
        reranker_type="overlap",
        reranker_candidate_k=20,
        reranker_alpha=0.5,
    )
    assert base.endswith("_rr0")
    assert rr.endswith("_rr1_overlap_rc20_ra050")
    assert base != rr
