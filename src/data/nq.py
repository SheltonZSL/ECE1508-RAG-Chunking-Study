from __future__ import annotations

import itertools
import shutil
from pathlib import Path
from typing import Any

from tqdm import tqdm

from src.config.types import DatasetConfig
from src.utils.io import write_jsonl

try:
    from datasets import load_dataset
except ImportError as exc:  # pragma: no cover
    load_dataset = None  # type: ignore[assignment]
    _DATASETS_IMPORT_ERROR = exc
else:
    _DATASETS_IMPORT_ERROR = None


def _require_datasets() -> None:
    if load_dataset is None:
        raise RuntimeError(
            "datasets package is required. Install requirements first."
        ) from _DATASETS_IMPORT_ERROR


def _safe_load_dataset(
    path: str,
    *,
    name: str | None = None,
    split: str,
    streaming: bool = False,
    trust_remote_code: bool = False,
):
    try:
        return load_dataset(
            path,
            name,
            split=split,
            streaming=streaming,
            trust_remote_code=trust_remote_code,
        )
    except ValueError as exc:
        # Happens after switching between incompatible datasets versions with stale cache metadata.
        if "Feature type 'List' not found" not in str(exc):
            raise
        compat_cache = Path("data/hf_cache_compat")
        compat_cache.mkdir(parents=True, exist_ok=True)
        return load_dataset(
            path,
            name,
            split=split,
            streaming=streaming,
            cache_dir=str(compat_cache),
            download_mode="force_redownload",
            trust_remote_code=trust_remote_code,
        )


def _extract_answers(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    if isinstance(raw, dict):
        aliases = raw.get("aliases")
        if isinstance(aliases, list):
            return [str(x).strip() for x in aliases if str(x).strip()]
        value = raw.get("text")
        if isinstance(value, list):
            return [str(x).strip() for x in value if str(x).strip()]
    text = str(raw).strip()
    return [text] if text else []


def prepare_nq_open_data(cfg: DatasetConfig) -> tuple[Path, Path]:
    _require_datasets()
    data_dir = Path(cfg.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    query_rows = _prepare_queries(cfg)
    corpus_rows = _prepare_corpus(cfg)

    queries_path = data_dir / "queries.jsonl"
    corpus_path = data_dir / "corpus.jsonl"
    write_jsonl(queries_path, query_rows)
    write_jsonl(corpus_path, corpus_rows)
    return queries_path, corpus_path


def _prepare_queries(cfg: DatasetConfig) -> list[dict[str, Any]]:
    dataset = _safe_load_dataset(cfg.query_dataset, split=cfg.query_split)
    if cfg.max_queries is not None:
        dataset = dataset.select(range(min(cfg.max_queries, len(dataset))))

    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(tqdm(dataset, desc="Preparing queries")):
        answers = _extract_answers(row.get("answer"))
        if not answers:
            continue
        query_id = str(row.get("id", f"q_{idx}"))
        question = str(row.get("question", "")).strip()
        if not question:
            continue
        rows.append(
            {
                "query_id": query_id,
                "question": question,
                "answers": answers,
                "metadata": {"source": cfg.query_dataset, "split": cfg.query_split},
            }
        )
    return rows


def _prepare_corpus(cfg: DatasetConfig) -> list[dict[str, Any]]:
    def use_fallback(reason: str):
        dataset = _safe_load_dataset(
            "sentence-transformers/wikipedia-en-sentences",
            split="train",
            streaming=True,
        )
        return _build_generic_corpus_rows(
            dataset=dataset,
            cfg=cfg,
            source_dataset="sentence-transformers/wikipedia-en-sentences",
            source_config=None,
            fallback_reason=reason,
        )

    free_bytes = shutil.disk_usage(Path.cwd()).free
    if cfg.corpus_dataset == "wiki_dpr" and free_bytes < 4 * 1024**3:
        return use_fallback("insufficient_disk_for_wiki_dpr")

    try:
        dataset = _safe_load_dataset(
            cfg.corpus_dataset,
            name=cfg.corpus_config,
            split=cfg.corpus_split,
            streaming=cfg.corpus_streaming,
            trust_remote_code=True,
        )
    except Exception as exc:
        message = str(exc).lower()
        fallback_markers = [
            "dataset scripts are no longer supported",
            "trust_remote_code",
            "no space left on device",
            "feature type 'list' not found",
        ]
        if any(marker in message for marker in fallback_markers):
            return use_fallback(f"wiki_dpr_unavailable:{type(exc).__name__}")
        raise

    return _build_generic_corpus_rows(
        dataset=dataset,
        cfg=cfg,
        source_dataset=cfg.corpus_dataset,
        source_config=cfg.corpus_config,
        fallback_reason="",
    )


def _extract_corpus_fields(row: dict[str, Any], idx: int) -> tuple[str, str, str]:
    document_id = str(
        row.get("id")
        or row.get("docid")
        or row.get("document_id")
        or row.get("_id")
        or f"d_{idx}"
    )
    title = str(row.get("title", "")).strip()

    text_candidates = [
        row.get("text"),
        row.get("passage"),
        row.get("sentence"),
        row.get("contents"),
    ]
    text = ""
    for candidate in text_candidates:
        if candidate is None:
            continue
        if isinstance(candidate, str) and candidate.strip():
            text = candidate.strip()
            break
    return document_id, title, text


def _iter_limited(dataset, limit: int | None):
    if limit is None:
        return iter(dataset)
    # IterableDataset in streaming mode doesn't support len/select.
    if hasattr(dataset, "take"):
        return iter(dataset.take(limit))
    if hasattr(dataset, "select") and hasattr(dataset, "__len__"):
        return iter(dataset.select(range(min(limit, len(dataset)))))
    return itertools.islice(iter(dataset), limit)


def _build_generic_corpus_rows(
    dataset,
    cfg: DatasetConfig,
    source_dataset: str,
    source_config: str | None,
    fallback_reason: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    max_docs = cfg.max_corpus_docs or 10000
    iterator = _iter_limited(dataset, max_docs)

    for idx, row in enumerate(tqdm(iterator, desc="Preparing corpus", total=max_docs)):
        document_id, title, text = _extract_corpus_fields(row, idx)
        if not text:
            continue
        rows.append(
            {
                "document_id": document_id,
                "title": title,
                "text": text,
                "metadata": {
                    "source": source_dataset,
                    "config": source_config,
                    "fallback_reason": fallback_reason,
                },
            }
        )
    return rows
