from __future__ import annotations

import os
import platform
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

from src.utils.io import write_json, write_jsonl


def save_eval_outputs(
    *,
    out_dir: str | Path,
    metrics: dict[str, Any],
    predictions: list[dict[str, Any]],
    retrieval_hits: list[dict[str, Any]],
    error_analysis: str,
) -> None:
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "metrics.json", metrics)
    write_jsonl(output / "predictions.jsonl", predictions)
    write_jsonl(output / "retrieval_hits.jsonl", retrieval_hits)
    (output / "error_analysis.md").write_text(error_analysis, encoding="utf-8")


def _iso_utc(ts: datetime) -> str:
    return ts.astimezone(timezone.utc).isoformat()


def _safe_package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _safe_git_info(repo_root: Path) -> dict[str, str | None]:
    def run_git(*args: str) -> str | None:
        try:
            value = subprocess.check_output(
                ["git", *args],
                cwd=str(repo_root),
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            return value or None
        except Exception:
            return None

    return {
        "commit": run_git("rev-parse", "HEAD"),
        "branch": run_git("rev-parse", "--abbrev-ref", "HEAD"),
        "is_dirty": bool(run_git("status", "--porcelain")),
    }


def _safe_cuda_available() -> bool | None:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return None


def _safe_total_memory_gb() -> float | None:
    try:
        if hasattr(os, "sysconf"):
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            if isinstance(pages, int) and isinstance(page_size, int) and pages > 0 and page_size > 0:
                return round((pages * page_size) / (1024**3), 2)
    except Exception:
        pass
    try:
        import ctypes

        class _MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = _MemoryStatus()
        status.dwLength = ctypes.sizeof(_MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):  # type: ignore[attr-defined]
            return round(status.ullTotalPhys / (1024**3), 2)
    except Exception:
        pass
    return None


def build_run_manifest(
    *,
    script_name: str,
    script_args: dict[str, Any],
    config: Any,
    config_path: str,
    stage: str,
    start_time_utc: datetime,
    end_time_utc: datetime,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    config_payload: Any = asdict(config) if is_dataclass(config) else config
    duration_seconds = max(0.0, (end_time_utc - start_time_utc).total_seconds())

    manifest = {
        "schema_version": "1.0",
        "stage": stage,
        "script": {
            "name": script_name,
            "args": script_args,
        },
        "timestamps": {
            "start_utc": _iso_utc(start_time_utc),
            "end_utc": _iso_utc(end_time_utc),
            "duration_seconds": duration_seconds,
        },
        "environment": {
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "total_memory_gb": _safe_total_memory_gb(),
            "cuda_available": _safe_cuda_available(),
            "package_versions": {
                "datasets": _safe_package_version("datasets"),
                "transformers": _safe_package_version("transformers"),
                "faiss-cpu": _safe_package_version("faiss-cpu"),
                "rank-bm25": _safe_package_version("rank-bm25"),
                "torch": _safe_package_version("torch"),
                "numpy": _safe_package_version("numpy"),
            },
        },
        "git": _safe_git_info(repo_root),
        "config_path": config_path,
        "config": config_payload,
    }
    if extra:
        manifest["extra"] = extra
    return manifest


def save_run_manifest(out_dir: str | Path, manifest: dict[str, Any]) -> None:
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "run_manifest.json", manifest)


def build_error_analysis(predictions: list[dict[str, Any]], limit: int = 20) -> str:
    wrong = [row for row in predictions if row.get("em", 0.0) < 1.0]
    header = [
        "# Error Analysis",
        "",
        f"Total predictions: {len(predictions)}",
        f"Exact-match failures: {len(wrong)}",
        "",
    ]
    if not wrong:
        header.append("No EM errors found.")
        return "\n".join(header)

    header.append("## Representative Errors")
    header.append("")

    lines: list[str] = list(header)
    for idx, row in enumerate(wrong[:limit], start=1):
        lines.append(f"### {idx}. Query {row['query_id']}")
        lines.append(f"- Question: {row['question']}")
        lines.append(f"- Prediction: {row['prediction']}")
        lines.append(f"- Gold answers: {row['gold_answers']}")
        lines.append(f"- Retrieved chunk ids: {row['retrieved_chunk_ids']}")
        lines.append("")
    return "\n".join(lines)
