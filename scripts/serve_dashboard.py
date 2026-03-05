from __future__ import annotations

import argparse
import copy
import json
import os
import socketserver
import sys
import time
import traceback
import uuid
import webbrowser
from dataclasses import asdict
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import http.server

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.generation.hf_generator import HFGenerator
from src.pipeline.types import Query
from src.pipeline.workflows import (
    build_chunks,
    build_or_load_retriever,
    load_prepared_documents,
    load_prepared_queries,
)


def _to_int(raw: Any, default: int, low: int, high: int) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = default
    return max(low, min(high, value))


def _fallback_answer_from_hits(hits: list[Any]) -> str:
    if not hits:
        return "No evidence retrieved."
    text = str(hits[0].chunk_text).strip()
    if not text:
        return "Evidence retrieved, but text is empty."
    limit = 320
    return text[:limit] + ("..." if len(text) > limit else "")


class InteractiveRAGService:
    def __init__(self, config_path: str) -> None:
        self.base_config_path = config_path
        self.base_config = load_config(config_path)
        self._documents = None
        self._chunks_cache: dict[tuple[str, str, int, int], list[Any]] = {}
        self._retriever_cache: dict[tuple[str, str], Any] = {}
        self._generator_cache: dict[str, HFGenerator] = {}
        self._query_examples: list[str] | None = None

    def defaults(self) -> dict[str, Any]:
        cfg = self.base_config
        generation_available = bool(str(cfg.generator.model_name).strip())
        return {
            "config_path": self.base_config_path,
            "defaults": {
                "backend": cfg.retriever.backend,
                "strategy": cfg.chunking.strategy,
                "chunk_size": cfg.chunking.chunk_size,
                "overlap": cfg.chunking.overlap,
                "top_k": cfg.retrieval.top_k,
                "with_generation": generation_available,
            },
            "options": {
                "backends": cfg.run.matrix.backends,
                "strategies": cfg.run.matrix.strategies,
                "chunk_sizes": cfg.run.matrix.chunk_sizes,
                "overlaps": cfg.run.matrix.overlaps,
                "top_ks": cfg.run.matrix.top_ks,
            },
        }

    def examples(self, limit: int = 8) -> list[str]:
        if self._query_examples is not None:
            return self._query_examples[:limit]
        try:
            queries = load_prepared_queries(self.base_config)
            self._query_examples = [q.question for q in queries[: max(limit, 12)]]
        except Exception:
            self._query_examples = [
                "Who discovered penicillin?",
                "What is the capital of Japan?",
                "When was the first iPhone released?",
                "Which planet is known as the Red Planet?",
            ]
        return self._query_examples[:limit]

    def _load_documents_once(self):
        if self._documents is None:
            self._documents = load_prepared_documents(self.base_config)
        return self._documents

    def _build_config(self, payload: dict[str, Any]):
        cfg = copy.deepcopy(self.base_config)
        cfg.retriever.backend = str(payload.get("backend", cfg.retriever.backend)).strip().lower()
        cfg.chunking.strategy = str(payload.get("strategy", cfg.chunking.strategy)).strip().lower()
        cfg.chunking.chunk_size = _to_int(
            payload.get("chunk_size"), cfg.chunking.chunk_size, low=32, high=2048
        )
        cfg.chunking.overlap = _to_int(payload.get("overlap"), cfg.chunking.overlap, low=0, high=1024)
        cfg.retrieval.top_k = _to_int(payload.get("top_k"), cfg.retrieval.top_k, low=1, high=20)

        cfg_name = str(payload.get("config", self.base_config_path)).strip()
        if cfg_name and cfg_name != self.base_config_path:
            cfg = load_config(cfg_name)
            cfg.retriever.backend = str(payload.get("backend", cfg.retriever.backend)).strip().lower()
            cfg.chunking.strategy = str(payload.get("strategy", cfg.chunking.strategy)).strip().lower()
            cfg.chunking.chunk_size = _to_int(
                payload.get("chunk_size"), cfg.chunking.chunk_size, low=32, high=2048
            )
            cfg.chunking.overlap = _to_int(payload.get("overlap"), cfg.chunking.overlap, low=0, high=1024)
            cfg.retrieval.top_k = _to_int(payload.get("top_k"), cfg.retrieval.top_k, low=1, high=20)

        cfg.run.experiment_name = (
            f"interactive_{cfg.retriever.backend}_{cfg.chunking.strategy}_"
            f"c{cfg.chunking.chunk_size}_o{cfg.chunking.overlap}"
        )
        cfg.validate()
        return cfg

    def _get_chunks(self, cfg):
        key = (
            cfg.run.experiment_name,
            cfg.chunking.strategy,
            cfg.chunking.chunk_size,
            cfg.chunking.overlap,
        )
        if key in self._chunks_cache:
            return self._chunks_cache[key]

        docs = self._load_documents_once()
        index_root = Path(cfg.retriever.index_dir) / cfg.run.experiment_name
        index_root.mkdir(parents=True, exist_ok=True)
        chunks = build_chunks(cfg, docs, save_dir=index_root)
        self._chunks_cache[key] = chunks
        return chunks

    def _get_retriever(self, cfg, chunks):
        key = (cfg.run.experiment_name, cfg.retriever.backend)
        if key in self._retriever_cache:
            return self._retriever_cache[key]
        retriever = build_or_load_retriever(cfg, chunks, force_rebuild=False)
        self._retriever_cache[key] = retriever
        return retriever

    def _get_generator(self, cfg) -> HFGenerator:
        key = f"{cfg.generator.model_name}|{cfg.generator.fallback_model_name}|{cfg.generator.device}"
        if key in self._generator_cache:
            return self._generator_cache[key]
        generator = HFGenerator(cfg.generator)
        self._generator_cache[key] = generator
        return generator

    def ask(self, payload: dict[str, Any]) -> dict[str, Any]:
        question = str(payload.get("question", "")).strip()
        if not question:
            raise ValueError("Question cannot be empty.")

        with_generation = bool(payload.get("with_generation", True))
        cfg = self._build_config(payload)
        if with_generation and not str(cfg.generator.model_name).strip():
            with_generation = False
        chunks = self._get_chunks(cfg)
        retriever = self._get_retriever(cfg, chunks)

        query = Query(query_id=f"interactive_{uuid.uuid4().hex[:10]}", question=question, answers=[])
        retrieval_start = time.perf_counter()
        hits = retriever.retrieve([query], top_k=cfg.retrieval.top_k)[0]
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000.0

        contexts = [hit.chunk_text for hit in hits]
        context_char_len = sum(len(text) for text in contexts)
        generation_ms = 0.0
        if with_generation:
            generator = self._get_generator(cfg)
            generation_start = time.perf_counter()
            answer = generator.generate(question, contexts)
            generation_ms = (time.perf_counter() - generation_start) * 1000.0
        else:
            answer = _fallback_answer_from_hits(hits)

        total_ms = retrieval_ms + generation_ms
        return {
            "question": question,
            "answer": answer,
            "settings": {
                "backend": cfg.retriever.backend,
                "strategy": cfg.chunking.strategy,
                "chunk_size": cfg.chunking.chunk_size,
                "overlap": cfg.chunking.overlap,
                "top_k": cfg.retrieval.top_k,
                "with_generation": with_generation,
                "experiment_name": cfg.run.experiment_name,
            },
            "timings_ms": {
                "retrieval": retrieval_ms,
                "generation": generation_ms,
                "total": total_ms,
            },
            "context_char_len": context_char_len,
            "hits": [asdict(hit) for hit in hits],
        }


class DashboardHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    service: InteractiveRAGService
    root_dir: Path

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(self.root_dir), **kwargs)

    def _write_json(self, status: int, payload: dict[str, Any]) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/health":
            self._write_json(200, {"status": "ok", "time": time.time()})
            return
        if parsed.path == "/api/defaults":
            self._write_json(200, self.service.defaults())
            return
        if parsed.path == "/api/examples":
            params = parse_qs(parsed.query)
            limit = _to_int(params.get("limit", ["8"])[0], 8, low=1, high=50)
            self._write_json(200, {"examples": self.service.examples(limit=limit)})
            return
        super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/api/ask":
            self._write_json(404, {"error": "Not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length > 0 else b"{}"
            payload = json.loads(raw.decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("Request body must be a JSON object.")
            response = self.service.ask(payload)
            self._write_json(200, response)
        except ValueError as exc:
            self._write_json(400, {"error": str(exc)})
        except FileNotFoundError as exc:
            self._write_json(
                400,
                {
                    "error": str(exc),
                    "hint": "Run prepare_data.py first to create data/processed/*.jsonl",
                },
            )
        except Exception as exc:  # pragma: no cover
            self._write_json(
                500,
                {
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(limit=1),
                },
            )


class ThreadingTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve dashboard files and interactive RAG API.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--config", type=str, default="configs/portable_interactive.yaml")
    parser.add_argument("--open", action="store_true", help="Open demo page automatically")
    args = parser.parse_args()

    os.chdir(ROOT)
    service = InteractiveRAGService(args.config)
    DashboardHTTPRequestHandler.service = service
    DashboardHTTPRequestHandler.root_dir = ROOT

    with ThreadingTCPServer((args.host, args.port), DashboardHTTPRequestHandler) as httpd:
        dashboard_url = f"http://{args.host}:{args.port}/dashboard/"
        demo_url = f"http://{args.host}:{args.port}/dashboard/demo.html"
        print(f"Serving from: {ROOT}")
        print(f"Dashboard: {dashboard_url}")
        print(f"Interactive demo: {demo_url}")
        if args.open:
            webbrowser.open(demo_url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
