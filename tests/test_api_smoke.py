from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import yaml

from scripts.serve_dashboard import (
    DashboardHTTPRequestHandler,
    InteractiveRAGService,
    ROOT,
    ThreadingTCPServer,
)


def _get_json(url: str) -> dict:
    with urlopen(url, timeout=10) as resp:  # noqa: S310 - local test server only
        body = resp.read().decode("utf-8")
        return json.loads(body)


def _post_json(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    request = Request(
        url=url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=20) as resp:  # noqa: S310 - local test server only
        body = resp.read().decode("utf-8")
        return json.loads(body)


def _wait_until_ready(base_url: str, timeout_s: float = 10.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            payload = _get_json(f"{base_url}/api/health")
            if payload.get("status") == "ok":
                return
        except Exception:
            pass
        time.sleep(0.2)
    raise TimeoutError(f"API server did not become ready within {timeout_s} seconds.")


def _build_test_config(tmp_path: Path) -> Path:
    base_cfg_path = ROOT / "configs" / "portable_interactive.yaml"
    raw = yaml.safe_load(base_cfg_path.read_text(encoding="utf-8"))
    raw["dataset"]["data_dir"] = str((ROOT / "data" / "demo").as_posix())
    raw["retriever"]["index_dir"] = str((tmp_path / "indexes").as_posix())
    raw["run"]["experiment_name"] = "portable_api_smoke"
    cfg_path = tmp_path / "portable_api_smoke.yaml"
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return cfg_path


def test_dashboard_api_smoke(tmp_path) -> None:
    cfg_path = _build_test_config(tmp_path)
    service = InteractiveRAGService(str(cfg_path))
    DashboardHTTPRequestHandler.service = service
    DashboardHTTPRequestHandler.root_dir = ROOT

    server = ThreadingTCPServer(("127.0.0.1", 0), DashboardHTTPRequestHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    base_url = f"http://{host}:{port}"

    try:
        _wait_until_ready(base_url)

        health = _get_json(f"{base_url}/api/health")
        assert health.get("status") == "ok"

        defaults = _get_json(f"{base_url}/api/defaults")
        assert "defaults" in defaults
        assert "options" in defaults
        assert defaults["defaults"]["backend"] in ("dense", "bm25")

        examples = _get_json(f"{base_url}/api/examples?limit=3")
        assert "examples" in examples
        assert isinstance(examples["examples"], list)
        assert len(examples["examples"]) >= 1

        try:
            _get_json(f"{base_url}/README.md")
            assert False, "Expected README.md to be blocked by static path allowlist"
        except HTTPError as exc:
            assert exc.code == 404

        ask_payload = {
            "question": "Who discovered penicillin?",
            "backend": "bm25",
            "strategy": "fixed",
            "chunk_size": 80,
            "overlap": 12,
            "top_k": 5,
            "config": str(cfg_path),
            "with_generation": False,
        }
        ask = _post_json(f"{base_url}/api/ask", ask_payload)
        assert isinstance(ask.get("answer", ""), str)
        assert ask.get("answer", "").strip()
        assert isinstance(ask.get("hits"), list)
        assert len(ask["hits"]) > 0
        assert "timings_ms" in ask
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
