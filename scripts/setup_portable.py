from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str]) -> None:
    printable = " ".join(cmd)
    print(f"[setup_portable] running: {printable}")
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def _ensure_exists(path: Path, hint: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}\nHint: {hint}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One-click setup for instant interactive demo mode."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/portable_interactive.yaml",
        help="Config path for dashboard service.",
    )
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Install Python dependencies from requirements.txt before launching.",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open demo page automatically in browser.",
    )
    args = parser.parse_args()

    _ensure_exists(ROOT / "requirements.txt", "Run from repository root.")
    _ensure_exists(ROOT / args.config, "Check --config path.")
    _ensure_exists(ROOT / "data" / "demo" / "corpus.jsonl", "Make sure demo data is committed.")
    _ensure_exists(ROOT / "data" / "demo" / "queries.jsonl", "Make sure demo data is committed.")

    if args.install_deps:
        _run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

    serve_cmd = [
        sys.executable,
        "scripts/serve_dashboard.py",
        "--config",
        args.config,
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    if args.open:
        serve_cmd.append("--open")

    print("[setup_portable] Launching interactive demo service...")
    _run(serve_cmd)


if __name__ == "__main__":
    main()

