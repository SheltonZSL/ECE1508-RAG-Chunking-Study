from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str]) -> None:
    printable = " ".join(cmd)
    print(f"[setup_full] running: {printable}")
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def _ensure_exists(path: Path, hint: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}\nHint: {hint}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One-click setup for full experiment mode (prepare + index + optional serve)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline_lite.yaml",
        help="Config path used for prepare/index/eval/serve steps.",
    )
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Install Python dependencies from requirements.txt first.",
    )
    parser.add_argument(
        "--skip-prepare",
        action="store_true",
        help="Skip prepare_data step.",
    )
    parser.add_argument(
        "--skip-index",
        action="store_true",
        help="Skip build_index step.",
    )
    parser.add_argument(
        "--run-qa",
        action="store_true",
        help="Run QA evaluation step after index build.",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Start dashboard service after setup steps.",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open demo page automatically when serving.",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Force index rebuild in build_index step.",
    )
    args = parser.parse_args()

    _ensure_exists(ROOT / "requirements.txt", "Run from repository root.")
    _ensure_exists(ROOT / args.config, "Check --config path.")

    if args.install_deps:
        _run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

    if not args.skip_prepare:
        _run([sys.executable, "scripts/prepare_data.py", "--config", args.config])

    if not args.skip_index:
        cmd = [sys.executable, "scripts/build_index.py", "--config", args.config]
        if args.force_rebuild:
            cmd.append("--force-rebuild")
        _run(cmd)

    if args.run_qa:
        _run([sys.executable, "scripts/run_qa_eval.py", "--config", args.config])

    if args.serve:
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
        print("[setup_full] Launching dashboard service...")
        _run(serve_cmd)

    if not args.serve:
        print("[setup_full] Setup completed.")


if __name__ == "__main__":
    main()

