from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


LEGACY_TOPK_SUFFIX = re.compile(r"_k\d+$")


def _dir_size_bytes(path: Path) -> int:
    total = 0
    for file in path.rglob("*"):
        if file.is_file():
            total += file.stat().st_size
    return total


def _human_mb(size_bytes: int) -> str:
    return f"{size_bytes / (1024 * 1024):.2f} MB"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove legacy matrix index directories that were duplicated by top_k suffix."
    )
    parser.add_argument(
        "--index-dir",
        type=str,
        default="data/indexes",
        help="Root index directory.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete matched directories. Without this flag, only dry-run output is shown.",
    )
    args = parser.parse_args()

    root = Path(args.index_dir)
    if not root.exists():
        print(f"Index directory does not exist: {root}")
        return

    candidates = [d for d in root.iterdir() if d.is_dir() and LEGACY_TOPK_SUFFIX.search(d.name)]
    if not candidates:
        print("No legacy top_k index directories found.")
        return

    total_bytes = 0
    print("Matched legacy index directories:")
    for directory in sorted(candidates):
        size_bytes = _dir_size_bytes(directory)
        total_bytes += size_bytes
        print(f"- {directory} ({_human_mb(size_bytes)})")

    print(f"\nMatched directories: {len(candidates)}")
    print(f"Total reclaimable size: {_human_mb(total_bytes)}")

    if not args.apply:
        print("\nDry-run mode. Re-run with --apply to delete.")
        return

    for directory in candidates:
        shutil.rmtree(directory, ignore_errors=False)
    print("Deletion completed.")


if __name__ == "__main__":
    main()

