from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.pipeline.workflows import build_chunks, build_or_load_retriever, load_prepared_documents
from src.utils.seed import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Build retrieval index from prepared data.")
    parser.add_argument("--config", type=str, default="configs/baseline_dense.yaml")
    parser.add_argument("--force-rebuild", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(config.run.seed)

    documents = load_prepared_documents(config)
    index_root = Path(config.retriever.index_dir) / config.run.experiment_name
    index_root.mkdir(parents=True, exist_ok=True)

    chunks = build_chunks(config, documents, save_dir=index_root)
    build_or_load_retriever(config, chunks, force_rebuild=args.force_rebuild)

    print(f"Built index at: {index_root}")
    print(f"Chunk count: {len(chunks)}")


if __name__ == "__main__":
    main()
