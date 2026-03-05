from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.data import prepare_nq_open_data
from src.utils.seed import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare NQ queries and wiki corpus.")
    parser.add_argument("--config", type=str, default="configs/baseline_dense.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(config.run.seed)
    queries_path, corpus_path = prepare_nq_open_data(config.dataset)

    print(f"Prepared queries: {queries_path}")
    print(f"Prepared corpus: {corpus_path}")


if __name__ == "__main__":
    main()

