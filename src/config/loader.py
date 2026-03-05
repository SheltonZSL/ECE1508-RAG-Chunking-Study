from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .types import PipelineConfig, build_pipeline_config


def load_config(path: str | Path) -> PipelineConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle) or {}

    return build_pipeline_config(raw)

