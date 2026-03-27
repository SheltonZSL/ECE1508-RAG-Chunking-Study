from __future__ import annotations

import os


def normalize_hf_home_env() -> None:
    """Map deprecated TRANSFORMERS_CACHE to HF_HOME for current process."""
    legacy = str(os.getenv("TRANSFORMERS_CACHE", "")).strip()
    hf_home = str(os.getenv("HF_HOME", "")).strip()

    if legacy and not hf_home:
        os.environ["HF_HOME"] = legacy

    # Remove deprecated variable to avoid transformers deprecation warning.
    os.environ.pop("TRANSFORMERS_CACHE", None)
