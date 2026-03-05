from __future__ import annotations

import re

_WS_RE = re.compile(r"\s+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def normalize_whitespace(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


def split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    return [part.strip() for part in _SENTENCE_RE.split(text) if part.strip()]

