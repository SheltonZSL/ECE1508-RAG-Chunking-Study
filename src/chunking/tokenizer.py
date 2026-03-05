from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

try:
    from transformers import AutoTokenizer
except ImportError:  # pragma: no cover - optional at import time
    AutoTokenizer = None  # type: ignore[assignment]


@dataclass
class TokenizerAdapter:
    tokenizer_name: str

    def __post_init__(self) -> None:
        self._mode = "whitespace"
        self._tokenizer = None
        if AutoTokenizer is not None and self.tokenizer_name:
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
                self._mode = "hf"
            except Exception:
                self._mode = "whitespace"

    def encode(self, text: str) -> list[int] | list[str]:
        if self._mode == "hf" and self._tokenizer is not None:
            return self._tokenizer.encode(text, add_special_tokens=False)
        return text.split()

    def decode(self, token_ids: Iterable[int] | Iterable[str]) -> str:
        token_list = list(token_ids)
        if self._mode == "hf" and self._tokenizer is not None:
            return self._tokenizer.decode(token_list, skip_special_tokens=True).strip()
        return " ".join(str(tok) for tok in token_list).strip()

    def token_count(self, text: str) -> int:
        return len(self.encode(text))

