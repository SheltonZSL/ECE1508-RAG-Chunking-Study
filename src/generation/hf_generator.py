from __future__ import annotations

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.config.types import GeneratorConfig


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


class HFGenerator:
    def __init__(self, config: GeneratorConfig) -> None:
        self.config = config
        self.device = _resolve_device(config.device)
        self.model_name = config.model_name
        self.tokenizer = None
        self.model = None
        self._load_model(self.model_name)

    def _load_model(self, model_name: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        self.model_name = model_name

    def _build_prompt(self, question: str, contexts: list[str]) -> str:
        context_text = "\n\n".join(f"[{idx + 1}] {c}" for idx, c in enumerate(contexts))
        return (
            "Answer the question using the provided context.\n\n"
            f"Question: {question}\n\n"
            f"Context:\n{context_text}\n\n"
            "Answer:"
        )

    def generate(self, question: str, contexts: list[str]) -> str:
        if self.tokenizer is None or self.model is None:
            raise RuntimeError("Generator model is not loaded")

        prompt = self._build_prompt(question, contexts)
        try:
            return self._generate_once(prompt)
        except RuntimeError as exc:
            is_oom = "out of memory" in str(exc).lower()
            can_fallback = (
                is_oom
                and self.config.fallback_model_name
                and self.model_name != self.config.fallback_model_name
            )
            if not can_fallback:
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._load_model(self.config.fallback_model_name)
            return self._generate_once(prompt)

    def _generate_once(self, prompt: str) -> str:
        assert self.tokenizer is not None
        assert self.model is not None

        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        outputs = self.model.generate(
            **encoded,
            max_new_tokens=self.config.max_new_tokens,
            do_sample=self.config.temperature > 0,
            temperature=max(self.config.temperature, 1e-5),
        )
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return text.strip()

