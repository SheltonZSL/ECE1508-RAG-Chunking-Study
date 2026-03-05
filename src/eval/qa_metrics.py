from __future__ import annotations

import re
import string
from collections import Counter

_ARTICLES_RE = re.compile(r"\b(a|an|the)\b")
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = _ARTICLES_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def exact_match_score(prediction: str, ground_truth: str) -> float:
    return 1.0 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0.0


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens and not truth_tokens:
        return 1.0
    if not pred_tokens or not truth_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(truth_tokens)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision = same / len(pred_tokens)
    recall = same / len(truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def best_qa_scores(prediction: str, ground_truths: list[str]) -> tuple[float, float]:
    if not ground_truths:
        return 0.0, 0.0
    em = max(exact_match_score(prediction, truth) for truth in ground_truths)
    f1 = max(f1_score(prediction, truth) for truth in ground_truths)
    return em, f1


def compute_qa_aggregate(records: list[dict[str, float]]) -> dict[str, float]:
    if not records:
        return {"em": 0.0, "f1": 0.0}
    em = sum(item["em"] for item in records) / len(records)
    f1 = sum(item["f1"] for item in records) / len(records)
    return {"em": em, "f1": f1}

