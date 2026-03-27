from __future__ import annotations

import random
from math import floor


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if q <= 0:
        return sorted_values[0]
    if q >= 1:
        return sorted_values[-1]
    idx = q * (len(sorted_values) - 1)
    low = floor(idx)
    high = min(low + 1, len(sorted_values) - 1)
    weight = idx - low
    return sorted_values[low] * (1 - weight) + sorted_values[high] * weight


def bootstrap_mean_ci(
    values: list[float],
    *,
    n_bootstrap: int = 300,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], values[0]

    rng = random.Random(seed)
    n = len(values)
    means: list[float] = []

    for _ in range(max(10, n_bootstrap)):
        sample_sum = 0.0
        for _ in range(n):
            sample_sum += values[rng.randrange(0, n)]
        means.append(sample_sum / n)

    means.sort()
    alpha = max(0.0, min(1.0, 1.0 - ci))
    low_q = alpha / 2.0
    high_q = 1.0 - alpha / 2.0
    return _percentile(means, low_q), _percentile(means, high_q)

