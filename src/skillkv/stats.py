"""Skill-clustered bootstrap: tasks of one skill share text and are strongly correlated, so
clusters (skills), not tasks, are resampled."""

import random
from collections import defaultdict
from statistics import mean


def boot_ci(items, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    """items: (cluster, value). 95% percentile interval of the mean."""
    if not items:
        return float("nan"), float("nan")
    groups = defaultdict(list)
    for c, v in items:
        groups[c].append(v)
    keys = list(groups)
    rng = random.Random(seed)
    ms = sorted(mean(v for k in rng.choices(keys, k=len(keys)) for v in groups[k]) for _ in range(n))
    return ms[int(0.025 * n)], ms[int(0.975 * n)]


def fmt(items) -> str:
    if not items:
        return "—"
    lo, hi = boot_ci(items)
    return f"{mean(v for _, v in items):.3f} [{lo:.2f}, {hi:.2f}]"
