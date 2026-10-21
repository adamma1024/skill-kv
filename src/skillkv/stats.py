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


def perm_test(items, n: int = 10000, seed: int = 0) -> float:
    """Two-sided cluster sign-flip permutation p-value for H0: mean paired difference = 0.
    items: (cluster, difference). Whole clusters flip sign together (exchangeable under H0
    when the paired differences are symmetric about zero)."""
    groups = defaultdict(list)
    for c, v in items:
        groups[c].append(v)
    sums = [sum(v) for v in groups.values()]
    total = sum(len(v) for v in groups.values())
    obs = abs(sum(sums)) / total
    rng = random.Random(seed)
    hits = sum(abs(sum(s if rng.random() < 0.5 else -s for s in sums)) / total >= obs - 1e-12 for _ in range(n))
    return (hits + 1) / (n + 1)


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (same order as input)."""
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adj, running = [0.0] * len(pvals), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = running
    return adj
