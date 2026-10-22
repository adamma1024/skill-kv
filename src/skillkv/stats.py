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


def boot_ci2(items, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    """items: (cluster_a, cluster_b, value) for two crossed factors (e.g. attack x user case).
    Pigeonhole bootstrap (Owen, 2007): resample levels of each factor independently and weight
    each observation by the product of its levels' multiplicities. 95% percentile interval."""
    if not items:
        return float("nan"), float("nan")
    a_lv = sorted({a for a, _, _ in items}, key=str)
    b_lv = sorted({b for _, b, _ in items}, key=str)
    rng = random.Random(seed)
    ms = []
    for _ in range(n):
        wa, wb = defaultdict(int), defaultdict(int)
        for k in rng.choices(a_lv, k=len(a_lv)):
            wa[k] += 1
        for k in rng.choices(b_lv, k=len(b_lv)):
            wb[k] += 1
        num = den = 0.0
        for a, b, v in items:
            w = wa[a] * wb[b]
            num += w * v
            den += w
        if den:
            ms.append(num / den)
    ms.sort()
    return ms[int(0.025 * len(ms))], ms[int(0.975 * len(ms))]


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
