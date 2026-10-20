"""Summarize W2 (sampling, max 512 tokens, ablations) into markdown.

  uv run python scripts/summarize_w2.py > outputs/w2_summary.md
"""

import glob
import json
from collections import defaultdict
from statistics import mean, median

from skillkv.stats import fmt
from skillkv.synth import CONSTRAINT_TYPES

ORDER = ["C1", "C2", "C3", "C1M", "C3M", "C3G"]
PAIRS = [("C3", "C1", "approximate splice vs exact text"),
         ("C2", "C1", "exact sys-prefix vs exact text"),
         ("C3", "C2", "approximate splice vs exact sys-prefix"),
         ("C1M", "C1", "naming every item in the question (text)"),
         ("C3M", "C1M", "splice vs text when every item is named"),
         ("C3G", "C1", "pack saw a content-free question"),
         ("C3", "C3G", "no question at all vs content-free question")]


def cl(r):
    return (r["tag"], r["skill"])


def main() -> None:
    rows = [json.loads(l) for f in sorted(glob.glob("outputs/w2_*.jsonl")) for l in open(f) if l.strip()]
    if not rows:
        raise SystemExit("no outputs/w2_*.jsonl yet; run scripts/w2_all.sh first")
    for r in rows:
        r["score"] = sum(r["grade"].values()) / len(CONSTRAINT_TYPES)
    by = defaultdict(dict)  # cond -> (tag, task, seed) -> row
    for r in rows:
        by[r["cond"]][(r["tag"], r["task_id"], r["seed"])] = r
    conds = [c for c in ORDER if c in by]
    r0 = rows[0]
    n_skills = len({cl(r) for r in rows})
    n_tasks = len({(r["tag"], r["task_id"]) for r in rows})
    seeds = sorted({r["seed"] for r in rows})
    print(f"## W2 — {r0['model']}: {n_skills} skills, {n_tasks} tasks × seeds {seeds}, "
          f"T={r0['temp']} top-p={r0['top_p']} top-k={r0['top_k']}, max_tokens={r0['max_tokens']}\n")
    print("Rates with skill-clustered bootstrap 95% CI.\n")
    print("| cond | n | all-5 pass | constraint score | truncated | all-5 pass, untruncated only | median gen tok | median TTFT s |")
    print("|---|---|---|---|---|---|---|---|")
    for c in conds:
        rs = list(by[c].values())
        ok = [r for r in rs if not r["truncated"]]
        print(f"| {c} | {len(rs)} | {fmt([(cl(r), float(r['all_pass'])) for r in rs])} | "
              f"{fmt([(cl(r), r['score']) for r in rs])} | {mean(r['truncated'] for r in rs):.2f} | "
              f"{fmt([(cl(r), float(r['all_pass'])) for r in ok])} | "
              f"{median(r['gen_tokens'] for r in rs):.0f} | {median(r['ttft_s'] for r in rs):.3f} |")
    print("\nPaired differences (same task and sampling seed), skill-clustered bootstrap 95% CI:\n")
    print("| contrast | meaning | n pairs | Δ all-5 pass | Δ constraint score | Δ all-5 pass, both untruncated |")
    print("|---|---|---|---|---|---|")
    for a, b, meaning in PAIRS:
        if a not in by or b not in by:
            continue
        keys = sorted(set(by[a]) & set(by[b]))
        d = [(cl(by[a][k]), float(by[a][k]["all_pass"]) - float(by[b][k]["all_pass"])) for k in keys]
        ds = [(cl(by[a][k]), by[a][k]["score"] - by[b][k]["score"]) for k in keys]
        du = [(cl(by[a][k]), float(by[a][k]["all_pass"]) - float(by[b][k]["all_pass"])) for k in keys
              if not by[a][k]["truncated"] and not by[b][k]["truncated"]]
        print(f"| {a} − {b} | {meaning} | {len(keys)} | {fmt(d)} | {fmt(ds)} | {fmt(du)} |")
    if all(c in by for c in ("C1", "C3", "C1M")):
        gap = mean(r["all_pass"] for r in by["C3"].values()) - mean(r["all_pass"] for r in by["C1"].values())
        closed = mean(r["all_pass"] for r in by["C1M"].values()) - mean(r["all_pass"] for r in by["C1"].values())
        if gap > 0:
            print(f"\nShare of the C3−C1 gap closed by naming every item (C1M): {closed / gap:.0%}")
    print("\nConstraint pass rate by type:\n")
    print("| type | " + " | ".join(conds) + " |")
    print("|---|" + "---|" * len(conds))
    for t in CONSTRAINT_TYPES:
        print(f"| {t} | " + " | ".join(f"{mean(r['grade'][t] for r in by[c].values()):.2f}" for c in conds) + " |")


if __name__ == "__main__":
    main()
