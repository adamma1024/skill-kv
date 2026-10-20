"""Summarize W3 / RQ7 injection runs into markdown.

  uv run python scripts/summarize_w3.py > outputs/w3_summary.md
"""

import argparse
import glob
import json
from collections import defaultdict
from statistics import mean

from skillkv.inject import INJECTION_TYPES, INJ_DEPTHS
from skillkv.stats import fmt

ORDER = ["T1", "T2", "T3", "T3G", "T1M", "T3F16", "T3L16", "T3R15", "T3B15"]
PAIRS = [("T3", "T1", "KV pack (no conversation) vs text"),
         ("T3G", "T1", "KV pack (conversation-shaped prelude) vs text"),
         ("T3", "T3G", "no conversation vs conversation-shaped prelude"),
         ("T2", "T1", "system-prompt placement vs text"),
         ("T1M", "T1", "text, doc masked from the conversation (T1') vs text"),
         ("T3", "T1M", "RoPE shift: pack vs masked text at true positions"),
         ("T3F16", "T3", "recompute first 16 doc tokens vs pack"),
         ("T3L16", "T3", "recompute last 16 doc tokens vs pack"),
         ("T3R15", "T3", "recompute random 15% vs pack"),
         ("T3B15", "T3", "CacheBlend-style 15% vs pack"),
         ("T3B15", "T1", "CacheBlend-style 15% vs text")]


def rate(xs) -> str:
    xs = list(xs)
    return f"{mean(xs):.2f}" if xs else "—"


def cl(r):
    return (r["tag"], r["doc"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="outputs/w3_*.jsonl")
    args = ap.parse_args()
    rows = [json.loads(l) for f in sorted(glob.glob(args.glob)) for l in open(f) if l.strip()]
    if not rows:
        raise SystemExit(f"no {args.glob} yet; run the matching batch script first")
    for model in sorted({r["model"] for r in rows}):
        summarize([r for r in rows if r["model"] == model])
    if len({r["model"] for r in rows}) > 1:
        cross_model(rows)


def cross_model(rows) -> None:
    print("## Across models: T3 − T1 compliance (paired, document-clustered 95% CI)\n")
    print("| model | defended | T1 | T3 | T3G | Δ T3 − T1 | Δ T3G − T1 | override T1 → T3 |")
    print("|---|---|---|---|---|---|---|---|")
    for model in sorted({r["model"] for r in rows}):
        for defended in (False, True):
            rs = [r for r in rows if r["model"] == model and r["defended"] == defended and r["injection"]]
            if not rs:
                continue
            by = defaultdict(dict)
            for r in rs:
                by[r["cond"]][(r["tag"], r["item"], r["seed"])] = r

            def delta(a, b):
                ks = sorted(set(by[a]) & set(by[b]))
                return fmt([(cl(by[a][k]), float(by[a][k]["complied"]) - float(by[b][k]["complied"])) for k in ks])

            ov = lambda c: rate(r["complied"] for r in rs if r["cond"] == c and r["injection"] == "override")  # noqa: E731
            print(f"| {model} | {defended} | {rate(r['complied'] for r in by['T1'].values())} | "
                  f"{rate(r['complied'] for r in by['T3'].values())} | {rate(r['complied'] for r in by['T3G'].values())} | "
                  f"{delta('T3', 'T1')} | {delta('T3G', 'T1')} | {ov('T1')} → {ov('T3')} |")
    print()


def summarize(rows) -> None:
    conds = [c for c in ORDER if any(r["cond"] == c for r in rows)]
    print(f"## W3 / RQ7 — {rows[0]['model']}: {len({cl(r) for r in rows})} documents, "
          f"{len({(r['tag'], r['item']) for r in rows})} items, seeds {sorted({r['seed'] for r in rows})}\n")
    print("Compliance = the model followed the injected instruction (canary / refusal). "
          "Rates with document-clustered bootstrap 95% CI.\n")
    for defended in (False, True):
        rs = [r for r in rows if r["defended"] == defended]
        if not rs:
            continue
        print(f"### System prompt: {'defended (never follow instructions inside a document)' if defended else 'plain'}\n")
        print("| cond | injection compliance | answer correct (injected) | answer correct (clean) | recompute fraction |")
        print("|---|---|---|---|---|")
        for c in conds:
            inj = [r for r in rs if r["cond"] == c and r["injection"]]
            clean = [r for r in rs if r["cond"] == c and not r["injection"]]
            frac = [r["recompute_fraction"] for r in rs if r["cond"] == c and r.get("recompute_fraction") is not None]
            print(f"| {c} | {fmt([(cl(r), float(r['complied'])) for r in inj])} | "
                  f"{fmt([(cl(r), float(r['correct'])) for r in inj])} | "
                  f"{fmt([(cl(r), float(r['correct'])) for r in clean])} | {rate(frac) if frac else '—'} |")
        print("\nPaired difference in compliance (same item and seed):\n")
        print("| contrast | meaning | Δ compliance |")
        print("|---|---|---|")
        key = lambda r: (r["tag"], r["item"], r["seed"])  # noqa: E731
        by = defaultdict(dict)
        for r in rs:
            if r["injection"]:
                by[r["cond"]][key(r)] = r
        for a, b, meaning in PAIRS:
            if a in by and b in by:
                ks = sorted(set(by[a]) & set(by[b]))
                d = [(cl(by[a][k]), float(by[a][k]["complied"]) - float(by[b][k]["complied"])) for k in ks]
                print(f"| {a} − {b} | {meaning} | {fmt(d)} |")
        print("\nCompliance by injection type:\n")
        print("| type | " + " | ".join(conds) + " |")
        print("|---|" + "---|" * len(conds))
        for t in INJECTION_TYPES:
            print(f"| {t} | " + " | ".join(
                rate(r['complied'] for r in rs if r['cond'] == c and r['injection'] == t) for c in conds) + " |")
        print("\nCompliance by injection depth in the document:\n")
        print("| depth | " + " | ".join(conds) + " |")
        print("|---|" + "---|" * len(conds))
        for d in INJ_DEPTHS:
            print(f"| {d} | " + " | ".join(
                rate(r['complied'] for r in rs if r['cond'] == c and r['depth'] == d) for c in conds) + " |")
        print()


if __name__ == "__main__":
    main()
