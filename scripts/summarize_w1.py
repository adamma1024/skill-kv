"""Summarize W1 runs into markdown: success by condition, paired deltas vs C1, depth recall,
selection accuracy vs N.

  uv run python scripts/summarize_w1.py > outputs/w1_summary.md
"""

import glob
import json
from collections import defaultdict
from statistics import mean, median

from skillkv.stats import fmt
from skillkv.synth import CONSTRAINT_TYPES, DEPTHS

CONDS = ["C0", "C1", "C2", "C3"]


def cl(r):
    return (r["tag"], r["skill"])


def main() -> None:
    rows = [json.loads(l) for f in sorted(glob.glob("outputs/w1_*.jsonl")) for l in open(f) if l.strip()]
    if not rows:
        raise SystemExit("no outputs/w1_*.jsonl yet; run scripts/run_w1.py first")
    by_model = defaultdict(list)
    for r in rows:
        r["score"] = sum(r["grade"].values()) / len(CONSTRAINT_TYPES)
        by_model[r["model"]].append(r)

    for model, rs in by_model.items():
        tags = sorted({r["tag"] for r in rs})
        n_tasks = len({(r["tag"], r["task_id"]) for r in rs})
        print(f"## {model}  ({n_tasks} tasks, libraries {', '.join(tags)})\n")
        print("Rates with skill-clustered bootstrap 95% CI.\n")
        print("| cond | all-5 pass (oracle skill) | constraint score | end-to-end (selection × pass) | median TTFT s | median gen tok |")
        print("|---|---|---|---|---|---|")
        cell = {}
        for c in CONDS:
            cr = [r for r in rs if r["cond"] == c]
            if not cr:
                continue
            cell[c] = {(r["tag"], r["task_id"]): r for r in cr}
            e2e = [float(r["all_pass"] and (c == "C0" or r["sel_correct"])) for r in cr]
            print(f"| {c} | {fmt([(cl(r), float(r['all_pass'])) for r in cr])} | {fmt([(cl(r), r['score']) for r in cr])} | "
                  f"{mean(e2e):.3f} | {median(r['ttft_s'] for r in cr):.3f} | {median(r['gen_tokens'] for r in cr):.0f} |")
        print()
        if "C1" in cell:
            print("Paired difference vs C1 (same task), skill-clustered bootstrap 95% CI:\n")
            print("| cond − C1 | Δ all-5 pass | Δ constraint score | TTFT speedup vs C1 (median of per-task ratios) |")
            print("|---|---|---|---|")
            for c in ["C0", "C2", "C3"]:
                if c not in cell:
                    continue
                keys = sorted(set(cell[c]) & set(cell["C1"]))
                dp = [(cl(cell[c][k]), float(cell[c][k]["all_pass"]) - float(cell["C1"][k]["all_pass"])) for k in keys]
                ds = [(cl(cell[c][k]), cell[c][k]["score"] - cell["C1"][k]["score"]) for k in keys]
                sp = [cell["C1"][k]["ttft_s"] / cell[c][k]["ttft_s"] for k in keys]
                print(f"| {c} | {fmt(dp)} | {fmt(ds)} | {median(sp):.1f}× |")
            print()
        print("Constraint pass rate by planted depth (rows) and condition (columns):\n")
        print("| depth | " + " | ".join(c for c in CONDS if c in cell) + " |")
        print("|---|" + "---|" * len(cell))
        for d in DEPTHS:
            vals = []
            for c in CONDS:
                if c not in cell:
                    continue
                hits = [r["grade"][t] for r in cell[c].values() for t in CONSTRAINT_TYPES if r["depth"][t] == d]
                vals.append(f"{mean(hits):.2f}")
            print(f"| {d:.2f} | " + " | ".join(vals) + " |")
        print()
        print("Constraint pass rate by type:\n")
        print("| type | " + " | ".join(c for c in CONDS if c in cell) + " |")
        print("|---|" + "---|" * len(cell))
        for t in CONSTRAINT_TYPES:
            print(f"| {t} | " + " | ".join(f"{mean(r['grade'][t] for r in cell[c].values()):.2f}"
                                         for c in CONDS if c in cell) + " |")
        sel = [r for r in rs if r["cond"] == "C1"]
        if sel:
            print(f"\nSelection on the W1 libraries: lenient {mean(r['sel_correct'] for r in sel):.3f}, "
                  f"strict format {mean(r.get('sel_strict', r['sel_correct']) for r in sel):.3f}\n")

    srows = [json.loads(l) for f in sorted(glob.glob("outputs/selection_*.jsonl")) for l in open(f) if l.strip()]
    if srows:
        print("## Selection accuracy vs library size N\n")
        print("| model | N | catalog tokens | catalog prefill s | accuracy | strict format | median selection s |")
        print("|---|---|---|---|---|---|---|")
        groups = defaultdict(list)
        for r in srows:
            groups[(r["model"], r["n"])].append(r)
        for (m, n), g in sorted(groups.items()):
            print(f"| {m} | {n} | {g[0]['catalog_tokens']} | {g[0]['catalog_prefill_s']:.1f} | "
                  f"{mean(r['correct'] for r in g):.3f} | {mean(r['strict'] for r in g):.3f} | "
                  f"{median(r['sel_s'] for r in g):.2f} |")


if __name__ == "__main__":
    main()
