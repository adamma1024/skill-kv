"""Generate every number, table and figure in the paper from the raw outputs/*.jsonl.

  uv run --group paper python paper/make_assets.py

Writes paper/numbers.tex (macros used in the text), paper/tables/*.tex, paper/figures/*.pdf.
Nothing in the paper's results is typed by hand.
"""

import glob
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from skillkv.stats import boot_ci  # noqa: E402
from skillkv.synth import CONSTRAINT_TYPES  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "paper"
MODELS = [("Qwen/Qwen3-4B", "Qwen3-4B"), ("Qwen/Qwen3-8B", "Qwen3-8B"),
          ("mlx-community/Llama-3.2-3B-Instruct", "Llama-3.2-3B")]
# categorical slots 1-4 of the validated reference palette (light), fixed order
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5e0"

macros: dict[str, str] = {}


def load(pattern):
    return [json.loads(l) for f in sorted(glob.glob(str(ROOT / "outputs" / pattern))) for l in open(f) if l.strip()]


def m(name, value):
    macros[name] = value


def pct(x):
    return f"{100 * x:.0f}"


def ci(items):
    lo, hi = boot_ci(items)
    return mean(v for _, v in items), lo, hi


def fmt_ci(items, signed=False, digits=2):
    v, lo, hi = ci(items)
    def num(x, sign=False):
        t = f"{abs(x):.{digits}f}"
        if x < 0 and float(t) != 0:
            return "$-$" + t
        return ("+" if sign else "") + t
    return f"{num(v, signed)} [{num(lo)}, {num(hi)}]"


def paired(rows, a, b, key, field, cluster):
    by = defaultdict(dict)
    for r in rows:
        by[r["cond"]][key(r)] = r
    ks = sorted(set(by[a]) & set(by[b]))
    return [(cluster(by[a][k]), float(by[a][k][field]) - float(by[b][k][field])) for k in ks]


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
        ax.spines[s].set_linewidth(0.6)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=7, width=0.6, length=2.5)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------- pilot (speed)
def pilot():
    rows = []
    for name in ("Qwen--Qwen3-1.7B", "Qwen--Qwen3-4B"):
        rs = load(f"pilot_{name}.jsonl")
        text = {r["skill"]: r for r in rs if r["cond"] == "text"}
        sysr = {r["skill"]: r for r in rs if r["cond"] == "pack:sys+seam0"}
        full = {r["skill"]: r for r in rs if r["cond"] == "pack:full+seam0"}
        sp = [text[s]["ttft_s"] / sysr[s]["ttft_s"] for s in sysr]
        big = max(text.values(), key=lambda r: r["skill_tokens"])
        label = name.split("--")[1]
        rows.append((label, median(sp), min(sp), max(sp), big["skill_tokens"], big["ttft_s"],
                     sysr[big["skill"]]["ttft_s"], sysr[big["skill"]]["pack_mb"] * 1024 / big["skill_tokens"],
                     mean(r["kl_mean"] for r in sysr.values()), mean(r["kl_mean"] for r in full.values())))
    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"Model & TTFT speedup (median, range) & Text TTFT & Pack TTFT & KB/token & KL (sys / exact) \\",
             r"\midrule"]
    for (label, med, lo, hi, tok, t_text, t_pack, kb, kl_sys, kl_full) in rows:
        lines.append(f"{label} & {med:.1f}$\\times$ ({lo:.1f}--{hi:.1f}) & {t_text:.2f}\\,s & {t_pack:.2f}\\,s "
                     f"& {kb:.0f} & {kl_sys:.3f} / {kl_full:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tables" / "pilot.tex").write_text("\n".join(lines) + "\n")
    q4 = rows[1]
    m("PilotSpeedupOne", f"{rows[0][1]:.1f}")
    kls = [r[8] for r in rows]
    m("PilotKLRange", f"{min(kls):.2f}--{max(kls):.2f}")
    m("PilotSpeedupFour", f"{q4[1]:.1f}")
    m("PilotLongTok", f"{q4[4]:,}".replace(",", "{,}"))
    m("PilotLongText", f"{q4[5]:.2f}")
    m("PilotLongPack", f"{q4[6]:.2f}")
    m("PilotKBFour", f"{q4[7]:.0f}")


# ---------------------------------------------------------------- W2 (skills)
W2_ORDER = [("C1", "C1 text"), ("C2", "C2 sys-prefix"), ("C3", "C3 KV pack"),
            ("C1M", "C1M text, all named"), ("C3M", "C3M pack, all named"), ("C3G", "C3G pack, saw dialogue")]


def w2():
    rows = load("w2_*.jsonl")
    for r in rows:
        r["score"] = sum(r["grade"].values()) / len(CONSTRAINT_TYPES)
    cl = lambda r: (r["tag"], r["skill"])  # noqa: E731
    key = lambda r: (r["tag"], r["task_id"], r["seed"])  # noqa: E731
    by = defaultdict(list)
    for r in rows:
        by[r["cond"]].append(r)
    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"Condition & All-5 pass [95\% CI] & Constraint score & \texttt{code} & \texttt{signoff} & TTFT \\",
             r"\midrule"]
    stats = {}
    for c, label in W2_ORDER:
        rs = by[c]
        v, lo, hi = ci([(cl(r), float(r["all_pass"])) for r in rs])
        stats[c] = (v, lo, hi)
        lines.append(f"{label} & {v:.2f} [{lo:.2f}, {hi:.2f}] & {mean(r['score'] for r in rs):.2f} & "
                     f"{mean(r['grade']['code'] for r in rs):.2f} & {mean(r['grade']['signoff'] for r in rs):.2f} & "
                     f"{median(r['ttft_s'] for r in rs):.2f}\\,s \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tables" / "w2.tex").write_text("\n".join(lines) + "\n")

    d = lambda a, b: paired(rows, a, b, key, "all_pass", cl)  # noqa: E731
    m("WtwoNtasks", str(len({(r["tag"], r["task_id"]) for r in rows})))
    m("WtwoNskills", str(len({cl(r) for r in rows})))
    m("WtwoCone", f"{stats['C1'][0]:.2f}")
    m("WtwoCtwo", f"{stats['C2'][0]:.2f}")
    m("WtwoCthree", f"{stats['C3'][0]:.2f}")
    m("WtwoConeM", f"{stats['C1M'][0]:.2f}")
    m("WtwoCthreeM", f"{stats['C3M'][0]:.2f}")
    m("WtwoCthreeG", f"{stats['C3G'][0]:.2f}")
    m("WtwoDthree", fmt_ci(d("C3", "C1"), signed=True))
    m("WtwoDtwo", fmt_ci(d("C2", "C1"), signed=True))
    m("WtwoDthreeTwo", fmt_ci(d("C3", "C2"), signed=True))
    m("WtwoDMM", fmt_ci(d("C3M", "C1M"), signed=True))
    m("WtwoDG", fmt_ci(d("C3G", "C1"), signed=True))
    m("WtwoCodeCone", f"{mean(r['grade']['code'] for r in by['C1']):.2f}")
    m("WtwoCodeCthree", f"{mean(r['grade']['code'] for r in by['C3']):.2f}")
    m("WtwoTTFTCone", f"{median(r['ttft_s'] for r in by['C1']):.2f}")
    m("WtwoTTFTCthree", f"{median(r['ttft_s'] for r in by['C3']):.2f}")
    m("WtwoTrunc", pct(mean(r["truncated"] for r in rows)))

    fig, ax = plt.subplots(figsize=(3.3, 2.0))
    style(ax)
    xs = range(len(W2_ORDER))
    vals = [stats[c][0] for c, _ in W2_ORDER]
    errs = [[stats[c][0] - stats[c][1] for c, _ in W2_ORDER], [stats[c][2] - stats[c][0] for c, _ in W2_ORDER]]
    colors = [C[0] if c in ("C3", "C3M") else ("#9b9a93" if c in ("C1", "C1M", "C2") else C[1]) for c, _ in W2_ORDER]
    ax.bar(xs, vals, width=0.62, color=colors, edgecolor="white", linewidth=1.2)
    ax.errorbar(xs, vals, yerr=errs, fmt="none", ecolor=INK, elinewidth=0.7, capsize=2)
    for x, (c, _) in zip(xs, W2_ORDER):
        ax.text(x, stats[c][2] + 0.03, f"{stats[c][0]:.2f}", ha="center", va="bottom", fontsize=6.5, color=INK)
    ax.set_xticks(list(xs), [c for c, _ in W2_ORDER])
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("All 5 rules followed", fontsize=7, color=INK)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "figures" / "w2_skills.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- W3-W6 (injection)
def injection():
    agent = load("w5_*.jsonl") + [r for r in load("w3_*.jsonl") if r["cond"] == "T3G" and not r["defended"]]
    defended = [r for r in load("w3_*.jsonl") if r["defended"]]
    rag = load("w6_*.jsonl")
    cl = lambda r: (r["tag"], r["doc"])  # noqa: E731
    key = lambda r: (r["tag"], r["item"], r["seed"])  # noqa: E731
    inj = lambda rows: [r for r in rows if r["injection"]]  # noqa: E731

    def rate(rows, model, cond, field="complied", injected=True):
        rs = [r for r in rows if r["model"] == model and r["cond"] == cond and bool(r["injection"]) == injected]
        return mean(float(r[field]) for r in rs)

    # Table: agent layout
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"Model & T1 & T3 & T3G & T1$'$ & T3B15 & $\Delta$ T3$-$T1 [95\% CI] \\", r"\midrule"]
    for mid, label in MODELS:
        rs = [r for r in agent if r["model"] == mid]
        dd = fmt_ci(paired(inj(rs), "T3", "T1", key, "complied", cl), signed=True)
        lines.append(f"{label} & " + " & ".join(f"{rate(agent, mid, c):.2f}" for c in ("T1", "T3", "T3G", "T1M", "T3B15"))
                     + f" & {dd} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tables" / "agent_injection.tex").write_text("\n".join(lines) + "\n")

    # Table: mechanism + mitigation deltas
    lines = [r"\begin{tabular}{lrrr}", r"\toprule",
             r"Contrast & Qwen3-4B & Qwen3-8B & Llama-3.2-3B \\", r"\midrule"]
    for a, b, label in [("T3", "T1M", r"T3 $-$ T1$'$ (RoPE shift only)"),
                        ("T3B15", "T3", r"T3B15 $-$ T3 (15\% recompute)"),
                        ("T3B15", "T1", r"T3B15 $-$ T1 (residual)"),
                        ("T3F16", "T3", r"T3F16 $-$ T3 (first 16 recomputed)"),
                        ("T3L16", "T3", r"T3L16 $-$ T3 (last 16 recomputed)")]:
        cells = [fmt_ci(paired(inj([r for r in agent if r["model"] == mid]), a, b, key, "complied", cl), signed=True)
                 for mid, _ in MODELS]
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tables" / "mechanism.tex").write_text("\n".join(lines) + "\n")

    # Table: RAG layout
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"& \multicolumn{3}{c}{Injection compliance} & \multicolumn{3}{c}{Clean accuracy} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
             r"Model & R1 & R3 & R3B15 & R1 & R3 & R3B15 \\", r"\midrule"]
    for mid, label in MODELS:
        lines.append(f"{label} & " + " & ".join(f"{rate(rag, mid, c):.2f}" for c in ("R1", "R3", "R3B15")) + " & "
                     + " & ".join(f"{rate(rag, mid, c, 'correct', injected=False):.2f}" for c in ("R1", "R3", "R3B15"))
                     + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tables" / "rag.tex").write_text("\n".join(lines) + "\n")

    # macros
    names = {"Qwen/Qwen3-4B": "Four", "Qwen/Qwen3-8B": "Eight", "mlx-community/Llama-3.2-3B-Instruct": "Llama"}
    for mid, n in names.items():
        rs = inj([r for r in agent if r["model"] == mid])
        m(f"InjTone{n}", f"{rate(agent, mid, 'T1'):.2f}")
        m(f"InjTthree{n}", f"{rate(agent, mid, 'T3'):.2f}")
        m(f"InjD{n}", fmt_ci(paired(rs, "T3", "T1", key, "complied", cl), signed=True))
        m(f"InjResid{n}", fmt_ci(paired(rs, "T3B15", "T1", key, "complied", cl), signed=True))
        ov = lambda c: mean(r["complied"] for r in rs if r["cond"] == c and r["injection"] == "override")  # noqa: E731
        m(f"InjOvTone{n}", pct(ov("T1")))
        m(f"InjOvTthree{n}", pct(ov("T3")))
        rr = inj([r for r in rag if r["model"] == mid])
        m(f"RagD{n}", fmt_ci(paired(rr, "R3", "R1", key, "complied", cl), signed=True))
        m(f"RagDB{n}", fmt_ci(paired(rr, "R3B15", "R1", key, "complied", cl), signed=True))
        dr = inj([r for r in defended if r["model"] == mid])
        m(f"InjDefTthree{n}", f"{mean(r['complied'] for r in dr if r['cond'] == 'T3'):.2f}")
        m(f"CleanTthree{n}", f"{rate(agent, mid, 'T3', 'correct', injected=False):.2f}")
    shares = [(rate(agent, mid, "T3") - rate(agent, mid, "T3B15")) / (rate(agent, mid, "T3") - rate(agent, mid, "T1"))
              for mid, _ in MODELS]
    m("MitigShareRange", f"{100 * min(shares):.0f}--{100 * max(shares):.0f}")
    m("InjNitems", str(len({(r["tag"], r["item"]) for r in agent if r["model"] == MODELS[0][0]})))
    m("InjNdocs", str(len({cl(r) for r in agent if r["model"] == MODELS[0][0]})))

    # Figure: two panels, grouped by model
    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.15), gridspec_kw={"width_ratios": [4, 3]}, sharey=True)
    panels = [(axes[0], agent, [("T1", "T1 text"), ("T1M", "T1$'$ text, masked"), ("T3", "T3 KV pack"),
                                ("T3B15", "T3 + 15% recompute")], "Agent layout: document after the question"),
              (axes[1], rag, [("R1", "R1 full prefill"), ("R3", "R3 chunk packs"), ("R3B15", "R3 + 15% recompute")],
               "RAG layout: documents before the question")]
    for ax, rows, conds, title in panels:
        style(ax)
        w = 0.8 / len(conds)
        for j, (c, label) in enumerate(conds):
            xs = [i + (j - (len(conds) - 1) / 2) * w for i in range(len(MODELS))]
            vals = [rate(rows, mid, c) for mid, _ in MODELS]
            ax.bar(xs, vals, width=w, color=C[j], edgecolor="white", linewidth=1.0, label=label)
            for x, v in zip(xs, vals):
                ax.text(x, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=5.5, color=MUTED)
        ax.set_xticks(range(len(MODELS)), [l for _, l in MODELS])
        ax.set_title(title, fontsize=7.5, color=INK, loc="left")
        ax.legend(fontsize=6, frameon=False, loc="upper right", handlelength=1.2)
    axes[0].set_ylabel("Injection compliance", fontsize=7, color=INK)
    axes[0].set_ylim(0, 0.7)
    fig.tight_layout(pad=0.3, w_pad=1.0)
    fig.savefig(OUT / "figures" / "injection.pdf")
    plt.close(fig)


def selection():
    rows = load("selection_Qwen--Qwen3-4B_*.jsonl")
    by = defaultdict(list)
    for r in rows:
        by[r["n"]].append(r["correct"])
    m("SelAccFifty", f"{mean(by[50]):.2f}")
    m("SelAccHundred", f"{mean(by[100]):.2f}")


def main():
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    pilot()
    w2()
    injection()
    selection()
    (OUT / "numbers.tex").write_text(
        "% generated by paper/make_assets.py - do not edit\n"
        + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(macros.items())))
    print(f"{len(macros)} macros, tables: {sorted(p.name for p in (OUT / 'tables').iterdir())}")


if __name__ == "__main__":
    main()
