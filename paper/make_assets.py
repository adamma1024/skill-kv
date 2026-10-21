"""Generate every number, table and figure in the paper from the raw outputs/*.jsonl.

  uv run --group paper python paper/make_assets.py

Writes paper/numbers.tex (macros used in the text), paper/tables/*.tex, paper/figures/*.pdf.
Nothing in the paper's results is typed by hand.
"""

import glob
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from skillkv.stats import boot_ci  # noqa: E402
from skillkv.synth import CONSTRAINT_TYPES, make_library  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "paper"
MODELS = [("Qwen/Qwen3-4B", "Qwen3-4B", "Four"), ("Qwen/Qwen3-8B", "Qwen3-8B", "Eight"),
          ("mlx-community/Llama-3.2-3B-Instruct", "Llama-3.2-3B", "Llama")]
# categorical slots of the validated reference palette (light), fixed order and fixed meaning
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5e0"
macros: dict[str, str] = {}


def load(pattern):
    return [json.loads(l) for f in sorted(glob.glob(str(ROOT / "outputs" / pattern))) for l in open(f) if l.strip()]


def m(name, value):
    macros[name] = value


def num(x, sign=False, digits=2):
    t = f"{abs(x):.{digits}f}"
    if x < 0 and float(t) != 0:
        return "$-$" + t
    return ("+" if sign else "") + t


def ci(items):
    lo, hi = boot_ci(items)
    return mean(v for _, v in items), lo, hi


def fmt_ci(items, signed=False):
    if not items:
        return "--"
    v, lo, hi = ci(items)
    return f"{num(v, signed)} [{num(lo)}, {num(hi)}]"


def paired(rows, a, b, key, field, cluster):
    by = defaultdict(dict)
    for r in rows:
        by[r["cond"]][key(r)] = r
    ks = sorted(set(by[a]) & set(by[b]))
    return [(cluster(by[a][k]), float(by[a][k][field]) - float(by[b][k][field])) for k in ks]


def upper95(n):
    """One-sided 95% upper bound on a rate with 0 successes in n trials (exact binomial)."""
    return 1 - 0.05 ** (1 / n)


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
        ax.spines[s].set_linewidth(0.6)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=7, width=0.6, length=2.5)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def write_table(name, lines):
    (OUT / "tables" / f"{name}.tex").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------- pilot (fidelity, latency)
def pilot():
    rows = []
    for name in ("Qwen--Qwen3-1.7B", "Qwen--Qwen3-4B"):
        rs = load(f"pilot_{name}.jsonl")
        text = {r["skill"]: r for r in rs if r["cond"] == "text"}
        sysr = {r["skill"]: r for r in rs if r["cond"] == "pack:sys+seam0"}
        full = {r["skill"]: r for r in rs if r["cond"] == "pack:full+seam0"}
        sp = [text[s]["ttft_s"] / sysr[s]["ttft_s"] for s in sysr]
        big = max(text.values(), key=lambda r: r["skill_tokens"])
        rows.append((name.split("--")[1], median(sp), min(sp), max(sp), big["skill_tokens"], big["ttft_s"],
                     sysr[big["skill"]]["ttft_s"], sysr[big["skill"]]["pack_mb"] * 1024 / big["skill_tokens"],
                     mean(r["kl_mean"] for r in sysr.values()), mean(r["kl_mean"] for r in full.values())))
    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"& TTFT speedup & \multicolumn{2}{c}{Longest skill (%s tok)} & & KL to text \\" % f"{rows[1][4]:,}".replace(",", "{,}"),
             r"\cmidrule(lr){3-4}",
             r"Model & median (range) & text & pack & KB/token & sys / exact prelude \\", r"\midrule"]
    for (label, med, lo, hi, tok, t_text, t_pack, kb, kl_sys, kl_full) in rows:
        lines.append(f"{label} & {med:.1f}$\\times$ ({lo:.1f}--{hi:.1f}) & {t_text:.2f}\\,s & {t_pack:.2f}\\,s "
                     f"& {kb:.0f} & {kl_sys:.3f} / {kl_full:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_table("pilot", lines)
    kls = [r[8] for r in rows]
    m("PilotKLRange", f"{min(kls):.3f}--{max(kls):.3f}")
    m("PilotSpeedupOne", f"{rows[0][1]:.1f}")
    m("PilotSpeedupFour", f"{rows[1][1]:.1f}")
    m("PilotKBFour", f"{rows[1][7]:.0f}")


# ---------------------------------------------------------------- W1 (greedy; 1.7B and 4B)
def w1():
    rows = load("w1_*.jsonl")
    cl = lambda r: (r["tag"], r["skill"])  # noqa: E731
    key = lambda r: (r["tag"], r["task_id"])  # noqa: E731
    for r in rows:
        r["sgrade"] = strict_grade(r)
        r["lenient"], r["all_pass"] = r["all_pass"], all(r["sgrade"].values())
    for mid, tag in (("Qwen/Qwen3-1.7B", "OneSeven"), ("Qwen/Qwen3-4B", "Four")):
        rs = [r for r in rows if r["model"] == mid]
        for c in ("C1", "C3"):
            sel = [r for r in rs if r["cond"] == c]
            cc = c.replace("1", "one").replace("3", "three")
            m(f"Wone{tag}{cc}", f"{mean(r['all_pass'] for r in sel):.2f}")
            m(f"Wone{tag}Score{cc}", f"{mean(sum(r['sgrade'].values()) / 5 for r in sel):.2f}")
        m(f"Wone{tag}D", fmt_ci(paired(rs, "C3", "C1", key, "all_pass", cl), signed=True))


# ---------------------------------------------------------------- W2 (skills, sampling)
W2_ORDER = [("C1", "C1 text"), ("C2", "C2 system prefix"), ("C3", "C3 pack"),
            ("C1M", "C1M text, all rules named"), ("C3M", "C3M pack, all rules named"),
            ("C3G", "C3G pack, conversation prelude")]
_LIBS: dict[str, dict] = {}


def strict_grade(r):
    """Per-rule grade with position: `code` must be the first non-empty line and `signoff` the last
    (the lenient grader in synth.grade also accepts answers that merely quote the rule)."""
    if r["tag"] not in _LIBS:
        seed, n = map(int, re.match(r"s(\d+)_n(\d+)", r["tag"]).groups())
        _LIBS[r["tag"]] = {s.name: {c.type: c.value for c in s.constraints} for s in make_library(n, seed, 400, 2400)}
    v = _LIBS[r["tag"]][r["skill"]]
    lines = [l.strip() for l in r["answer"].strip().splitlines() if l.strip()]
    g = dict(r["grade"])
    if not lines:
        g["code"] = g["signoff"] = False
        return g
    first = re.sub(r"^[`*#>\s]+", "", lines[0])
    g["code"] = re.match(r"code\s*:\s*" + re.escape(v["code"]), first, re.I) is not None
    g["signoff"] = re.search(r"--\s*" + re.escape(v["signoff"]) + r"\s*--", lines[-1], re.I) is not None
    return g


def w2():
    rows = load("w2_*.jsonl")
    for r in rows:
        r["sgrade"] = strict_grade(r)
        r["lenient"] = r["all_pass"]
        r["all_pass"] = all(r["sgrade"].values())  # primary: positional grading
        r["score"] = sum(r["sgrade"].values()) / len(CONSTRAINT_TYPES)
    cl = lambda r: (r["tag"], r["skill"])  # noqa: E731
    key = lambda r: (r["tag"], r["task_id"], r["seed"])  # noqa: E731
    present = [(mid, label, tag) for mid, label, tag in MODELS if any(r["model"] == mid for r in rows)]
    stats = {}
    for mid, label, tag in present:
        rs = [r for r in rows if r["model"] == mid]
        for c, _ in W2_ORDER:
            stats[mid, c] = ci([(cl(r), float(r["all_pass"])) for r in rs if r["cond"] == c])
        d = lambda a, b, field="all_pass": paired(rs, a, b, key, field, cl)  # noqa: E731
        m(f"Wtwo{tag}Cone", f"{stats[mid, 'C1'][0]:.2f}")
        m(f"Wtwo{tag}Cthree", f"{stats[mid, 'C3'][0]:.2f}")
        m(f"Wtwo{tag}Ctwo", f"{stats[mid, 'C2'][0]:.2f}")
        m(f"Wtwo{tag}ConeM", f"{stats[mid, 'C1M'][0]:.2f}")
        m(f"Wtwo{tag}CthreeG", f"{stats[mid, 'C3G'][0]:.2f}")
        m(f"Wtwo{tag}Dthree", fmt_ci(d("C3", "C1"), signed=True))
        m(f"Wtwo{tag}DthreeTwo", fmt_ci(d("C3", "C2"), signed=True))
        m(f"Wtwo{tag}DMM", fmt_ci(d("C3M", "C1M"), signed=True))
        m(f"Wtwo{tag}DG", fmt_ci(d("C3G", "C1"), signed=True))
        m(f"Wtwo{tag}DLenient", fmt_ci(d("C3", "C1", "lenient"), signed=True))
        m(f"Wtwo{tag}LenientCthree", f"{mean(r['lenient'] for r in rs if r['cond'] == 'C3'):.2f}")
        for c, _ in W2_ORDER:
            m(f"Wtwo{tag}Trunc{c.replace('1', 'one').replace('2', 'two').replace('3', 'three')}",
              f"{100 * mean(r['truncated'] for r in rs if r['cond'] == c):.0f}")
        for c in ("C1", "C3"):
            sel = [r for r in rs if r["cond"] == c]
            cc = c.replace("1", "one").replace("3", "three")
            m(f"Wtwo{tag}Code{cc}", f"{mean(r['sgrade']['code'] for r in sel):.2f}")
            m(f"Wtwo{tag}Sign{cc}", f"{mean(r['sgrade']['signoff'] for r in sel):.2f}")
            m(f"Wtwo{tag}Head{cc}", f"{mean(r['sgrade']['heading'] for r in sel):.2f}")
            m(f"Wtwo{tag}TTFT{cc}", f"{median(r['ttft_s'] for r in sel):.2f}")
        m(f"Wtwo{tag}Trunc", f"{100 * mean(r['truncated'] for r in rs):.0f}")
    m("WtwoNtasks", str(len({(r["tag"], r["task_id"]) for r in rows if r["model"] == MODELS[0][0]})))
    m("WtwoNskills", str(len({cl(r) for r in rows if r["model"] == MODELS[0][0]})))
    m("WtwoNmodels", str(len(present)))

    lines = [r"\begin{tabular}{l" + "c" * 5 + "}", r"\toprule",
             r"Model & \texttt{code} & \texttt{signoff} & \texttt{heading} & \texttt{filename} & \texttt{param} \\",
             r"\midrule"]
    for mid, label, _ in present:
        rs = [r for r in rows if r["model"] == mid]
        cells = []
        for t in ("code", "signoff", "heading", "filename", "param"):
            a_, b_ = (mean(r["sgrade"][t] for r in rs if r["cond"] == c) for c in ("C1", "C3"))
            cells.append(f"{a_:.2f} $\\rightarrow$ {b_:.2f}")
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_table("skillrules", lines)

    wcon = [("C3", "C1", "all_pass", r"C3 $-$ C1 (pack vs text)"),
            ("C3", "C1", "lenient", r"C3 $-$ C1, lenient grading"),
            ("C3", "C2", "all_pass", r"C3 $-$ C2 (pack vs system prefix)"),
            ("C3M", "C1M", "all_pass", r"C3M $-$ C1M (all rules named)"),
            ("C3G", "C1", "all_pass", r"C3G $-$ C1 (conversation-shaped prelude)")]
    lines = [r"\begin{tabular}{l" + "r" * len(present) + "}", r"\toprule",
             r"Contrast & " + " & ".join(label for _, label, _ in present) + r" \\", r"\midrule"]
    for a_, b_, field, label in wcon:
        cells = [fmt_ci(paired([r for r in rows if r["model"] == mid], a_, b_, key, field, cl), signed=True)
                 for mid, _, _ in present]
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_table("w2contrasts", lines)

    head = " & ".join(label for _, label, _ in present)
    lines = [r"\begin{tabular}{l" + "r" * len(present) + "}", r"\toprule",
             r"Condition & " + head + r" \\", r"\midrule"]
    for c, label in W2_ORDER:
        cells = [f"{stats[mid, c][0]:.2f} [{stats[mid, c][1]:.2f}, {stats[mid, c][2]:.2f}]" for mid, _, _ in present]
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_table("w2", lines)

    fig, ax = plt.subplots(figsize=(6.75, 1.95))
    style(ax)
    w = 0.8 / len(present)
    for j, (mid, label, _) in enumerate(present):
        xs = [i + (j - (len(present) - 1) / 2) * w for i in range(len(W2_ORDER))]
        vals = [stats[mid, c][0] for c, _ in W2_ORDER]
        errs = [[stats[mid, c][0] - stats[mid, c][1] for c, _ in W2_ORDER],
                [stats[mid, c][2] - stats[mid, c][0] for c, _ in W2_ORDER]]
        ax.bar(xs, vals, width=w, color=C[j], edgecolor="white", linewidth=1.0, label=label)
        ax.errorbar(xs, vals, yerr=errs, fmt="none", ecolor=INK, elinewidth=0.6, capsize=1.5)
        for x, v in zip(xs, vals):
            if v == 0:
                ax.plot([x - w / 2.4, x + w / 2.4], [0.006, 0.006], color=C[j], linewidth=1.6)
                ax.text(x, 0.02, "0", ha="center", va="bottom", fontsize=5.5, color=MUTED)
    ax.set_xticks(range(len(W2_ORDER)), [c for c, _ in W2_ORDER])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("All 5 rules followed", fontsize=7, color=INK)
    ax.legend(fontsize=6.5, frameon=False, loc="upper left", ncol=len(present))
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "figures" / "w2_skills.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- W3-W6 (injection)
def injection():
    w3 = load("w3_*.jsonl")
    agent = load("w5_*.jsonl") + [r for r in w3 if r["cond"] in ("T2", "T3G") and not r["defended"]]
    defended = [r for r in w3 if r["defended"]]
    rag = load("w6_*.jsonl")
    cl = lambda r: (r["tag"], r["doc"])  # noqa: E731
    key = lambda r: (r["tag"], r["item"], r["seed"])  # noqa: E731
    inj = lambda rows: [r for r in rows if r["injection"]]  # noqa: E731

    def sel(rows, mid, cond, injected=True, kind=None):
        return [r for r in rows if r["model"] == mid and r["cond"] == cond and bool(r["injection"]) == injected
                and (kind is None or r["injection"] == kind)]

    def rate(rows, mid, cond, field="complied", injected=True, kind=None):
        return mean(float(r[field]) for r in sel(rows, mid, cond, injected, kind))

    conds = ["T1", "T2", "T3", "T3G", "T1M"]
    lines = [r"\begin{tabular}{l" + "r" * (len(conds) + 2) + "}", r"\toprule",
             r"& \multicolumn{5}{c}{Plain system prompt} & \multicolumn{2}{c}{Defended prompt} \\",
             r"\cmidrule(lr){2-6}\cmidrule(lr){7-8}",
             r"Model & T1 & T2 & T3 & T3G & T1$'$ & T1 & T3 \\", r"\midrule"]
    for mid, label, _ in MODELS:
        cells = [f"{rate(agent, mid, c):.2f}" for c in conds] + [f"{rate(defended, mid, c):.2f}" for c in ("T1", "T3")]
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_table("agent_injection", lines)

    lines = [r"\begin{tabular}{lccc}", r"\toprule", r"Model & override & append & refuse \\", r"\midrule"]
    for mid, label, _ in MODELS:
        cells = [f"{rate(agent, mid, 'T1', kind=k):.2f} $\\rightarrow$ {rate(agent, mid, 'T3', kind=k):.2f}"
                 for k in ("override", "append", "refuse")]
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_table("pertype", lines)

    contrasts = [("T3", "T1", r"T3 $-$ T1 (pack vs text)"),
                 ("T3G", "T1", r"T3G $-$ T1 (conversation-shaped prelude)"),
                 ("T3", "T1M", r"T3 $-$ T1$'$ (position shift)"),
                 ("T3F16", "T3", r"T3F16 $-$ T3 (first 16 recomputed)"),
                 ("T3L16", "T3", r"T3L16 $-$ T3 (last 16 recomputed)"),
                 ("T3R15", "T3", r"T3R15 $-$ T3 (random 15\%)"),
                 ("T3B15", "T3", r"T3B15 $-$ T3 (deviation 15\%)"),
                 ("T3B15", "T3R15", r"T3B15 $-$ T3R15 (deviation vs random)"),
                 ("T3B15", "T1", r"T3B15 $-$ T1 (residual)"),
                 ("T2", "T1", r"T2 $-$ T1 (system-prompt placement)")]
    lines = [r"\begin{tabular}{lrrr}", r"\toprule", r"Contrast & Qwen3-4B & Qwen3-8B & Llama-3.2-3B \\", r"\midrule"]
    for a, b, label in contrasts:
        cells = [fmt_ci(paired(inj([r for r in agent if r["model"] == mid]), a, b, key, "complied", cl), signed=True)
                 for mid, _, _ in MODELS]
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    cells = [fmt_ci(paired(inj([r for r in defended if r["model"] == mid]), "T3", "T1", key, "complied", cl), signed=True)
             for mid, _, _ in MODELS]
    lines.append(r"T3 $-$ T1, defended prompt & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_table("mechanism", lines)

    rconds = ("R1", "R3", "R3N", "R3B15")
    lines = [r"\begin{tabular}{l" + "r" * 8 + "}", r"\toprule",
             r"& \multicolumn{4}{c}{Injection compliance} & \multicolumn{4}{c}{Clean accuracy} \\",
             r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
             r"Model & R1 & R3 & R3N & R3B15 & R1 & R3 & R3N & R3B15 \\", r"\midrule"]
    for mid, label, _ in MODELS:
        lines.append(f"{label} & " + " & ".join(f"{rate(rag, mid, c):.2f}" for c in rconds) + " & "
                     + " & ".join(f"{rate(rag, mid, c, 'correct', injected=False):.2f}" for c in rconds) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_table("rag", lines)

    for mid, _, n in MODELS:
        rs = inj([r for r in agent if r["model"] == mid])
        rr = inj([r for r in rag if r["model"] == mid])
        m(f"InjTone{n}", f"{rate(agent, mid, 'T1'):.2f}")
        m(f"InjTthree{n}", f"{rate(agent, mid, 'T3'):.2f}")
        m(f"InjD{n}", fmt_ci(paired(rs, "T3", "T1", key, "complied", cl), signed=True))
        m(f"InjDG{n}", fmt_ci(paired(rs, "T3G", "T1", key, "complied", cl), signed=True))
        m(f"InjDMask{n}", fmt_ci(paired(rs, "T3", "T1M", key, "complied", cl), signed=True))
        m(f"InjResid{n}", fmt_ci(paired(rs, "T3B15", "T1", key, "complied", cl), signed=True))
        m(f"InjOvTone{n}", f"{rate(agent, mid, 'T1', kind='override'):.2f}")
        m(f"InjOvTthree{n}", f"{rate(agent, mid, 'T3', kind='override'):.2f}")
        m(f"InjOvTthreeG{n}", f"{rate(agent, mid, 'T3G', kind='override'):.2f}")
        m(f"InjDefTone{n}", f"{rate(defended, mid, 'T1'):.2f}")
        m(f"InjDefTthree{n}", f"{rate(defended, mid, 'T3'):.2f}")
        m(f"RagD{n}", fmt_ci(paired(rr, "R3", "R1", key, "complied", cl), signed=True))
        m(f"RagDB{n}", fmt_ci(paired(rr, "R3B15", "R1", key, "complied", cl), signed=True))
        m(f"RagNinj{n}", str(len(sel(rag, mid, "R1"))))
    n_inj = len(sel(agent, "Qwen/Qwen3-8B", "T1"))
    n_ov = len(sel(agent, "Qwen/Qwen3-8B", "T1", kind="override"))
    m("InjNtrials", str(n_inj))
    m("InjNtrialsOverride", str(n_ov))
    m("InjZeroUpper", f"{100 * upper95(n_inj):.1f}")
    m("InjNinjected", str(len({(r["tag"], r["item"]) for r in inj(agent) if r["model"] == MODELS[0][0]})))
    m("InjNdocs", str(len({cl(r) for r in agent if r["model"] == MODELS[0][0]})))
    m("FracBlend", f"{mean(r['recompute_fraction'] for r in agent if r['cond'] == 'T3B15'):.2f}")
    m("FracRandom", f"{mean(r['recompute_fraction'] for r in agent if r['cond'] == 'T3R15'):.2f}")
    shares = [(rate(agent, mid, "T3") - rate(agent, mid, "T3B15")) / (rate(agent, mid, "T3") - rate(agent, mid, "T1"))
              for mid, _, _ in MODELS]
    m("MitigShareRange", f"{100 * min(shares):.0f}--{100 * max(shares):.0f}")
    m("CleanAccMin", f"{min(rate(agent, mid, c, 'correct', injected=False) for mid, _, _ in MODELS for c in ('T1', 'T3')):.2f}")
    m("RagRthreeAccFour", f"{rate(rag, 'Qwen/Qwen3-4B', 'R3', 'correct', injected=False):.2f}")
    m("RagRthreeNAccFour", f"{rate(rag, 'Qwen/Qwen3-4B', 'R3N', 'correct', injected=False):.2f}")
    m("RagRthreeFour", f"{rate(rag, 'Qwen/Qwen3-4B', 'R3'):.2f}")
    accs = sorted(rate(rag, mid, "R3N", "correct", injected=False) for mid, _, _ in MODELS)
    m("RagRthreeNAccRange", f"{accs[0]:.2f}--{accs[-1]:.2f}")

    roles = {"text": C[0], "pack": C[1], "recompute": C[2], "masked": C[3]}
    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.2), gridspec_kw={"width_ratios": [4, 3]}, sharey=True)
    panels = [(axes[0], agent, [("T1", "T1 text", "text"), ("T1M", "T1$'$ text, masked", "masked"),
                                ("T3", "T3 pack", "pack"), ("T3B15", "T3B15 pack + 15% recompute", "recompute")],
               "Agent layout: document after the question"),
              (axes[1], rag, [("R1", "R1 full prefill", "text"), ("R3", "R3 chunk packs", "pack"),
                              ("R3B15", "R3B15 + 15% recompute", "recompute")],
               "RAG layout: documents before the question")]
    for ax, rows, conds_, title in panels:
        style(ax)
        w = 0.8 / len(conds_)
        for j, (c, label, role) in enumerate(conds_):
            xs = [i + (j - (len(conds_) - 1) / 2) * w for i in range(len(MODELS))]
            stats = [ci([(cl(r), float(r["complied"])) for r in sel(rows, mid, c)]) for mid, _, _ in MODELS]
            vals = [s[0] for s in stats]
            ax.bar(xs, vals, width=w, color=roles[role], edgecolor="white", linewidth=1.0, label=label)
            ax.errorbar(xs, vals, yerr=[[s[0] - s[1] for s in stats], [s[2] - s[0] for s in stats]],
                        fmt="none", ecolor=INK, elinewidth=0.6, capsize=1.5)
            for x, v in zip(xs, vals):
                if v == 0:
                    ax.plot([x - w / 2.4, x + w / 2.4], [0.004, 0.004], color=roles[role], linewidth=1.6)
                    ax.text(x, 0.012, "0", ha="center", va="bottom", fontsize=5.5, color=MUTED)
        ax.set_xticks(range(len(MODELS)), [l for _, l, _ in MODELS])
        ax.set_title(title, fontsize=7.5, color=INK, loc="left")
        ax.legend(fontsize=6, frameon=False, loc="upper right", handlelength=1.2)
    axes[0].set_ylabel("Injection compliance", fontsize=7, color=INK)
    axes[0].set_ylim(0, 0.72)
    fig.tight_layout(pad=0.3, w_pad=1.0)
    fig.savefig(OUT / "figures" / "injection.pdf")
    plt.close(fig)


def main():
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    pilot()
    w1()
    w2()
    injection()
    (OUT / "numbers.tex").write_text(
        "% generated by paper/make_assets.py - do not edit\n"
        + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(macros.items())))
    print(f"{len(macros)} macros, tables: {sorted(p.name for p in (OUT / 'tables').iterdir())}")


if __name__ == "__main__":
    main()
