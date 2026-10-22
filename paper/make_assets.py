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

from skillkv.stats import boot_ci, boot_ci2, holm, perm_test  # noqa: E402
from skillkv.synth import CONSTRAINT_TYPES, make_library  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "paper"
MODELS = [("Qwen/Qwen3-4B", "Qwen3-4B", "Four"), ("Qwen/Qwen3-8B", "Qwen3-8B", "Eight"),
          ("mlx-community/Llama-3.2-3B-Instruct", "Llama-3.2-3B", "Llama")]
# categorical slots of the validated reference palette (light), fixed order and fixed meaning
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5e0"
macros: dict[str, str] = {}


def load(pattern):
    return [json.loads(l) for f in sorted(glob.glob(str(ROOT / "outputs" / pattern))) for l in open(f) if l.strip()]


def m(name, value):
    if not name.isalpha():
        raise ValueError(f"LaTeX macro names must be letters only: {name!r}")
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
    agent = load("w5_*.jsonl") + load("w12_*.jsonl") + [r for r in w3 if r["cond"] in ("T2", "T3G") and not r["defended"]]
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
                 ("T3", "T1M", r"T3 $-$ T1$'$ (pack vs.\ masked text)"),
                 ("T3F16", "T3", r"T3F16 $-$ T3 (first 16 recomputed)"),
                 ("T3L16", "T3", r"T3L16 $-$ T3 (last 16 recomputed)"),
                 ("T3R15", "T3", r"T3R15 $-$ T3 (random 15\%)"),
                 ("T3B15", "T3", r"T3B15 $-$ T3 (deviation 15\%)"),
                 ("T3R15F", "T3", r"T3R15F $-$ T3 (random 15\% + 2 full layers)"),
                 ("T3B15", "T3R15F", r"T3B15 $-$ T3R15F (deviation vs random, equal compute)"),
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
        m(f"InjDevRand{n}", fmt_ci(paired(rs, "T3B15", "T3R15F", key, "complied", cl), signed=True))
        m(f"InjRandF{n}", fmt_ci(paired(rs, "T3R15F", "T3", key, "complied", cl), signed=True))
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
    m("FracRandomF", f"{100 * mean(r['recompute_fraction'] for r in agent if r['cond'] == 'T3R15F'):.0f}")
    shares = [(rate(agent, mid, "T3") - rate(agent, mid, "T3B15")) / (rate(agent, mid, "T3") - rate(agent, mid, "T1"))
              for mid, _, _ in MODELS]
    m("MitigShareRange", f"{100 * min(shares):.0f}--{100 * max(shares):.0f}")
    m("CleanAccMin", f"{min(rate(agent, mid, c, 'correct', injected=False) for mid, _, _ in MODELS for c in ('T1', 'T3')):.2f}")
    m("RagRthreeAccFour", f"{rate(rag, 'Qwen/Qwen3-4B', 'R3', 'correct', injected=False):.2f}")
    m("RagRthreeNAccFour", f"{rate(rag, 'Qwen/Qwen3-4B', 'R3N', 'correct', injected=False):.2f}")
    m("RagRthreeFour", f"{rate(rag, 'Qwen/Qwen3-4B', 'R3'):.2f}")
    accs = sorted(rate(rag, mid, "R3N", "correct", injected=False) for mid, _, _ in MODELS)
    m("RagRthreeNAccRange", f"{accs[0]:.2f}--{accs[-1]:.2f}")


# ---------------------------------------------------------------- round 1: scale, natural docs, recompute sweep
ALL_MODELS = MODELS + [("mlx-community/Qwen3-14B-4bit", "Qwen3-14B (4-bit)", "Fourteen"),
                       ("mlx-community/Qwen3-32B-4bit", "Qwen3-32B (4-bit)", "ThirtyTwo"),
                       ("mlx-community/Llama-3.1-8B-Instruct", "Llama-3.1-8B", "LlamaEight"),
                       ("mlx-community/Qwen3-8B-4bit", "Qwen3-8B (4-bit)", "EightQ")]


def complete(rs, cond, ref="T1"):
    """Only report a condition once it has as many rows as the reference condition."""
    n = sum(r["cond"] == cond for r in rs)
    return n > 0 and n == sum(r["cond"] == ref for r in rs)


def scale_natural():
    syn = [r for r in load("w5_*.jsonl") + load("w8_*.jsonl") if not r["defended"]]
    nat = load("w7_*.jsonl")
    cl = lambda r: (r["tag"], r["doc"])  # noqa: E731
    key = lambda r: (r["tag"], r["item"], r["seed"])  # noqa: E731

    def inj(rows, mid):
        return [r for r in rows if r["model"] == mid and r["injection"]]

    def rate(rows, cond, field="complied"):
        xs = [float(r[field]) for r in rows if r["cond"] == cond]
        return mean(xs) if xs else None

    # main table: both document types, five models
    lines = [r"\begin{tabular}{l" + "rrrr" * 2 + "}", r"\toprule",
             r"& \multicolumn{4}{c}{Synthetic procedure documents} & \multicolumn{4}{c}{Natural documents (Wikipedia)} \\",
             r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
             r"Model & T1 & T1$'$ & T3 & $\Delta$ T3$-$T1 & T1 & T1$'$ & T3 & $\Delta$ T3$-$T1 \\", r"\midrule"]
    for mid, label, tag in ALL_MODELS:
        if not inj(syn, mid) and not inj(nat, mid):
            continue
        cells = []
        for rows, kind in ((syn, "Syn"), (nat, "Nat")):
            rs = inj(rows, mid)
            if not (rs and complete(rs, "T3") and complete(rs, "T1M")):
                cells += ["--"] * 4
                continue
            cells += [f"{rate(rs, c):.2f}" for c in ("T1", "T1M", "T3")]
            d = fmt_ci(paired(rs, "T3", "T1", key, "complied", cl), signed=True)
            cells.append(d)
            m(f"Scale{kind}D{tag}", d)
            m(f"Scale{kind}Tone{tag}", f"{rate(rs, 'T1'):.2f}")
            m(f"Scale{kind}Tthree{tag}", f"{rate(rs, 'T3'):.2f}")
            m(f"Scale{kind}DMask{tag}", fmt_ci(paired(rs, "T3", "T1M", key, "complied", cl), signed=True))
            m(f"Scale{kind}DMaskText{tag}", fmt_ci(paired(rs, "T1M", "T1", key, "complied", cl), signed=True))
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_table("scale", lines)

    # natural-document accuracy cost (clean items)
    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule", r"Model & T1 & T1$'$ & T3 & T3B15 & $\Delta$ T3$-$T1 \\", r"\midrule"]
    n_drop = n_acc = 0
    for mid, label, tag in ALL_MODELS:
        rs = [r for r in nat if r["model"] == mid and not r["injection"]]
        if not rs or not complete(rs, "T3"):
            continue
        cells = [(f"{rate(rs, c, 'correct'):.2f}" if complete(rs, c) else "--") for c in ("T1", "T1M", "T3", "T3B15")]
        d = paired(rs, "T3", "T1", key, "correct", cl)
        cells.append(fmt_ci(d, True))
        m(f"NatAccD{tag}", fmt_ci(d, True))
        n_acc += 1
        n_drop += ci(d)[2] < 0
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
        m(f"NatAccTone{tag}", f"{rate(rs, 'T1', 'correct'):.2f}")
        m(f"NatAccTthree{tag}", f"{rate(rs, 'T3', 'correct'):.2f}")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_table("natural_accuracy", lines)
    m("NatAccNdrop", str(n_drop))
    m("NatAccNmodels", str(n_acc))
    m("NatNdocs", str(len({r["doc"] for r in nat})))
    m("NatNinjected", str(len({(r["tag"], r["item"]) for r in nat if r["injection"]
                                and r["model"] == MODELS[0][0]})))

    # recompute dose-response (synthetic documents)
    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    style(ax)
    sweep = [("T3", 0.0), ("T3B15", None), ("T3B30", None), ("T3B50", None)]
    fr_of = lambda c: 100 * mean(r["recompute_fraction"] for r in syn if r["cond"] == c and r.get("recompute_fraction") is not None)  # noqa: E731
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Model & T3 (0\%) & " + " & ".join(f"{c} ({fr_of(c):.0f}\\%)" for c in ("T3B15", "T3B30", "T3B50")) + r" \\",
             r"\midrule"]
    j = 0
    for mid, label, tag in ALL_MODELS:
        rs = inj(syn, mid)
        avail = [c for c, _ in sweep if complete(rs, c)]
        if len(avail) < 2:
            continue
        xs, ys, los, his, cells = [], [], [], [], []
        for c, _ in sweep:
            if c not in avail:
                cells.append("--")
                continue
            fr = [r["recompute_fraction"] for r in rs if r["cond"] == c and r.get("recompute_fraction") is not None]
            x = mean(fr) if fr else 0.0
            v, lo, hi = ci(paired(rs, c, "T1", key, "complied", cl))
            xs.append(x); ys.append(v); los.append(v - lo); his.append(hi - v)
            cells.append(f"{num(v, True)} [{num(lo)}, {num(hi)}]")
            m(f"Sweep{'Bzero' if c == 'T3' else c.replace('T3B', 'B').replace('15', 'fifteen').replace('30', 'thirty').replace('50', 'fifty')}{tag}", cells[-1])
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
        k = [t for _, _, t in ALL_MODELS].index(tag)  # colour and marker follow the model, not its rank
        ax.errorbar(xs, ys, yerr=[los, his], color=C[k], marker=MARKERS[k], markersize=3.5, linewidth=1.5,
                    elinewidth=0.6, capsize=1.5, label=label)
        j += 1
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_table("sweep", lines)
    ax.axhline(0, color=MUTED, linewidth=0.6)
    ax.set_xlabel("Fraction of prefill compute spent on recompute", fontsize=7, color=INK)
    ax.set_ylabel("Excess compliance over text (T3x$-$T1)", fontsize=7, color=INK)
    ax.legend(fontsize=5.5, frameon=False)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "figures" / "sweep.pdf")
    plt.close(fig)



def ia_rows():
    """InjecAgent rows at the 256-token budget, complete (model, seed) runs only, annotated with the
    attack instruction and the user case (the two crossed factors)."""
    from skillkv.injecagent import load_cases
    cases = {(c.kind, c.index): c for c in load_cases(ROOT / "data" / "injecagent", 4, "enhanced")}
    n_full = len(cases) * 4 + len(range(0, len(cases), 4)) * 2  # injected x 4 conds + clean x 2 conds
    rows = load("ia256_enhanced_*.jsonl")
    count = defaultdict(int)
    for r in rows:
        count[r["model"], r["seed"]] += 1
    rows = [r for r in rows if count[r["model"], r["seed"]] == n_full]
    for r in rows:
        c = cases[r["kind"], r["index"]]
        r["attack"], r["user"], r["complied"] = c.attacker_instruction, c.user_instruction, r["attack_success"]
    return rows, cases


IA_KEY = lambda r: (r["kind"], r["index"], r["seed"])  # noqa: E731
IA_CL = lambda r: r["attack"]  # noqa: E731


def fmt_ci2(rows, a, b, field="complied"):
    """Paired a-b difference with a two-way (attack x user case) pigeonhole bootstrap CI."""
    by = defaultdict(dict)
    for r in rows:
        by[r["cond"]][IA_KEY(r)] = r
    ks = sorted(set(by[a]) & set(by[b]))
    if not ks:
        return "--"
    items = [(by[a][k]["attack"], by[a][k]["user"], float(by[a][k][field]) - float(by[b][k][field])) for k in ks]
    lo, hi = boot_ci2(items)
    return f"{num(mean(v for *_, v in items), True)} [{num(lo)}, {num(hi)}]"


def injecagent():
    rows, cases = ia_rows()
    if not rows:
        return
    lines = [r"\begin{tabular}{lrrrrrrr}", r"\toprule",
             r"& \multicolumn{4}{c}{ASR (stage 1)} & & & \\",
             r"\cmidrule(lr){2-5}",
             r"Model & T1 & T1$'$ & T3 & T3B15 & $\Delta$ T3$-$T1 [two-way CI] & DH / DS (T1 $\rightarrow$ T3) & clean T1 / T3 \\",
             r"\midrule"]
    for mid, label, tag in ALL_MODELS:
        rs = [r for r in rows if r["model"] == mid and r["variant"] == "injected"]
        if not rs:
            continue
        clean = [r for r in rows if r["model"] == mid and r["variant"] == "clean"]
        rate = lambda c, src=rs: mean(float(r["attack_success"]) for r in src if r["cond"] == c)  # noqa: E731
        d = fmt_ci2(rs, "T3", "T1")
        split = []
        for kind in ("dh", "ds"):
            ks = [r for r in rs if r["kind"] == kind]
            k_rate = lambda c: mean(float(r["attack_success"]) for r in ks if r["cond"] == c)  # noqa: E731
            split.append(f"{k_rate('T1'):.2f}$\\rightarrow${k_rate('T3'):.2f}")
            m(f"Ia{kind.upper()}{tag}", f"{k_rate('T1'):.2f} $\\rightarrow$ {k_rate('T3'):.2f}")
        lines.append(f"{label} & " + " & ".join(f"{rate(c):.2f}" for c in ("T1", "T1M", "T3", "T3B15"))
                     + f" & {d} & {' / '.join(split)} & {rate('T1', clean):.2f} / {rate('T3', clean):.2f} \\\\")
        m(f"IaTone{tag}", f"{rate('T1'):.2f}")
        m(f"IaTthree{tag}", f"{rate('T3'):.2f}")
        m(f"IaD{tag}", d)
        m(f"IaDOneWay{tag}", fmt_ci(paired(rs, "T3", "T1", IA_KEY, "attack_success", IA_CL), signed=True))
        m(f"IaDMask{tag}", fmt_ci2(rs, "T3", "T1M"))
        m(f"IaResid{tag}", fmt_ci2(rs, "T3B15", "T1"))
        for c in ("T1", "T3"):
            sel = [r for r in rs if r["cond"] == c]
            cc = c.replace("1", "one").replace("3", "three")
            m(f"IaTrunc{cc}{tag}", f"{100 * mean(r['truncated'] for r in sel):.1f}")
            m(f"IaNoTool{cc}{tag}", f"{mean(not r['tools_called'] for r in sel):.2f}")
        m(f"IaTrunc{tag}", f"{100 * mean(r['truncated'] for r in rs):.1f}")
        m(f"IaSeeds{tag}", str(len({r["seed"] for r in rs})))
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_table("injecagent", lines)
    all_inj = [r for r in rows if r["variant"] == "injected"]
    m("IaTruncMaxMain", f"{100 * max(mean(r['truncated'] for r in all_inj if r['model'] == mid and r['cond'] == c) for mid in {r['model'] for r in all_inj} for c in ('T1', 'T3')):.1f}")
    cells = defaultdict(list)
    for r in rows:
        cells[r["model"], r["variant"], r["cond"]].append(r["truncated"])
    m("IaTruncMax", f"{100 * max(mean(v) for v in cells.values()):.1f}")
    m("IaNcases", str(len(cases)))
    m("IaNattacks", str(len({c.attacker_instruction for c in cases.values()})))
    m("IaNusers", str(len({c.user_instruction for c in cases.values()})))


def prompts_appendix():
    """Verbatim prompt layouts, rendered from the same code that builds the experiments."""
    import textwrap
    from skillkv.agent import Layout
    from skillkv.chat import LLAMA3, QWEN
    from skillkv.inject import doc_layout, make_items, make_natural_items
    from skillkv.injecagent import build_segments, load_cases

    ident = lambda s: [s]  # noqa: E731
    def show(text, limit=None):
        if limit and len(text) > limit:
            text = text[: limit // 2] + "\n[... truncated for display ...]\n" + text[-limit // 2:]
        out = []
        for line in text.split("\n"):
            out += textwrap.wrap(line, 96, break_long_words=True, replace_whitespace=False) or [""]
        return "\\begin{footnotesize}\\begin{verbatim}\n" + "\n".join(out) + "\n\\end{verbatim}\\end{footnotesize}\n"

    items = make_items(12, 0)
    it = next(i for i in items if i.injection == "override")
    parts = []
    for fmt in (QWEN, LLAMA3):
        L = doc_layout(ident, items, defended=False, fmt=fmt)
        L.set_text(it.name, it.doc)
        seg = (L.sys_lazy_text, "".join(L.user_turn(it.question)) + "".join(L.tool_call(it.name)) + "".join(L.wrap),
               it.doc, "".join(L.suffix))
        parts.append(f"\\paragraph{{Agent layout, {fmt.name} format (synthetic document, override injection).}} "
                     "Segments in order: system, user turn + tool call + tool-result header, document, suffix. "
                     "\\cond{T3} packs are built after the system segment plus the header only.\n")
        parts.append(show("<<SYSTEM>>\n" + seg[0] + "\n<<TURN + CALL + HEADER>>\n" + seg[1] + "\n<<DOCUMENT>>\n"
                          + seg[2] + "\n<<SUFFIX>>\n" + seg[3], limit=4200))
    nat = next(i for i in make_natural_items(ROOT / "data" / "natural_docs.json") if i.injection == "append")
    parts.append("\\paragraph{Natural document (append injection), document segment only.}\n")
    parts.append(show(nat.doc, limit=2400))
    from skillkv.inject import DEFENSE
    parts.append("\\paragraph{Defended system prompt addition.}\n" + show(DEFENSE))
    case = load_cases(ROOT / "data" / "injecagent", 4, "enhanced")[0]
    seg = build_segments(ident, QWEN, case)
    parts.append("\\paragraph{InjecAgent (Qwen format).}\n")
    parts.append(show("".join(seg["sys"]) + "".join(seg["turn"]) + "".join(seg["call"]) + "".join(seg["wrap"])
                      + "".join(seg["doc"]) + "".join(seg["suffix"]), limit=3000))
    skills = make_library(12, 0, 400, 2400)
    from skillkv.synth import make_tasks
    L = Layout(ident, [(s.name, s.description, s.text) for s in skills])
    t = make_tasks(skills[0], 1, 0)[0]
    tm = make_tasks(skills[0], 1, 0, mention_all=True)[0]
    parts.append("\\paragraph{Skill task (\\cond{C1}) and the all-rules-named variant (\\cond{C1M}).}\n")
    parts.append(show(t.query + "\n\n" + tm.query))
    (OUT / "appendix_prompts.tex").write_text("% generated by paper/make_assets.py\n" + "".join(parts))



def attention():
    rows = load("attn_*.jsonl")
    if not rows:
        return
    comp = {}
    for r in load("w5_*.jsonl") + load("w7_*.jsonl"):
        if r["seed"] == 0 and not r["defended"] and r["injection"]:
            comp[r["model"], r["tag"], r["item"], r["cond"]] = r["complied"]
    cl = lambda r: (r["tag"], r["doc"])  # noqa: E731
    key = lambda r: (r["tag"], r["item"])  # noqa: E731
    src = lambda r: "nat" if r["tag"].startswith("nat") else "syn"  # noqa: E731

    def contrast(rs, a, b, field):
        by = defaultdict(dict)
        for r in rs:
            by[r["cond"]][key(r)] = r
        ks = sorted(set(by[a]) & set(by[b]))
        return [(cl(by[a][k]), 100 * (by[a][k]["mean"][field] - by[b][k]["mean"][field])) for k in ks]

    def boot_diff(flips, stays, n=2000, seed=0):
        """CI for mean(flip) - mean(stay), resampling documents."""
        import random
        docs = sorted({c for c, _ in flips + stays})
        fd, sd = defaultdict(list), defaultdict(list)
        for c, v in flips:
            fd[c].append(v)
        for c, v in stays:
            sd[c].append(v)
        rng, est = random.Random(seed), []
        for _ in range(n):
            pick = rng.choices(docs, k=len(docs))
            f_ = [v for d in pick for v in fd[d]]
            s_ = [v for d in pick for v in sd[d]]
            if f_ and s_:
                est.append(mean(f_) - mean(s_))
        est.sort()
        v = mean(v for _, v in flips) - mean(v for _, v in stays)
        return f"{num(v, True)} [{num(est[int(0.025 * len(est))])}, {num(est[int(0.975 * len(est))])}]"

    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"& \multicolumn{2}{c}{$\Delta$ attention to injected span} & \multicolumn{2}{c}{$\Delta$ attention to user turn} & Flip $-$ stay \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
             r"Model & Synthetic & Natural & Synthetic & Natural & (pooled) \\", r"\midrule"]
    present = [(mid, label, tag) for mid, label, tag in ALL_MODELS if any(r["model"] == mid for r in rows)]
    for mid, label, tag in present:
        rs = [r for r in rows if r["model"] == mid]
        cells = []
        for field in ("injection", "user_turn"):
            for source in ("syn", "nat"):
                d = contrast([r for r in rs if src(r) == source], "T3", "T1", field)
                cells.append(fmt_ci(d, True))
                m(f"Attn{'Inj' if field == 'injection' else 'User'}{source.title()}{tag}", fmt_ci(d, True))
        flips, stays = [], []
        for r3 in (r for r in rs if r["cond"] == "T3"):
            r1 = next((r for r in rs if r["cond"] == "T1" and key(r) == key(r3)), None)
            c1, c3 = comp.get((mid, r3["tag"], r3["item"], "T1")), comp.get((mid, r3["tag"], r3["item"], "T3"))
            if r1 is None or c1 is None or c3 is None:
                continue
            d = (cl(r3), 100 * (r3["mean"]["injection"] - r1["mean"]["injection"]))
            (flips if (c3 and not c1) else stays).append(d)
        fs = boot_diff(flips, stays) if flips and stays else "--"
        cells.append(fs)
        m(f"AttnFlipStay{tag}", fs)
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_table("attention", lines)

    fig, axes = plt.subplots(1, len(present), figsize=(6.75, 1.9), sharey=True)
    for ax, (mid, label, _) in zip(axes, present):
        style(ax)
        for j, source in enumerate(("syn", "nat")):
            rs = [r for r in rows if r["model"] == mid and src(r) == source]
            by = defaultdict(dict)
            for r in rs:
                by[r["cond"]][key(r)] = r
            ks = sorted(set(by["T1"]) & set(by["T3"]))
            n_layers = len(rs[0]["per_layer"]["injection"])
            ys = [100 * mean(by["T3"][k]["per_layer"]["injection"][l] - by["T1"][k]["per_layer"]["injection"][l] for k in ks)
                  for l in range(n_layers)]
            ax.plot([l / (n_layers - 1) for l in range(n_layers)], ys, color=C[j], linewidth=1.3,
                    label="synthetic" if source == "syn" else "natural")
        ax.axhline(0, color=MUTED, linewidth=0.6)
        ax.set_title(label, fontsize=7, color=INK)
        ax.set_xlabel("Relative depth", fontsize=6.5, color=INK)
    axes[0].set_ylabel("T3$-$T1 attention to\ninjected span (pp)", fontsize=6.5, color=INK)
    axes[0].legend(fontsize=6, frameon=False)
    fig.tight_layout(pad=0.3, w_pad=0.6)
    fig.savefig(OUT / "figures" / "attention_layers.pdf")
    plt.close(fig)


def primary():
    """Pre-designated primary contrasts (T3 - T1 per model x dataset), cluster sign-flip
    permutation p-values and Holm adjustment across the family. InjecAgent crosses attacks with
    user cases, so its p-value is the larger of the two one-factor sign-flip tests and its CI is
    the two-way pigeonhole bootstrap."""
    syn = [r for r in load("w5_*.jsonl") + load("w8_*.jsonl") if not r["defended"] and r["injection"]]
    nat = [r for r in load("w7_*.jsonl") if r["injection"]]
    ia = [r for r in ia_rows()[0] if r["variant"] == "injected"]
    k_doc = lambda r: (r["tag"], r["item"], r["seed"])  # noqa: E731
    c_doc = lambda r: (r["tag"], r["doc"])  # noqa: E731
    fam = []
    for mid, label, tag in ALL_MODELS:
        for name, rows in (("Synthetic", syn), ("Natural", nat), ("InjecAgent", ia)):
            rs = [r for r in rows if r["model"] == mid]
            if not rs or not complete(rs, "T3"):
                continue
            if name == "InjecAgent":
                p_ = max(perm_test(paired(rs, "T3", "T1", IA_KEY, "complied", c)) for c in (IA_CL, lambda r: r["user"]))
                cell = fmt_ci2(rs, "T3", "T1")
            else:
                d = paired(rs, "T3", "T1", k_doc, "complied", c_doc)
                p_, cell = perm_test(d), fmt_ci(d, True)
            fam.append((label, name, cell, p_))
    adj = holm([p for *_, p in fam])
    fp = lambda x: "$<$0.001" if x < 0.001 else f"{x:.3f}"  # noqa: E731
    lines = [r"\begin{tabular}{llrrr}", r"\toprule", r"Model & Data & $\Delta$ T3$-$T1 [95\% CI] & $p$ & $p_\text{Holm}$ \\", r"\midrule"]
    for (label, name, cell, p_), pa in zip(fam, adj):
        lines.append(f"{label} & {name} & {cell} & {fp(p_)} & {fp(pa)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_table("primary", lines)
    m("PrimaryN", str(len(fam)))
    m("PrimarySig", str(sum(pa < 0.05 for pa in adj)))
    m("PrimaryPos", str(sum(pa < 0.05 and not cell.startswith("$-$") for (_, _, cell, _), pa in zip(fam, adj))))


def mechanism_all():
    syn = [r for r in load("w5_*.jsonl") + load("w8_*.jsonl") + load("w3_*.jsonl") if not r["defended"] and r["injection"]]
    # w3 and w5 share T1/T3 rows for the three bf16 models (identical generations); keep one copy per key
    seen, dedup = set(), []
    for r in syn:
        k = (r["model"], r["tag"], r["item"], r["seed"], r["cond"])
        if k not in seen:
            seen.add(k)
            dedup.append(r)
    syn = dedup
    nat = [r for r in load("w7_*.jsonl") if r["injection"]]
    ia = [r for r in ia_rows()[0] if r["variant"] == "injected"]
    cl = lambda r: (r["tag"], r["doc"])  # noqa: E731
    key = lambda r: (r["tag"], r["item"], r["seed"])  # noqa: E731
    ia_cl, ia_key = IA_CL, IA_KEY
    cols = [("Syn.", syn, "T3", "T1M", key, cl), ("Nat.", nat, "T3", "T1M", key, cl), ("InjecAgent", ia, "T3", "T1M", ia_key, ia_cl),
            ("Syn.", syn, "T3G", "T1", key, cl), ("Nat.", nat, "T3G", "T1", key, cl)]
    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"& \multicolumn{3}{c}{T3 $-$ T1$'$ (pack vs.\ masked text)} & \multicolumn{2}{c}{T3G $-$ T1 (conversation prelude)} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-6}",
             r"Model & Syn. & Nat. & InjecAgent & Syn. & Nat. \\", r"\midrule"]
    for mid, label, tag in ALL_MODELS:
        cells = []
        for name, rows, a, b, k, c in cols:
            rs = [r for r in rows if r["model"] == mid]
            if not (complete(rs, a) and complete(rs, b)):
                cells.append("--")
            elif name == "InjecAgent":
                cells.append(fmt_ci2(rs, a, b))
            else:
                cells.append(fmt_ci(paired(rs, a, b, k, "complied", c), signed=True))
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_table("mechanism_all", lines)


SYN_KEY = lambda r: (r["tag"], r["item"], r["seed"])  # noqa: E731
SYN_CL = lambda r: (r["tag"], r["doc"])  # noqa: E731


def dedup(rows):
    seen, out = set(), []
    for r in rows:
        k = (r["model"], r["tag"], r["item"], r["seed"], r["cond"], r.get("spotlight", False))
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out


def syn_rows():
    return dedup([r for r in load("w5_*.jsonl") + load("w8_*.jsonl") + load("w10_*_s*_n12.jsonl")
                  + load("w11_*_s*_n12.jsonl") + load("w12_*_s*_n12.jsonl")
                  if not r["defended"] and not r.get("spotlight", False)])


def nat_rows():
    return dedup(load("w7_*.jsonl") + load("w10_*_nat0.jsonl") + load("w11_*_nat0.jsonl"))


def localization():
    """Partial-mask ablation: which conversation edge does the document need to see? One table per source."""
    conds = [("T1", "T1"), ("T1Mreq", "call masked"), ("T1Mq", "question masked"), ("T1Mcall", "request masked"),
             ("T1M", "T1$'$ (both)"), ("T3", "T3 (pack)")]
    n_rows = 0
    for kind, rows in (("Syn", [r for r in syn_rows() if r["injection"]]), ("Nat", [r for r in nat_rows() if r["injection"]])):
        lines = [r"\begin{tabular}{l" + "r" * len(conds) + "}", r"\toprule",
                 r"Model & " + " & ".join(lab for _, lab in conds) + r" \\", r"\midrule"]
        for mid, label, tag in ALL_MODELS:
            rs = [r for r in rows if r["model"] == mid]
            if not (rs and complete(rs, "T1Mreq") and complete(rs, "T1Mcall")):
                continue
            cells = [f"{mean(float(r['complied']) for r in rs if r['cond'] == c):.2f}" if complete(rs, c) else "--"
                     for c, _ in conds]
            lines.append(f"{label} & " + " & ".join(cells) + r" \\")
            n_rows += kind == "Syn"
            for c, name in (("T1Mreq", "Req"), ("T1Mcall", "Call"), ("T1Mq", "Q")):
                if complete(rs, c):
                    m(f"Loc{kind}{name}{tag}", fmt_ci(paired(rs, c, "T1", SYN_KEY, "complied", SYN_CL), True))
            m(f"Loc{kind}CallReq{tag}", fmt_ci(paired(rs, "T1Mcall", "T1Mreq", SYN_KEY, "complied", SYN_CL), True))
        lines += [r"\bottomrule", r"\end{tabular}"]
        write_table(f"localization_{kind.lower()}", lines)
    m("LocNmodels", str(n_rows))


def spotlight():
    """Spotlighting (datamarking) under packs, synthetic documents."""
    sp = [r for r in load("w9_*.jsonl") if r.get("spotlight") and r["injection"]]
    plain = [r for r in syn_rows() if r["injection"]]
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"& \multicolumn{3}{c}{Plain documents} & \multicolumn{3}{c}{Spotlighting (datamarked)} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
             r"Model & T1 & T3 & $\Delta$ & T1 & T3 & $\Delta$ \\", r"\midrule"]
    n = 0
    for mid, label, tag in ALL_MODELS:
        s_ = [r for r in sp if r["model"] == mid]
        if not s_ or not complete(s_, "T3"):
            continue
        cells = []
        for rows, kind in ((plain, "Plain"), (s_, "Spot")):
            rs = [r for r in rows if r["model"] == mid]
            d = fmt_ci(paired(rs, "T3", "T1", SYN_KEY, "complied", SYN_CL), True)
            cells += [f"{mean(float(r['complied']) for r in rs if r['cond'] == c):.2f}" for c in ("T1", "T3")] + [d]
            m(f"{kind}D{tag}", d)
            m(f"{kind}Tone{tag}", cells[-3])
            m(f"{kind}Tthree{tag}", cells[-2])
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
        n += 1
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_table("spotlight", lines)
    m("SpotNmodels", str(n))


def pertype_all():
    """Compliance per injection type, T1 -> T3, every model (main text)."""
    syn = [r for r in syn_rows() if r["injection"]]
    nat = [r for r in nat_rows() if r["injection"]]
    cols = [(syn, "override"), (syn, "append"), (syn, "refuse"), (nat, "override"), (nat, "append")]
    lines = [r"\begin{tabular}{lccccc}", r"\toprule",
             r"& \multicolumn{3}{c}{Synthetic} & \multicolumn{2}{c}{Natural} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-6}",
             r"Model & override & append & refuse & override & append \\", r"\midrule"]
    for mid, label, tag in ALL_MODELS:
        cells = []
        for rows, kind in cols:
            rs = [r for r in rows if r["model"] == mid and r["injection"] == kind]
            if rs and complete(rs, "T3"):
                a_, b_ = (mean(float(r["complied"]) for r in rs if r["cond"] == c) for c in ("T1", "T3"))
                cells.append(f"{a_:.2f} $\\rightarrow$ {b_:.2f}")
                src = "Syn" if rows is syn else "Nat"
                m(f"Type{src}{kind.title()}{tag}", f"{a_:.2f} $\\rightarrow$ {b_:.2f}")
            else:
                cells.append("--")
        if any(c != "--" for c in cells):
            lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_table("pertype_all", lines)


def cost():
    """Median post-selection TTFT and measured recompute fraction per condition (synthetic)."""
    syn = syn_rows()
    conds = ["T1", "T3", "T3B15", "T3B30", "T3B50"]
    lines = [r"\begin{tabular}{l" + "r" * len(conds) + "}", r"\toprule",
             r"Model & T1 (prefill) & T3 (pack) & T3B15 & T3B30 & T3B50 \\", r"\midrule"]
    for mid, label, tag in ALL_MODELS:
        rs = [r for r in syn if r["model"] == mid]
        if not rs or not complete(rs, "T3B50"):
            continue
        cells = []
        for c in conds:
            sel = [r for r in rs if r["cond"] == c]
            if not complete(rs, c):
                cells.append("--")
                continue
            t = median(r["ttft_s"] for r in sel)
            fr = [r["recompute_fraction"] for r in sel if r.get("recompute_fraction") is not None]
            cells.append(f"{t:.2f}\\,s" + (f" ({100 * mean(fr):.0f}\\%)" if fr else ""))
            m(f"Ttft{c.replace('1', 'one').replace('3', 'three').replace('5', 'five').replace('0', 'zero')}{tag}", f"{t:.2f}")
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_table("cost", lines)
    docs = load("w5_*.jsonl") + load("w7_*.jsonl") + load("w8_*.jsonl") + load("w9_*.jsonl") + load("w10_*.jsonl")
    m("DocTrunc", str(sum(r["gen_tokens"] >= 160 for r in docs)))
    m("DocNgen", f"{len(docs):,}".replace(",", "{,}"))
    fr = [r["recompute_fraction"] for r in syn if r["cond"] == "T3B15" and r.get("recompute_fraction") is not None]
    m("FracBfifteen", f"{100 * mean(fr):.0f}")
    fr = [r["recompute_fraction"] for r in syn if r["cond"] == "T3B50" and r.get("recompute_fraction") is not None]
    m("FracBfifty", f"{100 * mean(fr):.0f}")


def template():
    """Robustness: T1 vs T3 with prompts rendered by each tokenizer's official chat template."""
    rows = load("tpl_*.jsonl")
    from skillkv.inject import make_items, make_natural_items
    n_items = len(make_items(12, 0)) + len(make_items(12, 1)) + len(make_natural_items(ROOT / "data" / "natural_docs.json"))
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"& \multicolumn{2}{c}{Synthetic: $\Delta$ T3$-$T1} & \multicolumn{2}{c}{Natural: $\Delta$ T3$-$T1} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
             r"Model & hand-assembled & official template & hand-assembled & official template \\", r"\midrule"]
    syn = [r for r in syn_rows() if r["injection"]]
    nat = [r for r in nat_rows() if r["injection"]]
    n = 0
    for mid, label, tag in ALL_MODELS:
        # a template run is reported only once every item of all three sets has both conditions
        if sum(r["model"] == mid for r in rows) != 2 * n_items:
            continue
        t = [r for r in rows if r["model"] == mid and r["injection"]]
        cells = []
        for src, ref in (("s", syn), ("nat", nat)):
            ts = [r for r in t if r["tag"].startswith(src)]
            # the template run uses one sampling seed; compare to seed 0 of the main runs
            hs = [r for r in ref if r["model"] == mid and r["seed"] == 0]
            for c in ("T1", "T3"):
                m(f"TplRate{c.replace('1', 'one').replace('3', 'three')}{'Nat' if src == 'nat' else 'Syn'}{tag}",
                  f"{mean(float(r['complied']) for r in ts if r['cond'] == c):.2f}")
            for rows_, name in ((hs, "Hand"), (ts, "Tpl")):
                d = fmt_ci(paired(rows_, "T3", "T1", SYN_KEY, "complied", SYN_CL), True)
                cells.append(d)
                m(f"Tpl{name}{'Nat' if src == 'nat' else 'Syn'}{tag}", d)
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
        n += 1
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_table("template", lines)
    m("TplNmodels", str(n))


def glmm():
    """Sensitivity check for InjecAgent: crossed random-effects logistic model (attack, user case and
    case intercepts; variational Bayes), T3 vs T1 on injected cases."""
    import pandas as pd
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    rows, _ = ia_rows()
    for mid, label, tag in ALL_MODELS:
        rs = [r for r in rows if r["model"] == mid and r["variant"] == "injected" and r["cond"] in ("T1", "T3")]
        if not rs:
            continue
        d = pd.DataFrame([{"y": int(r["attack_success"]), "t3": int(r["cond"] == "T3"), "attack": r["attack"], "user": r["user"],
                           "case": f"{r['kind']}{r['index']}"} for r in rs])
        fit = BinomialBayesMixedGLM.from_formula(
            "y ~ t3", {"attack": "0 + C(attack)", "user": "0 + C(user)", "case": "0 + C(case)"}, d).fit_vb()
        b, sd = fit.fe_mean[1], fit.fe_sd[1]
        m(f"Glmm{tag}", f"{num(b, True)} ({sd:.2f})")


def main():
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    pilot()
    w1()
    w2()
    injection()
    scale_natural()
    injecagent()
    prompts_appendix()
    attention()
    mechanism_all()
    primary()
    localization()
    spotlight()
    pertype_all()
    cost()
    template()
    glmm()
    (OUT / "numbers.tex").write_text(
        "% generated by paper/make_assets.py - do not edit\n"
        + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(macros.items())))
    print(f"{len(macros)} macros, tables: {sorted(p.name for p in (OUT / 'tables').iterdir())}")


if __name__ == "__main__":
    main()
