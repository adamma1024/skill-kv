"""Synthetic, verifiable SKILL.md library.

Each skill is a realistic-looking procedure document for one (object, action) domain, e.g.
`invoice-reconciliation`, with five hard constraints planted at controlled relative depths:

  code      deliverable must start with `CODE: QX-4821`
  filename  output file name must use the prefix `vx7_`
  param     a named setting must be a specific 4-digit value
  heading   step list under an exact `### Plan Amber-Ridge` heading
  signoff   deliverable ends with `-- kestrel-42 --`

Types rotate over the depth bins across skills, so every (type, depth) cell is covered.
A task names the domain (not the skill) and asks for the deliverable; only the skill text
holds the constraint values, so the grader measures what the model retained from the skill.
"""

import random
import re
from dataclasses import dataclass
from pathlib import Path

CONSTRAINT_TYPES = ("code", "filename", "param", "heading", "signoff")
DEPTHS = (0.05, 0.25, 0.45, 0.65, 0.85)

OBJECTS = ["invoice", "sensor", "log", "dataset", "backup", "certificate", "container", "schema",
           "feature flag", "payment", "shipment", "firmware", "report", "ticket", "cache",
           "search index", "newsletter", "api key", "translation", "image asset"]
ACTIONS = [("reconciliation", "reconcile"), ("calibration", "calibrate"), ("rotation", "rotate"),
           ("deduplication", "deduplicate"), ("migration", "migrate"), ("audit", "audit"),
           ("rollout", "roll out"), ("validation", "validate"), ("archival", "archive"),
           ("triage", "triage")]
PARAM_KEYS = ["retry_window_ms", "batch_ceiling", "max_parallel_jobs", "lookback_days",
              "sample_size", "timeout_seconds", "chunk_rows", "grace_period_min", "error_budget",
              "checkpoint_interval"]
WORDS_A = ["Amber", "Cobalt", "Juniper", "Saffron", "Granite", "Willow", "Indigo", "Copper",
           "Hazel", "Onyx", "Cedar", "Coral", "Slate", "Ember", "Frost", "Maple"]
WORDS_B = ["Ridge", "Harbor", "Meadow", "Summit", "Canyon", "Delta", "Grove", "Beacon",
           "Bluff", "Inlet", "Mesa", "Hollow", "Crest", "Fjord", "Prairie", "Reef"]
ANIMALS = ["kestrel", "otter", "lynx", "heron", "marten", "ibex", "puffin", "gecko", "bison",
           "osprey", "tapir", "wren", "condor", "civet", "dingo", "egret"]
CONTEXTS = ["for the quarterly close", "before Friday's release", "for the new EU region",
            "after last night's incident", "for the compliance review", "for the staging cluster",
            "ahead of the customer migration", "for the data team's audit"]

THINGS = ["source system", "staging copy", "owner list", "change ticket", "dry-run output",
          "dependency map", "monitoring dashboard", "rollback plan", "approval record",
          "baseline snapshot", "access policy", "sample batch"]
STATES = ["current", "signed off", "reachable", "frozen for edits", "consistent with production",
          "recorded in the ticket", "reviewed by the owner", "under the agreed size"]
ISSUES = ["unexpected gaps", "duplicate entries", "stale timestamps", "permission errors",
          "drift from the baseline", "partial writes", "schema mismatches", "slow responses"]
REMEDIES = ["re-run the dry run", "escalate to the owner", "compare against the baseline snapshot",
            "split the work into smaller batches", "restore from the last checkpoint",
            "document the anomaly before continuing", "ask for a second review"]
REVIEWS = ["trace every decision", "repeat the run later", "audit the outcome",
           "spot regressions quickly", "hand the work over cleanly"]
SECTIONS = ["Overview", "When to use this skill", "Inputs", "Workflow", "Checks", "Examples",
            "Common pitfalls", "Reference"]
TEMPLATES = [
    "Before you {verb} any {obj} records, confirm that the {thing} is {state}.",
    "If the {thing} shows {issue}, stop and {remedy}.",
    "Keep a short log of each {obj} {noun} step so that reviewers can {review}.",
    "Treat the {thing} as the source of truth whenever it disagrees with ad hoc notes.",
    "Work through the {obj} items in small groups; large groups hide {issue}.",
    "A good {obj} {noun} leaves the {thing} {state} and the history easy to follow.",
    "When in doubt, {remedy} rather than guessing.",
    "Do not skip the {thing}: most failed runs trace back to {issue}.",
    "Share the {thing} with the owner once it is {state}.",
    "Compare the result with the {thing} and note any {issue} you find.",
    "The {noun} is only complete when the {thing} is {state}.",
    "Prefer reversible steps; it should always be possible to {remedy}.",
]


@dataclass(frozen=True)
class Constraint:
    type: str
    value: str
    depth: float
    statement: str
    key: str | None = None  # param name, for type == "param"


@dataclass(frozen=True)
class SynthSkill:
    name: str
    description: str
    text: str
    constraints: tuple[Constraint, ...]
    object_noun: str
    action_phrase: str


@dataclass(frozen=True)
class Task:
    skill_name: str
    query: str


def _unique(rng: random.Random, used: set, fn):
    for _ in range(10_000):
        v = fn()
        if v not in used:
            used.add(v)
            return v
    raise RuntimeError("could not draw a unique constraint value; library too large for the value space")


def _statement(t: str, value: str, key: str | None) -> str:
    return {
        "code": f"Activation code: every deliverable produced with this skill must begin with the line `CODE: {value}`.",
        "filename": f"Output files produced by this skill must be named with the prefix `{value}_` followed by a short snake_case description.",
        "param": f"The `{key}` setting must always be set to {value} for this workflow; never use the default.",
        "heading": f"Put the step list under the exact heading `### {value}`.",
        "signoff": f"Finish every deliverable with the sign-off line `-- {value} --`.",
    }[t]


def _filler(rng: random.Random, n_words: int, obj: str, verb: str, noun: str) -> list[str]:
    paras, words, sec = [], 0, 0
    per_section = max(n_words // len(SECTIONS), 30)
    since_heading = per_section
    while words < n_words:
        if since_heading >= per_section and sec < len(SECTIONS):
            paras.append(f"## {SECTIONS[sec]}")
            sec += 1
            since_heading = 0
        if SECTIONS[max(sec - 1, 0)] == "Examples" and rng.random() < 0.3:
            p = f"```bash\n{obj.replace(' ', '_')}ctl {verb.replace(' ', '-')} --dry-run --owner team-{rng.randint(1, 9)}\n```"
        else:
            p = " ".join(rng.choice(TEMPLATES).format(
                verb=verb, obj=obj, noun=noun, thing=rng.choice(THINGS), state=rng.choice(STATES),
                issue=rng.choice(ISSUES), remedy=rng.choice(REMEDIES), review=rng.choice(REVIEWS))
                for _ in range(rng.randint(2, 4)))
        paras.append(p)
        w = len(p.split())
        words += w
        since_heading += w
    return paras


def make_library(n: int, seed: int = 0, min_words: int = 400, max_words: int = 2400) -> list[SynthSkill]:
    domains = [(o, a) for o in OBJECTS for a in ACTIONS]
    if n > len(domains):
        raise ValueError(f"n={n} exceeds the {len(domains)} available domains")
    rng = random.Random(seed)
    rng.shuffle(domains)
    lengths = [round(min_words + (max_words - min_words) * i / max(n - 1, 1)) for i in range(n)]
    rng.shuffle(lengths)
    used: dict[str, set] = {t: set() for t in CONSTRAINT_TYPES}
    letters = "abcdefghjkmnpqrstuvwxyz"
    skills = []
    for i, ((obj, (noun, verb)), n_words) in enumerate(zip(domains[:n], lengths)):
        name = f"{obj.replace(' ', '-')}-{noun}"
        desc = f"Use when you need to {verb} {obj} records: procedure, checks and deliverable format."
        key = rng.choice(PARAM_KEYS)
        values = {
            "code": _unique(rng, used["code"], lambda: f"{rng.choice(letters).upper()}{rng.choice(letters).upper()}-{rng.randint(1000, 9999)}"),
            "filename": _unique(rng, used["filename"], lambda: f"{rng.choice(letters)}{rng.choice(letters)}{rng.randint(2, 9)}"),
            "param": _unique(rng, used["param"], lambda: str(rng.randint(1000, 9999))),
            "heading": _unique(rng, used["heading"], lambda: f"Plan {rng.choice(WORDS_A)}-{rng.choice(WORDS_B)}"),
            "signoff": _unique(rng, used["signoff"], lambda: f"{rng.choice(ANIMALS)}-{rng.randint(10, 99)}"),
        }
        cons = [Constraint(t, values[t], DEPTHS[(i + j) % len(DEPTHS)],
                           _statement(t, values[t], key if t == "param" else None),
                           key if t == "param" else None)
                for j, t in enumerate(CONSTRAINT_TYPES)]
        paras = _filler(rng, n_words, obj, verb, noun)
        cum, total = [], 0
        for p in paras:
            cum.append(total)
            total += len(p.split())
        # insert from the deepest so earlier indices stay valid
        for c in sorted(cons, key=lambda c: -c.depth):
            idx = next((k for k, w in enumerate(cum) if w >= c.depth * total), len(paras))
            paras.insert(idx, c.statement)
        title = f"{obj.title()} {noun.title()}"
        text = f"---\nname: {name}\ndescription: {desc}\n---\n# {title}\n\n" + "\n\n".join(paras) + "\n"
        skills.append(SynthSkill(name, desc, text, tuple(cons), obj, verb))
    return skills


def make_tasks(skill: SynthSkill, k: int = 2, seed: int = 0, mention_all: bool = False) -> list[Task]:
    """k tasks for `skill`. mention_all names all five deliverable items (never their values):
    the W2 ablation testing whether C1 only drops rules the query does not ask about."""
    rng = random.Random(f"{seed}:{skill.name}")
    key = next(c.key for c in skill.constraints if c.type == "param")
    ctxs = rng.sample(CONTEXTS, k)
    if mention_all:
        ask = (f"start with the skill's activation code line, put a short step list under the "
               f"skill's required heading, give the output file name and the {key} value you will "
               f"use, and end with the skill's sign-off line")
    else:
        ask = f"a short step list, the output file name, and the {key} value you will use"
    return [Task(skill.name,
                 f"I need to {skill.action_phrase} our {skill.object_noun} records {ctx}. "
                 f"Use the right skill and produce the deliverable: {ask}. Write the deliverable as plain text.")
            for ctx in ctxs]


def grade(answer: str, skill: SynthSkill) -> dict[str, bool]:
    v = {c.type: c.value for c in skill.constraints}
    pats = {
        "code": r"code\s*:\s*" + re.escape(v["code"]),
        "filename": r"(?<![A-Za-z0-9])" + re.escape(v["filename"]) + r"_[A-Za-z0-9_]+",
        "param": r"(?<!\d)" + re.escape(v["param"]) + r"(?!\d)",
        "heading": r"#+\s*" + re.escape(v["heading"]),
        "signoff": r"--\s*" + re.escape(v["signoff"]) + r"\s*--",
    }
    return {t: re.search(p, answer, re.I) is not None for t, p in pats.items()}


def write_library(skills: list[SynthSkill], root: Path) -> None:
    for s in skills:
        d = Path(root) / s.name
        d.mkdir(parents=True, exist_ok=True)
        (d / "SKILL.md").write_text(s.text)
