"""W2: is C3 > C1 real? Sampling (3 seeds, shared per task across conditions), max_tokens 512,
oracle skill, plus mechanism ablations:

  C1   lazy text, task query                     C2   sys-prefix exact pack
  C3   KV splice, pack saw [sys][<tool_response>] (no question)
  C1M  C1 with a query that names all five deliverable items (values never revealed)
  C3M  C3 with that same mention-all query
  C3G  KV splice, pack saw a content-free question + the load_skill call (C1-shaped prefix)

If C1 only drops rules the question does not ask for, C1M should close most of the C3-C1 gap.
C3G separates "saw no question" (C3) from "saw a question-shaped context" (C3G).

  uv run python scripts/run_w2.py --model Qwen/Qwen3-4B --seed 0
"""

import argparse
import json
import zlib
from pathlib import Path

from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache
from mlx_lm.sample_utils import make_sampler

from skillkv.agent import Layout, generate, post_selection
from skillkv.pack import build_pack, load_pack, prefill, save_pack, token_hash
from skillkv.synth import grade, make_library, make_tasks

# condition -> (layout condition, mention_all query, pack kind)
CONDS = {"C1": ("C1", False, None), "C2": ("C2", False, "C2"), "C3": ("C3", False, "C3"),
         "C1M": ("C1", True, None), "C3M": ("C3", True, "C3"), "C3G": ("C3", False, "C3G")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--n-skills", type=int, default=12)
    ap.add_argument("--tasks-per-skill", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0, help="library seed")
    ap.add_argument("--sample-seeds", default="0,1,2")
    ap.add_argument("--temp", type=float, default=0.7)  # Qwen3 non-thinking recommendation
    ap.add_argument("--top-p", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--conds", default=",".join(CONDS))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    conds = args.conds.split(",")
    unknown = set(conds) - set(CONDS)
    if unknown:
        raise SystemExit(f"unknown conditions {sorted(unknown)}; choose from {list(CONDS)}")
    slug = args.model.replace("/", "--")
    tag = f"s{args.seed}_n{args.n_skills}"
    out = args.out or Path(f"outputs/w2_{slug}_{tag}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    pack_dir = Path("packs") / slug / f"synth_{tag}"
    pack_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.sample_seeds.split(",")]

    lib = make_library(args.n_skills, args.seed, 400, 2400)
    by_name = {s.name: s for s in lib}
    base_tasks = [t for s in lib for t in make_tasks(s, args.tasks_per_skill, args.seed)]
    all_tasks = [t for s in lib for t in make_tasks(s, args.tasks_per_skill, args.seed, mention_all=True)]
    tasks = list(zip(base_tasks, all_tasks))[: args.limit]

    done = set()
    if out.exists():
        done = {(r["task_id"], r["seed"], r["cond"]) for r in map(json.loads, out.read_text().splitlines()) if r}

    model, tok = load(args.model)
    enc = lambda s: tok.encode(s, add_special_tokens=False)  # noqa: E731
    eos = set(getattr(tok, "eos_token_ids", None) or [tok.eos_token_id])
    sampler = make_sampler(temp=args.temp, top_p=args.top_p, top_k=args.top_k)
    L = Layout(enc, [(s.name, s.description, s.text) for s in lib])

    def prelude(kind, name):
        return {"C2": L.c2_prelude, "C3": L.pack_prelude, "C3G": L.generic_prelude(name)}[kind]

    def path(kind, name):
        return pack_dir / f"{name}.{kind}.safetensors"

    for kind in sorted({CONDS[c][2] for c in conds} - {None}):
        for s in lib:
            p, pre = path(kind, s.name), prelude(kind, s.name)
            if p.exists() and load_pack(p).prelude_hash == token_hash(pre):
                continue
            save_pack(p, build_pack(model, pre, L.skill(s.name), model_id=args.model))
            print(f"built {kind} pack {s.name}", flush=True)

    lazy = make_prompt_cache(model)
    prefill(model, lazy, L.sys_lazy)
    n_lazy = lazy[0].offset
    c2 = make_prompt_cache(model)
    prefill(model, c2, L.c2_prelude)
    n_c2 = c2[0].offset

    with out.open("a") as f:
        for i, (task, task_all) in enumerate(tasks):
            tid = f"{task.skill_name}#{i}"
            skill = by_name[task.skill_name]
            depth = {c.type: c.depth for c in skill.constraints}
            for seed in seeds:
                gen_seed = zlib.crc32(f"{tag}:{tid}:{seed}".encode())  # shared across conditions
                for cond in conds:
                    if (tid, seed, cond) in done:
                        continue
                    lcond, mention_all, kind = CONDS[cond]
                    q = task_all.query if mention_all else task.query
                    cache, base = (c2, n_c2) if lcond == "C2" else (lazy, n_lazy)
                    trim_prompt_cache(cache, cache[0].offset - base)
                    logits, ttft = post_selection(
                        model, cache, L, lcond, q, skill.name, model_id=args.model,
                        pack=(lambda k=kind: load_pack(path(k, skill.name))) if kind else None)
                    ids = generate(model, cache, logits, args.max_tokens, eos, sampler=sampler, seed=gen_seed)
                    answer = tok.decode(ids)
                    g = grade(answer, skill)
                    row = {"model": args.model, "tag": tag, "task_id": tid, "skill": skill.name,
                           "cond": cond, "seed": seed, "query": q, "ttft_s": ttft,
                           "gen_tokens": len(ids), "truncated": len(ids) >= args.max_tokens,
                           "grade": g, "all_pass": all(g.values()), "depth": depth, "answer": answer,
                           "temp": args.temp, "top_p": args.top_p, "top_k": args.top_k,
                           "max_tokens": args.max_tokens}
                    f.write(json.dumps(row) + "\n")
                    f.flush()
                    print(f"[{i + 1}/{len(tasks)}] {tid:<30} s{seed} {cond:<3} pass={sum(g.values())}/5 "
                          f"gen={len(ids)}{' TRUNC' if row['truncated'] else ''} ttft={ttft:.3f}s", flush=True)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
