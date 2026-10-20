"""W1: task success of C0-C3 on the synthetic verifiable skill library.

Per task: (1) the model selects a skill with load_skill from the catalog (shared by C1-C3);
(2) each condition answers with the oracle skill loaded, so end-to-end success =
selection correct AND oracle-loaded answer passes. Rows are appended as they finish
(re-running skips done rows).

  uv run python scripts/run_w1.py --model Qwen/Qwen3-1.7B
"""

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache

from skillkv.agent import Layout, generate, parse_selection, post_selection
from skillkv.pack import build_pack, load_pack, prefill, save_pack, token_hash
from skillkv.synth import grade, make_library, make_tasks, write_library


def reset(cache, n: int) -> None:
    trim_prompt_cache(cache, cache[0].offset - n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--n-skills", type=int, default=12)
    ap.add_argument("--tasks-per-skill", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-words", type=int, default=400)
    ap.add_argument("--max-words", type=int, default=2400)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--conds", default="C0,C1,C2,C3")
    ap.add_argument("--limit", type=int, help="first N tasks only (smoke test)")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    slug = args.model.replace("/", "--")
    tag = f"s{args.seed}_n{args.n_skills}"
    out = args.out or Path(f"outputs/w1_{slug}_{tag}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    pack_dir = Path("packs") / slug / f"synth_{tag}"
    pack_dir.mkdir(parents=True, exist_ok=True)
    conds = args.conds.split(",")

    lib = make_library(args.n_skills, args.seed, args.min_words, args.max_words)
    write_library(lib, Path(f"outputs/library_{tag}"))
    by_name = {s.name: s for s in lib}
    tasks = [t for s in lib for t in make_tasks(s, args.tasks_per_skill, args.seed)][: args.limit]

    done = set()
    if out.exists():
        done = {(r["task_id"], r["cond"]) for r in map(json.loads, out.read_text().splitlines()) if r}

    model, tok = load(args.model)
    enc = lambda s: tok.encode(s, add_special_tokens=False)  # noqa: E731
    eos = set(getattr(tok, "eos_token_ids", None) or [tok.eos_token_id])
    L = Layout(enc, [(s.name, s.description, s.text) for s in lib])
    max_pos = getattr(model.args, "max_position_embeddings", None)

    # packs, precomputed once per (model, library): C3 after [sys][<tool_response>],
    # C2 after [sys "# Loaded skill"] (an exact prefix)
    preludes = {"C3": L.pack_prelude, "C2": L.c2_prelude}
    pack_path = lambda cond, name: pack_dir / f"{name}.{cond}.safetensors"  # noqa: E731
    for cond, prelude in preludes.items():
        if cond not in conds:
            continue
        want = token_hash(prelude)
        for s in lib:
            path = pack_path(cond, s.name)
            if path.exists() and load_pack(path).prelude_hash == want:
                continue
            t0 = time.perf_counter()
            save_pack(path, build_pack(model, prelude, L.skill(s.name), model_id=args.model))
            print(f"{cond} pack {s.name}: {len(L.skill(s.name))} tok, built {time.perf_counter() - t0:.1f}s", flush=True)

    lazy = make_prompt_cache(model)
    prefill(model, lazy, L.sys_lazy)
    n_lazy = lazy[0].offset
    c2 = make_prompt_cache(model)
    prefill(model, c2, L.c2_prelude)
    n_c2 = c2[0].offset
    eager, n_eager, t_eager = None, 0, None
    if "C0" in conds:
        if max_pos and len(L.sys_eager) + 1000 > max_pos:
            raise SystemExit(f"C0 needs {len(L.sys_eager)} system tokens; model max is {max_pos}. "
                             "Use fewer/shorter skills or drop C0.")
        eager = make_prompt_cache(model)
        t0 = time.perf_counter()
        prefill(model, eager, L.sys_eager)
        t_eager = time.perf_counter() - t0
        n_eager = eager[0].offset
        print(f"C0 eager system prompt: {n_eager} tokens, one-time prefill {t_eager:.1f}s", flush=True)

    with out.open("a") as f:
        for i, task in enumerate(tasks):
            tid = f"{task.skill_name}#{i}"
            skill = by_name[task.skill_name]
            todo = [c for c in conds if (tid, c) not in done]
            if not todo:
                continue
            # selection (identical context for C1-C3)
            reset(lazy, n_lazy)
            t0 = time.perf_counter()
            first = prefill(model, lazy, L.user_turn(task.query))
            sel_ids = generate(model, lazy, first, 48, eos, decode=tok.decode, stop="</tool_call>")
            t_sel = time.perf_counter() - t0
            sel_text = tok.decode(sel_ids)
            selected = parse_selection(sel_text, known=set(by_name))
            sel_strict = parse_selection(sel_text) == skill.name
            depth = {c.type: c.depth for c in skill.constraints}
            for cond in todo:
                cache, base = {"C0": (eager, n_eager), "C2": (c2, n_c2)}.get(cond, (lazy, n_lazy))
                reset(cache, base)
                path = pack_path(cond, skill.name)
                logits, ttft = post_selection(model, cache, L, cond, task.query,
                                              None if cond == "C0" else skill.name,
                                              pack=lambda: load_pack(path), model_id=args.model)
                t0 = time.perf_counter()
                ids = generate(model, cache, logits, args.max_tokens, eos)
                t_gen = time.perf_counter() - t0
                answer = tok.decode(ids)
                g = grade(answer, skill)
                row = {"model": args.model, "tag": tag, "task_id": tid, "skill": skill.name,
                       "skill_tokens": len(L.skill(skill.name)), "cond": cond, "query": task.query,
                       "selected": selected, "sel_correct": selected == skill.name, "sel_strict": sel_strict,
                       "sel_s": t_sel,
                       "sel_text": sel_text, "ttft_s": ttft, "gen_tokens": len(ids), "gen_s": t_gen,
                       "grade": g, "all_pass": all(g.values()), "depth": depth, "answer": answer,
                       "c0_prefill_once_s": t_eager if cond == "C0" else None}
                f.write(json.dumps(row) + "\n")
                f.flush()
                print(f"[{i + 1}/{len(tasks)}] {tid:<32} {cond} sel={'ok' if row['sel_correct'] else selected} "
                      f"pass={sum(g.values())}/5 ttft={ttft:.3f}s gen={len(ids)}tok/{t_gen:.1f}s", flush=True)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
