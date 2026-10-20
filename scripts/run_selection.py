"""Selection accuracy vs library size N: one load_skill decision per task from the catalog only.

  uv run python scripts/run_selection.py --model Qwen/Qwen3-4B --sizes 12,25,50,100
"""

import argparse
import json
import time
from pathlib import Path

from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache

from skillkv.agent import Layout, generate, parse_selection
from skillkv.pack import prefill
from skillkv.synth import make_library, make_tasks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--sizes", default="12,25,50,100")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tasks", type=int, default=100)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    slug = args.model.replace("/", "--")
    out = args.out or Path(f"outputs/selection_{slug}_s{args.seed}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    model, tok = load(args.model)
    enc = lambda s: tok.encode(s, add_special_tokens=False)  # noqa: E731
    eos = set(getattr(tok, "eos_token_ids", None) or [tok.eos_token_id])

    with out.open("w") as f:
        for n in map(int, args.sizes.split(",")):
            lib = make_library(n, args.seed, min_words=40, max_words=60)  # only the catalog matters here
            names = {s.name for s in lib}
            L = Layout(enc, [(s.name, s.description, s.text) for s in lib])
            cache = make_prompt_cache(model)
            t0 = time.perf_counter()
            prefill(model, cache, L.sys_lazy)
            t_catalog = time.perf_counter() - t0
            base = cache[0].offset
            tasks = [make_tasks(s, 1, args.seed)[0] for s in lib][: args.max_tasks]
            ok = strict = 0
            for t in tasks:
                trim_prompt_cache(cache, cache[0].offset - base)
                t0 = time.perf_counter()
                first = prefill(model, cache, L.user_turn(t.query))
                ids = generate(model, cache, first, 48, eos, decode=tok.decode, stop="</tool_call>")
                dt = time.perf_counter() - t0
                text = tok.decode(ids)
                sel = parse_selection(text, known=names)
                ok += sel == t.skill_name
                strict += parse_selection(text) == t.skill_name
                f.write(json.dumps({"model": args.model, "n": n, "catalog_tokens": base,
                                    "catalog_prefill_s": t_catalog, "skill": t.skill_name,
                                    "selected": sel, "correct": sel == t.skill_name,
                                    "strict": parse_selection(text) == t.skill_name,
                                    "sel_s": dt, "text": text}) + "\n")
            print(f"N={n:<4} catalog {base} tok ({t_catalog:.1f}s)  acc {ok / len(tasks):.3f}  "
                  f"strict {strict / len(tasks):.3f}  ({len(tasks)} tasks)", flush=True)


if __name__ == "__main__":
    main()
