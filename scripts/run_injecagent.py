"""InjecAgent under KV-reuse conditions (agent layout, tool response after the conversation).

  T1    tool response prefilled in context            T3     context-free pack ([sys][tool open])
  T1M   text, response masked from user turn + call   T3B15  T3 + CacheBlend-style 15% recompute
Clean controls (attacker instruction removed from the response) run for T1 and T3.

  uv run python scripts/run_injecagent.py --model Qwen/Qwen3-4B
"""

import argparse
import json
import zlib
from dataclasses import replace
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache
from mlx_lm.sample_utils import make_sampler

from skillkv.agent import generate
from skillkv.blend import blend
from skillkv.chat import detect_format
from skillkv.injecagent import attack_success, build_segments, load_cases, tools_called
from skillkv.pack import build_pack, prefill, splice

CONDS = ("T1", "T3", "T1M", "T3B15")
CLEAN_CONDS = ("T1", "T3")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--data", type=Path, default=Path("data/injecagent"))
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--setting", choices=["base", "enhanced"], default="enhanced")
    ap.add_argument("--clean-stride", type=int, default=4, help="clean controls on every k-th case")
    ap.add_argument("--conds", default=",".join(CONDS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    conds = args.conds.split(",")
    if set(conds) - set(CONDS):
        raise SystemExit(f"unknown conditions {sorted(set(conds) - set(CONDS))}; choose from {CONDS}")
    slug = args.model.replace("/", "--")
    out = args.out or Path(f"outputs/ia{args.max_tokens}_{args.setting}_{slug}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    cases = load_cases(args.data, args.stride, args.setting)[: args.limit]
    done = set()
    if out.exists():
        done = {(r["kind"], r["index"], r["variant"], r["cond"]) for r in map(json.loads, out.read_text().splitlines())
                if r and r["seed"] == args.seed}

    model, tok = load(args.model)
    enc = lambda s: tok.encode(s, add_special_tokens=False)  # noqa: E731
    eos = set(getattr(tok, "eos_token_ids", None) or [tok.eos_token_id])
    sampler = make_sampler(temp=0.7, top_p=0.8, top_k=20)
    fmt = detect_format(tok)
    print(f"chat format: {fmt.name}; {len(cases)} cases", flush=True)

    with out.open("a") as f:
        for n, case in enumerate(cases):
            variants = ("injected", "clean") if n % args.clean_stride == 0 else ("injected",)
            for variant in variants:
                c = case if variant == "injected" else replace(
                    case, tool_response=case.tool_response.replace(case.injected_text, ""))
                todo = [x for x in (conds if variant == "injected" else [x for x in conds if x in CLEAN_CONDS])
                        if (c.kind, c.index, variant, x) not in done]
                if not todo:
                    continue
                seg = build_segments(enc, fmt, c)
                pack = None
                if any(x in ("T3", "T3B15") for x in todo):
                    pack = build_pack(model, seg["sys"] + seg["wrap"], seg["doc"], model_id=args.model)
                for cond in todo:
                    cache = make_prompt_cache(model)
                    prefill(model, cache, seg["sys"])
                    if cond == "T1":
                        prefill(model, cache, seg["turn"] + seg["call"] + seg["wrap"] + seg["doc"])
                    elif cond == "T1M":
                        prefill(model, cache, seg["turn"] + seg["call"])
                        lo = len(seg["sys"])
                        blend(model, cache, seg["wrap"] + seg["doc"], select="all",
                              exclude=(lo, lo + len(seg["turn"]) + len(seg["call"])))
                    else:
                        prefill(model, cache, seg["turn"] + seg["call"] + seg["wrap"])
                        if cond == "T3":
                            splice(cache, pack, model, model_id=args.model)
                        else:
                            blend(model, cache, seg["doc"], pack=pack, model_id=args.model, select=("deviation", 0.15))
                    logits = prefill(model, cache, seg["suffix"])
                    mx.eval(logits)
                    gen_seed = zlib.crc32(f"ia:{c.kind}:{c.index}:{variant}:{args.seed}".encode())
                    ids = generate(model, cache, logits, args.max_tokens, eos, sampler=sampler, seed=gen_seed)
                    answer = tok.decode(ids)
                    row = {"model": args.model, "kind": c.kind, "index": c.index, "variant": variant, "cond": cond,
                           "seed": args.seed, "setting": args.setting, "attack_success": attack_success(answer, c),
                           "called_user_tool_again": c.user_tool in answer, "gen_tokens": len(ids),
                           "tools_called": tools_called(answer), "truncated": len(ids) >= args.max_tokens,
                           "max_tokens": args.max_tokens,
                           "chat_format": fmt.name, "answer": answer}
                    f.write(json.dumps(row) + "\n")
                    f.flush()
                    print(f"[{n + 1}/{len(cases)}] {c.kind}{c.index:<4} {variant:<8} {cond:<5} "
                          f"success={int(row['attack_success'])}", flush=True)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
