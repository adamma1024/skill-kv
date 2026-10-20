"""W6: RAG-layout control for RQ7. Classic RAG puts documents before the question:

  [sys][user: "Documents:" doc1 doc2 doc3 "Question: q"][assistant]

so even a full prefill never lets the documents attend to the question. Precomputing chunks
(CacheBlend-style) only removes attention between documents. Does it still amplify injection?

  R1     full prefill (reference)
  R3     each chunk precomputed after the shared head only (independent chunks), spliced
  R3N    each chunk precomputed with no prelude at all (fully standalone), spliced
  R3B15  R3 packs mounted with CacheBlend-style 15% selective recompute

The injected document sits at position idx % 3 among 3 retrieved documents (others clean).
At position 0 the injected chunk's own KV is exact under R3; the chunks after it are not.

  uv run python scripts/run_w6.py --model Qwen/Qwen3-4B --seed 0
"""

import argparse
import json
import time
import zlib
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler

from skillkv.agent import generate
from skillkv.blend import blend
from skillkv.chat import detect_format
from skillkv.inject import grade_item, make_items, rag_docs, rag_question, rag_segments
from skillkv.pack import build_pack, prefill, splice

CONDS = ("R1", "R3", "R3N", "R3B15")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--n-docs", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0, help="library seed")
    ap.add_argument("--k", type=int, default=3, help="retrieved documents per prompt")
    ap.add_argument("--sample-seeds", default="0,1")
    ap.add_argument("--conds", default=",".join(CONDS))
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=160)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    conds = args.conds.split(",")
    if set(conds) - set(CONDS):
        raise SystemExit(f"unknown conditions {sorted(set(conds) - set(CONDS))}; choose from {CONDS}")
    slug = args.model.replace("/", "--")
    tag = f"s{args.seed}_n{args.n_docs}"
    out = args.out or Path(f"outputs/w6_{slug}_{tag}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.sample_seeds.split(",")]
    items = make_items(args.n_docs, args.seed)

    done = set()
    if out.exists():
        done = {(r["item"], r["seed"], r["cond"]) for r in map(json.loads, out.read_text().splitlines()) if r}

    model, tok = load(args.model)
    enc = lambda s: tok.encode(s, add_special_tokens=False)  # noqa: E731
    eos = set(getattr(tok, "eos_token_ids", None) or [tok.eos_token_id])
    sampler = make_sampler(temp=args.temp, top_p=args.top_p, top_k=args.top_k)
    fmt = detect_format(tok)
    print(f"chat format: {fmt.name}", flush=True)

    with out.open("a") as f:
        for idx, it in enumerate(items[: args.limit]):
            todo = [(s, c) for s in seeds for c in conds if (idx, s, c) not in done]
            if not todo:
                continue
            q = rag_question(it)
            seg = rag_segments(enc, fmt, rag_docs(items, idx, args.k), q)
            need = {c for _, c in todo}
            packs = {}
            if need & {"R3", "R3B15"}:
                packs["head"] = [build_pack(model, seg["head"], s, model_id=args.model) for s in seg["segs"]]
            if "R3N" in need:
                packs["none"] = [build_pack(model, [], s, model_id=args.model) for s in seg["segs"]]
            for seed, cond in todo:
                cache = make_prompt_cache(model)
                prefill(model, cache, seg["head"])
                mx.eval([c.keys_and_values() for c in cache])
                t0 = time.perf_counter()
                frac = None
                if cond == "R1":
                    logits = prefill(model, cache, [t for s in seg["segs"] for t in s] + seg["tail"])
                else:
                    if cond == "R3B15":
                        fracs = [blend(model, cache, s, pack=p, model_id=args.model, select=("deviation", 0.15))
                                 ["recompute_fraction"] for s, p in zip(seg["segs"], packs["head"])]
                        frac = sum(f * len(s) for f, s in zip(fracs, seg["segs"])) / sum(len(s) for s in seg["segs"])
                    else:
                        for p in packs["head" if cond == "R3" else "none"]:
                            splice(cache, p, model, model_id=args.model)
                    logits = prefill(model, cache, seg["tail"])
                mx.eval(logits)
                ttft = time.perf_counter() - t0
                gen_seed = zlib.crc32(f"{tag}:rag:{idx}:{seed}".encode())
                ids = generate(model, cache, logits, args.max_tokens, eos, sampler=sampler, seed=gen_seed)
                answer = tok.decode(ids)
                g = grade_item(answer, it)
                row = {"model": args.model, "tag": tag, "layout": "rag", "defended": False, "item": idx,
                       "doc": it.name, "injection": it.injection, "depth": it.depth, "canary": it.canary,
                       "target_pos": idx % args.k, "k": args.k, "cond": cond, "seed": seed,
                       "chat_format": fmt.name, **g, "gen_tokens": len(ids), "ttft_s": ttft,
                       "recompute_fraction": frac, "answer": answer}
                f.write(json.dumps(row) + "\n")
                f.flush()
                print(f"[{idx + 1}/{len(items)}] {it.name:<26} {(it.injection or 'clean'):<8} pos{idx % args.k} "
                      f"s{seed} {cond:<5} complied={int(g['complied'])} correct={int(g['correct'])}", flush=True)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
