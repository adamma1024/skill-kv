"""Round 3: where does the first answer token attend under T1 / T3 / T1M?

For each injected item (synthetic and natural documents), build the context up to the last
suffix token, then run that token with explicit attention (skillkv.attn) and record the
head-mean attention mass on: system prompt, user turn + tool call, tool-result header,
injected instruction, rest of the document, suffix.

  uv run python scripts/run_attn.py --model Qwen/Qwen3-4B
"""

import argparse
import json
from pathlib import Path
from statistics import mean

from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

from skillkv.attn import attention_profile, span_mass
from skillkv.blend import blend
from skillkv.chat import detect_format
from skillkv.inject import doc_layout, make_items, make_natural_items
from skillkv.pack import build_pack, prefill, splice

CONDS = ("T1", "T3", "T1M")


def token_span(tok, text: str, ids: list[int], start: int, end: int) -> tuple[int, int]:
    """Token indices [lo, hi) of `ids` (= encoding of `text`) covering characters [start, end)."""
    enc = tok._tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    if list(enc["input_ids"]) != list(ids):
        raise ValueError("offset tokenization differs from the encoding used for the prompt")
    offs = enc["offset_mapping"]
    lo = next(i for i, (a, b) in enumerate(offs) if b > start)
    hi = next((i for i, (a, b) in enumerate(offs) if a >= end), len(offs))
    return lo, hi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    slug = args.model.replace("/", "--")
    out = args.out or Path(f"outputs/attn_{slug}.jsonl")
    model, tok = load(args.model)
    enc = lambda s: tok.encode(s, add_special_tokens=False)  # noqa: E731
    fmt = detect_format(tok)
    sets = [(f"s{s}_n12", make_items(12, s)) for s in (0, 1)] + [("nat0", make_natural_items("data/natural_docs.json"))]
    with out.open("w") as f:
        for tag, items in sets:
            L = doc_layout(enc, items, defended=False, fmt=fmt)
            todo = [(i, it) for i, it in enumerate(items) if it.injection][: args.limit]
            for idx, it in todo:
                L.set_text(it.name, it.doc)
                doc = L.skill(it.name)
                turn = L.user_turn(it.question) + L.tool_call(it.name)
                s0 = it.doc.index(it.injection_text)
                ilo, ihi = token_span(tok, it.doc, doc, s0, s0 + len(it.injection_text))
                n_sys, n_turn, n_wrap = len(L.sys_lazy), len(turn), len(L.wrap)
                d0 = n_sys + n_turn + n_wrap
                spans = {"system": (0, n_sys), "user_turn": (n_sys, n_sys + n_turn), "header": (n_sys + n_turn, d0),
                         "injection": (d0 + ilo, d0 + ihi), "doc_before": (d0, d0 + ilo),
                         "doc_after": (d0 + ihi, d0 + len(doc)), "suffix": (d0 + len(doc), d0 + len(doc) + len(L.suffix))}
                pack = build_pack(model, L.pack_prelude, doc, model_id=args.model)
                for cond in CONDS:
                    cache = make_prompt_cache(model)
                    prefill(model, cache, L.sys_lazy)
                    if cond == "T1":
                        prefill(model, cache, turn + L.wrap + doc)
                    elif cond == "T3":
                        prefill(model, cache, turn + L.wrap)
                        splice(cache, pack, model, model_id=args.model)
                    else:
                        prefill(model, cache, turn)
                        blend(model, cache, L.wrap + doc, select="all", exclude=(n_sys, n_sys + n_turn))
                    prefill(model, cache, L.suffix[:-1])
                    probs, _ = attention_profile(model, cache, L.suffix[-1])
                    mass = span_mass(probs, spans)
                    row = {"model": args.model, "tag": tag, "item": idx, "doc": it.name, "injection": it.injection,
                           "depth": it.depth, "cond": cond, "chat_format": fmt.name,
                           "mean": {k: mean(v) for k, v in mass.items()}, "per_layer": mass,
                           "spans": spans}
                    f.write(json.dumps(row) + "\n")
                    f.flush()
                print(f"{tag} {idx:3d} {it.injection:<8} " + "  ".join(
                    f"{c}" for c in CONDS), flush=True)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
