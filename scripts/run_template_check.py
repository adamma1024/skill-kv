"""Robustness check: T1 vs T3 with prompts rendered by the tokenizer's own apply_chat_template.

The full T1 prompt is rendered by the official template (system, user, assistant tool call, tool
result, generation prompt). We locate the rendered document inside it and split the text into
[before tool header][tool header][document][suffix]; the T3 pack is built after [system][tool header]
where [system] is the rendered system turn. Everything else follows run_w3.py.

  uv run python scripts/run_template_check.py --model Qwen/Qwen3-4B
"""

import argparse
import json
import zlib
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler

from skillkv.agent import generate
from skillkv.chat import detect_format
from skillkv.inject import doc_layout, grade_item, make_items, make_natural_items
from skillkv.pack import build_pack, prefill, splice

TOOL_HEADER = {"qwen": "<|im_start|>user\n<tool_response>\n", "llama3": "<|start_header_id|>ipython<|end_header_id|>\n\n"}
SYSTEM_END = {"qwen": "<|im_end|>\n", "llama3": "<|eot_id|>"}


def render(tok, fmt, system, item):
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": item.question},
            {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {
                "name": "read_doc", "arguments": {"name": item.name}}}]},
            {"role": "tool", "content": item.doc}]
    kw = {"enable_thinking": False} if fmt.name == "qwen" else {}
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
    probe = item.doc[:60]
    start = text.find(probe)
    if start < 0:  # template escapes the content (e.g. JSON-quoted)
        esc = json.dumps(item.doc, ensure_ascii=False)[1:-1]
        start = text.find(esc[:60])
        doc_r = esc
    else:
        doc_r = item.doc
    if start < 0:
        raise ValueError("could not locate the document in the rendered template")
    end = start + len(doc_r)
    if text[start:end] != doc_r:
        raise ValueError("rendered document is not contiguous in the template output")
    hdr = text.rfind(TOOL_HEADER[fmt.name], 0, start)
    sys_end = text.find(SYSTEM_END[fmt.name]) + len(SYSTEM_END[fmt.name])
    return {"system": text[:sys_end], "turn": text[sys_end:hdr], "header": text[hdr:start],
            "doc": text[start:end], "suffix": text[end:]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    slug = args.model.replace("/", "--")
    out = args.out or Path(f"outputs/tpl_{slug}.jsonl")
    model, tok = load(args.model)
    enc = lambda s: tok.encode(s, add_special_tokens=False)  # noqa: E731
    eos = set(getattr(tok, "eos_token_ids", None) or [tok.eos_token_id])
    sampler = make_sampler(temp=0.7, top_p=0.8, top_k=20)
    fmt = detect_format(tok)
    sets = [(f"s{s}_n12", make_items(12, s)) for s in (0, 1)] + [("nat0", make_natural_items("data/natural_docs.json"))]
    with out.open("w") as f:
        for tag, items in sets:
            L = doc_layout(enc, items, fmt=fmt)
            system = L.sys_lazy_text[len(fmt.sys_open): -len(fmt.end)]
            for idx, it in list(enumerate(items))[: args.limit]:
                seg = render(tok, fmt, system, it)
                ids = {k: enc(v) for k, v in seg.items()}
                pack = build_pack(model, ids["system"] + ids["header"], ids["doc"], model_id=args.model)
                for cond in ("T1", "T3"):
                    cache = make_prompt_cache(model)
                    prefill(model, cache, ids["system"])
                    prefill(model, cache, ids["turn"] + ids["header"])
                    if cond == "T1":
                        prefill(model, cache, ids["doc"])
                    else:
                        splice(cache, pack, model, model_id=args.model)
                    logits = prefill(model, cache, ids["suffix"])
                    mx.eval(logits)
                    gen_seed = zlib.crc32(f"tpl:{tag}:{idx}".encode())
                    out_ids = generate(model, cache, logits, 160, eos, sampler=sampler, seed=gen_seed)
                    answer = tok.decode(out_ids)
                    g = grade_item(answer, it)
                    f.write(json.dumps({"model": args.model, "tag": tag, "item": idx, "doc": it.name,
                                        "injection": it.injection, "cond": cond, "seed": 0, "template": "official",
                                        **g, "gen_tokens": len(out_ids), "answer": answer}) + "\n")
                    f.flush()
                print(f"{tag} {idx}", flush=True)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
