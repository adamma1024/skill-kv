"""Pilot: is splicing a precompiled SKILL.md KV pack fast and faithful compared with prefilling
the skill text in context (progressive disclosure)?

Context layout (identical token ids in every condition, built by concatenating segment ids):
  [sys + skill catalog] [user query + assistant load_skill call] [<tool_response>] [SKILL.md] [</tool_response> + assistant]
Conditions:
  text            prefill SKILL.md in context (reference)
  pack:<prelude>  splice a precomputed pack; prelude = what the pack saw before SKILL.md:
                  none | wrap (<tool_response> header) | sys (sys+catalog+header, i.e. all but the turn)
                  | full (the exact context: sanity control, must match text)
  +seam<k>        re-prefill the last k pack tokens in the real context
Oracle skill selection (the selection call is the same in every condition, so it is excluded).

  uv run python scripts/pilot.py --model Qwen/Qwen3-1.7B --skills-dir ~/.claude/skills
"""

import argparse
import fcntl
import json
import os
import re
import time
from pathlib import Path
from statistics import median

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache

from skillkv.pack import build_pack, load_pack, prefill, save_pack, splice

SYSTEM = ("You are a coding agent. Skills are loaded with the load_skill tool. "
          "Once a skill is loaded, follow its instructions exactly.")


def read_skills(root: Path) -> list[dict]:
    skills = []
    for f in sorted(root.glob("*/SKILL.md")):
        text = f.read_text()
        m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
        front = m.group(1) if m else ""
        name = re.search(r"^name:\s*(.+)$", front, re.M)
        desc = re.search(r"^description:\s*(.+)$", front, re.M)
        skills.append({"name": (name.group(1) if name else f.parent.name).strip().strip('"'),
                       "description": (desc.group(1) if desc else "").strip().strip('"')[:200],
                       "text": text, "chars": len(text)})
    if not skills:
        raise SystemExit(f"no */SKILL.md under {root}")
    return skills


def spread(skills: list[dict], k: int) -> list[dict]:
    by_size = sorted(skills, key=lambda s: s["chars"])
    if k >= len(by_size):
        return by_size
    return [by_size[round(i * (len(by_size) - 1) / (k - 1))] for i in range(k)]


def cold_read_seconds(path: Path) -> float:
    """Read a file bypassing the unified buffer cache (F_NOCACHE) to estimate SSD load time."""
    fd = os.open(path, os.O_RDONLY)
    try:
        fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
        t0 = time.perf_counter()
        while os.read(fd, 1 << 24):
            pass
        return time.perf_counter() - t0
    finally:
        os.close(fd)


def greedy(model, cache, first_logits, n: int, eos: set[int]) -> list[int]:
    out, y = [], mx.argmax(first_logits[0, -1]).item()
    while len(out) < n and y not in eos:
        out.append(y)
        logits = model(mx.array([[y]]), cache=cache)
        y = mx.argmax(logits[0, -1]).item()
    return out


def forced_logprobs(model, cache, first_logits, ref: list[int]) -> mx.array:
    """Log-probs at every reference position, teacher-forcing `ref`. [len(ref), V] float32."""
    parts = [first_logits[0, -1:]]
    if len(ref) > 1:
        parts.append(prefill(model, cache, ref[:-1], return_all=True)[0])
    logits = mx.concatenate(parts, axis=0).astype(mx.float32)
    return logits - mx.logsumexp(logits, axis=-1, keepdims=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--skills-dir", type=Path, default=Path.home() / ".claude/skills")
    ap.add_argument("--num-skills", type=int, default=6)
    ap.add_argument("--gen", type=int, default=96)
    ap.add_argument("--seams", default="0,32")
    ap.add_argument("--preludes", default="none,wrap,sys,full")
    ap.add_argument("--repeats", type=int, default=3, help="timing repeats (median)")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    slug = args.model.replace("/", "--")
    out = args.out or Path(f"outputs/pilot_{slug}.jsonl")
    pack_dir = Path("packs") / slug
    out.parent.mkdir(parents=True, exist_ok=True)
    pack_dir.mkdir(parents=True, exist_ok=True)

    model, tok = load(args.model)
    enc = lambda s: tok.encode(s, add_special_tokens=False)  # noqa: E731
    eos = set(tok.eos_token_ids) if hasattr(tok, "eos_token_ids") else {tok.eos_token_id}

    library = read_skills(args.skills_dir.expanduser())
    catalog = "\n".join(f"- {s['name']}: {s['description']}" for s in library)
    sys_ids = enc(f"<|im_start|>system\n{SYSTEM}\n\n# Available skills\n{catalog}<|im_end|>\n")
    wrap_ids = enc("<|im_start|>user\n<tool_response>\n")
    suffix_ids = enc("\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
    prelude_kinds = args.preludes.split(",")
    seams = [int(x) for x in args.seams.split(",")]

    # warm up kernels
    prefill(model, make_prompt_cache(model), sys_ids[:512])

    rows = []
    for s in spread(library, args.num_skills):
        seg = enc(s["text"])
        task = s["description"].split(". ")[0] or s["name"]
        call = json.dumps({"name": "load_skill", "arguments": {"name": s["name"]}})
        turn_ids = enc(f"<|im_start|>user\nUse the {s['name']} skill. Task: {task}. List the first "
                       f"three concrete steps you will take, following the skill.<|im_end|>\n"
                       f"<|im_start|>assistant\n<tool_call>\n{call}\n</tool_call><|im_end|>\n")
        base_ids = sys_ids + turn_ids + wrap_ids
        cache = make_prompt_cache(model)
        prefill(model, cache, base_ids)
        base_len = cache[0].offset

        def reset():
            trim_prompt_cache(cache, cache[0].offset - base_len)

        # reference: skill text prefilled in context
        times = []
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            first = prefill(model, cache, seg + suffix_ids)
            mx.eval(mx.argmax(first[0, -1]))
            times.append(time.perf_counter() - t0)
            reset()
        t_text = median(times)
        prefill(model, cache, seg + suffix_ids)
        ref = greedy(model, cache, first, args.gen, eos)
        reset()
        prefill(model, cache, seg + suffix_ids)
        ref_lp = forced_logprobs(model, cache, first, ref)
        reset()
        common = {"skill": s["name"], "skill_tokens": len(seg), "context_before": base_len,
                  "model": args.model, "ref_len": len(ref)}
        rows.append({**common, "cond": "text", "ttft_s": t_text})
        print(f"{s['name']}: {len(seg)} skill tokens after {base_len}; text ttft {t_text:.3f}s", flush=True)

        preludes = {"none": [], "wrap": wrap_ids, "sys": sys_ids + wrap_ids, "full": base_ids}
        for pname in prelude_kinds:
            pre = preludes[pname]
            path = pack_dir / f"{s['name']}__{pname}.safetensors"
            t0 = time.perf_counter()
            save_pack(path, build_pack(model, pre, seg, model_id=args.model))
            t_build = time.perf_counter() - t0
            t_cold = cold_read_seconds(path)
            for seam in seams:
                loads, times = [], []
                for _ in range(args.repeats):
                    t0 = time.perf_counter()
                    pack = load_pack(path)
                    mx.eval([kv for layer in pack.layers for kv in layer])
                    loads.append(time.perf_counter() - t0)
                    splice(cache, pack, model, model_id=args.model, seam=seam)
                    first_p = prefill(model, cache, seg[len(seg) - seam:] + suffix_ids)
                    mx.eval(mx.argmax(first_p[0, -1]))
                    times.append(time.perf_counter() - t0)
                    reset()
                t_load, t_pack = median(loads), median(times)
                splice(cache, pack, model, model_id=args.model, seam=seam)
                prefill(model, cache, seg[len(seg) - seam:] + suffix_ids)
                gen = greedy(model, cache, first_p, args.gen, eos)
                reset()
                splice(cache, pack, model, model_id=args.model, seam=seam)
                prefill(model, cache, seg[len(seg) - seam:] + suffix_ids)
                lp = forced_logprobs(model, cache, first_p, ref)
                reset()
                kl = mx.sum(mx.exp(ref_lp) * (ref_lp - lp), axis=-1)
                top1 = mx.argmax(lp, axis=-1) == mx.array(ref)
                match = next((i for i, (a, b) in enumerate(zip(gen, ref)) if a != b), min(len(gen), len(ref)))
                row = {**common, "cond": f"pack:{pname}+seam{seam}", "ttft_s": t_pack, "load_warm_s": t_load,
                       "load_cold_s": t_cold, "build_s": t_build, "pack_mb": pack.nbytes / 2**20,
                       "kl_mean": mx.mean(kl).item(), "kl_max": mx.max(kl).item(),
                       "top1_agree": mx.mean(top1.astype(mx.float32)).item(),
                       "greedy_match_prefix": match, "exact": gen == ref}
                rows.append(row)
                print(f"  {row['cond']:<18} ttft {t_pack:.3f}s  KL {row['kl_mean']:.4f}  "
                      f"top1 {row['top1_agree']:.3f}  match {match}/{len(ref)}  {row['pack_mb']:.0f}MB",
                      flush=True)

    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
