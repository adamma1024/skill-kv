"""Compare the hand-assembled agent-layout prompt (T1) with each tokenizer's apply_chat_template.

Prints, per model, whether token ids are identical and, if not, the first differing region in text.

  uv run python scripts/check_templates.py
"""

import json

from huggingface_hub import snapshot_download
from mlx_lm.utils import load_tokenizer

from skillkv.chat import detect_format
from skillkv.inject import doc_layout, make_items

MODELS = ["Qwen/Qwen3-4B", "mlx-community/Llama-3.2-3B-Instruct"]


def main() -> None:
    items = make_items(12, 0)
    it = next(i for i in items if i.injection == "override")
    for mid in MODELS:
        tok = load_tokenizer(snapshot_download(mid, allow_patterns=["*.json", "*.txt", "*.model", "*.jinja"]))
        fmt = detect_format(tok)
        enc = lambda s: tok.encode(s, add_special_tokens=False)  # noqa: E731
        L = doc_layout(enc, items, fmt=fmt)
        L.set_text(it.name, it.doc)
        ours = L.sys_lazy + L.user_turn(it.question) + L.tool_call(it.name) + L.wrap + L.skill(it.name) + L.suffix
        system = L.sys_lazy_text[len(fmt.sys_open): -len(fmt.end)]
        call = {"name": "read_doc", "arguments": {"name": it.name}}
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": it.question},
                {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": call}]},
                {"role": "tool", "content": it.doc}]
        kw = {"enable_thinking": False} if fmt.name == "qwen" else {}
        ref_text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
        ref = tok.encode(ref_text, add_special_tokens=False)
        same = ours == ref
        print(f"== {mid}: identical={same} (ours {len(ours)} tokens, template {len(ref)})")
        if not same:
            i = next((k for k, (a, b) in enumerate(zip(ours, ref)) if a != b), min(len(ours), len(ref)))
            print("   first difference at token", i)
            print("   ours    :", json.dumps(tok.decode(ours[max(0, i - 15): i + 25])))
            print("   template:", json.dumps(tok.decode(ref[max(0, i - 15): i + 25])))
            # count how many differing regions
            print("   tail ours    :", json.dumps(tok.decode(ours[-30:])))
            print("   tail template:", json.dumps(tok.decode(ref[-30:])))


if __name__ == "__main__":
    main()
