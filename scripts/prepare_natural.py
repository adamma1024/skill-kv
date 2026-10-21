"""Build data/natural_docs.json from SQuAD v1.1 validation (CC BY-SA 4.0): one multi-paragraph
Wikipedia document per article (consecutive paragraphs, ~550-900 words) plus one SQuAD question
whose answer lies in it.

  uv run --group data python scripts/prepare_natural.py
"""

import json
import random
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

MIN_WORDS, MAX_WORDS = 550, 900


def main() -> None:
    path = hf_hub_download("rajpurkar/squad", "plain_text/validation-00000-of-00001.parquet", repo_type="dataset")
    rows = pq.read_table(path).to_pylist()
    by_title: dict[str, list] = {}
    for r in rows:
        by_title.setdefault(r["title"], []).append(r)
    rng = random.Random(0)
    docs = []
    for title in sorted(by_title):
        paras, seen = [], set()
        for r in by_title[title]:  # dataset order = article order
            if r["context"] not in seen:
                seen.add(r["context"])
                paras.append(r["context"])
        chosen, words = [], 0
        for p in paras:
            if words >= MIN_WORDS or words + len(p.split()) > MAX_WORDS:
                break
            chosen.append(p)
            words += len(p.split())
        qs = [r for r in by_title[title] if r["context"] in chosen
              and all(a.lower() not in r["question"].lower() for a in r["answers"]["text"])
              and min(len(a.split()) for a in r["answers"]["text"]) <= 4]  # short answers grade reliably
        if not chosen or not qs:
            raise SystemExit(f"no usable question for {title!r}")
        q = rng.choice(qs)
        docs.append({"title": title.replace("_", " "), "paragraphs": chosen, "question": q["question"].strip(),
                     "answers": sorted(set(q["answers"]["text"])), "squad_id": q["id"], "words": words})
    out = Path("data/natural_docs.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(docs, indent=1, ensure_ascii=False))
    print(f"{len(docs)} documents, words {min(d['words'] for d in docs)}-{max(d['words'] for d in docs)} -> {out}")


if __name__ == "__main__":
    main()
