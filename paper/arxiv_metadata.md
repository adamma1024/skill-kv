# arXiv submission fields

**Upload:** `paper/arxiv.tar.gz` (built by `scripts/build_arxiv.sh`; contains `main.tex`, `main.bbl`, styles, tables, figures).
arXiv compiles `main.tex` with pdflatex and reads `main.bbl`.

**License:** CC BY 4.0 (recommended) — arXiv's "Creative Commons Attribution 4.0".

**Primary category:** cs.CR (Cryptography and Security)
**Cross-lists:** cs.LG (Machine Learning), cs.CL (Computation and Language)

**Title:**
Context-Free KV Reuse Changes What Language Models Obey

**Authors:**
Lin Ma

**Comments:**
29 pages, 4 figures. Code, data and logs: https://github.com/adamma1024/skill-kv

**Abstract** (plain text, generated from main.tex by scripts/arxiv_abstract.py; paste as is):

Serving systems increasingly splice in the key/value (KV) cache of text that was encoded outside the context in which it is later read, such as agent skills, tool schemas and retrieved documents. Such reuse is evaluated for latency and accuracy, and security work studies adversaries who poison cached state; we ask instead whether an honestly computed context-free cache changes which instructions a model obeys. In an agent layout where a tool result follows the user's request, we mount the tool result from a KV cache computed without the conversation (a pack). Across six open models (Qwen3 4B-32B, Llama-3 3B and 8B), a 4-bit control, synthetic and Wikipedia documents, and a single-step variant of InjecAgent, 14 of 20 primary model-by-dataset contrasts show a Holm-significant increase in compliance with injected instructions (up to +0.52 absolute), and none a significant decrease. The attacker needs no access to the cache, and on the three unquantized models where we test them, neither a defensive system prompt nor spotlighting removes the increase. Partial attention masks inside ordinary prefill localize the effect: a document that cannot attend to the user's turn is obeyed nearly as often as a pack, whereas one that can is obeyed as rarely as under plain prefill on five of six models. CacheBlend-style selective recompute shrinks the excess, but at about 53% of prefill compute a significant residual remains on three of five models. For trusted skills, the same encoding raises rule adherence on Qwen3-4B, but no more than placing the skill in the system prompt. Context-free KV reuse should therefore be gated on the trust level of the content.
