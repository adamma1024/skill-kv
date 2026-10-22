# arXiv submission fields

**Upload:** `paper/arxiv.tar.gz` (built by `scripts/build_arxiv.sh`; contains `main.tex`, `main.bbl`, styles, tables, figures).
arXiv compiles `main.tex` with pdflatex and reads `main.bbl`.

**License:** CC BY 4.0 (recommended) — arXiv's "Creative Commons Attribution 4.0".

**Primary category:** cs.CR (Cryptography and Security)
**Cross-lists:** cs.LG (Machine Learning), cs.CL (Computation and Language)

**Title:**
Context-Free KV Reuse Changes What Language Models Obey

**Authors:**
Adam Ma

**Comments:**
27 pages, 3 figures. Code, data and logs: https://github.com/adamma1024/skill-kv

**Abstract** (plain text, paste as is):

Serving systems increasingly reuse the key/value (KV) cache of text that was encoded outside the context in which it is later read (agent skills, tool schemas, retrieved documents) by splicing it in at a non-prefix position. Such reuse is evaluated for latency and accuracy, and recent security work studies adversaries who poison cached state. We ask a different question: does an honestly computed context-free cache change which instructions a model obeys? In an agent layout where a tool result follows the user's request, we mount the tool result from a KV cache computed without the conversation (a pack). This raises compliance with instructions injected into the tool result. We test six open models (Qwen3 4B-32B and Llama-3 3B and 8B) plus a 4-bit control, synthetic and Wikipedia documents, and a stage-1 adaptation of InjecAgent. Of 20 primary model-by-dataset contrasts, 14 show a significant increase after Holm correction, of up to +0.52; none shows a significant decrease. The attacker needs no access to the cache, and on the three bf16 models where we test them, neither a defensive system prompt nor spotlighting removes the increase. A partial-mask ablation on all six models localizes the increase to the missing attention from the document to the user's turn. CacheBlend-style selective recompute shrinks the excess, but at about 53% of prefill compute a significant residual remains on three of five models. For trusted agent skills, the same encoding raises adherence to standalone rules on Qwen3-4B, but no more than placing the skill in the system prompt does. Context-free reuse should therefore be keyed on the trust level of the content.
