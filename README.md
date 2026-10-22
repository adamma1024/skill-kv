# Context-Free KV Reuse Changes What Language Models Obey

Code, data and raw logs for the paper. Every number, table and figure in the paper is regenerated from the logs in `outputs/`
by one script.

**Finding.** In an agent loop where a tool result is read after the user's request, mounting that tool result from a KV cache
computed *without the conversation* (a "pack", as in non-prefix KV reuse) raises compliance with instructions injected into it.
The cache is computed honestly from the visible text; no cache poisoning is involved. A partial-mask ablation localizes the
increase to the missing attention from the document to the user turn. CacheBlend-style selective recompute shrinks the excess
but leaves a significant residual on three of five models at about half of prefill compute.

## Layout

| Path | What it is |
|---|---|
| `src/skillkv/pack.py` | Build, save, load and splice KV packs into an mlx-lm cache (RoPE re-rotation with the model's own RoPE). |
| `src/skillkv/blend.py` | Layer-wise selective recompute (CacheBlend-style deviation, random, first/last-k) and attention masking. |
| `src/skillkv/attn.py` | Explicit-attention profile of the last prompt token. |
| `src/skillkv/inject.py`, `injecagent.py`, `synth.py` | Injected documents (synthetic, SQuAD), InjecAgent adaptation, synthetic skills and graders. |
| `src/skillkv/stats.py` | Cluster bootstrap, two-way pigeonhole bootstrap, cluster sign-flip permutation test, Holm. |
| `scripts/run_*.py` | Experiment runners; `scripts/r*_queue.sh` are the exact run queues used for the paper. |
| `outputs/*.jsonl` | Raw per-trial logs (one JSON object per generation). |
| `paper/` | TMLR source; `paper/make_assets.py` writes `numbers.tex`, `tables/` and `figures/` from `outputs/`. |
| `data/` | SQuAD-derived documents (CC BY-SA 4.0) and InjecAgent test cases (MIT); see the READMEs there. |
| `docs/design.md` | Lab notebook (in Chinese): design decisions and results per work package. |

## Reproduce

Requires Apple silicon (MLX) and [uv](https://docs.astral.sh/uv/). Models are pulled from Hugging Face on first use.

```bash
uv sync
uv run pytest -q                                           # engine equivalence tests on tiny random models (seconds)

# one experiment, e.g. the agent-layout injection test with masks and recompute on Qwen3-4B
uv run python scripts/run_w3.py --model Qwen/Qwen3-4B --seed 0 --defended 0 --conds T1,T3,T1M,T3B15

# regenerate every number, table and figure, then build the paper
uv run --group paper python paper/make_assets.py
cd paper && tectonic -X compile main.tex                   # anonymous TMLR version
scripts/build_arxiv.sh                                     # named preprint + arXiv source tarball
```

The full set of runs is in `scripts/r1_queue.sh` ... `scripts/r8_queue.sh` (about three days on one M5 Pro, 64 GB).
`packs/` is git-ignored: packs are model-specific and large (about 144 KB per token for Qwen3-4B in bf16).

## Citation

```bibtex
@article{ma2026contextfree,
  title  = {Context-Free {KV} Reuse Changes What Language Models Obey},
  author = {Ma, Lin},
  year   = {2026},
  note   = {Preprint}
}
```

## License

Code: MIT (see `LICENSE`). Data under `data/` keeps its original licenses (SQuAD: CC BY-SA 4.0; InjecAgent: MIT).
