# skill-kv

Precompile SKILL.md files into KV packs and splice them into a live mlx-lm KV cache when an agent calls `load_skill`, instead of prefilling the skill text.
Design, research questions and pilot results: [docs/design.md](docs/design.md).

```bash
uv sync
uv run pytest -q                                                   # tiny random Qwen3, seconds
uv run python scripts/pilot.py --model Qwen/Qwen3-1.7B             # pilot over ~/.claude/skills
uv run python scripts/pilot.py --model Qwen/Qwen3-4B --preludes wrap,sys,full --seams 0
scripts/w1_all.sh                                                  # W1: C0-C3 task success + selection vs N (~1 h)
uv run python scripts/summarize_w1.py > outputs/w1_summary.md
```

`packs/` and `outputs/` are git-ignored. Packs are model-specific and large: about 112 KB per token for Qwen3-1.7B and 144 KB per token for Qwen3-4B (bf16).

## Paper

```bash
uv run --group paper python paper/make_assets.py   # numbers.tex, tables/, figures/ from outputs/*.jsonl
cd paper && tectonic -X compile main.tex           # -> paper/main.pdf
```
