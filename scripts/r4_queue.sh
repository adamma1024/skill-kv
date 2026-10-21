#!/usr/bin/env bash
# Round 4: Qwen3-32B (4-bit) follow-ups; starts after round 3.
set -uo pipefail
cd "$(dirname "$0")/.."
until grep -q "R3-DONE" logs/r3_queue.log 2>/dev/null; do sleep 120; done
m=mlx-community/Qwen3-32B-4bit; slug=${m//\//--}
for s in 0 1; do uv run python scripts/run_w3.py --model $m --seed "$s" --defended 0 --conds T3B15,T3B50 \
  --out "outputs/w8_${slug}_s${s}_n12.jsonl"; done
uv run python scripts/run_attn.py --model $m
uv run python scripts/run_injecagent.py --model $m --setting enhanced
echo R4-DONE
