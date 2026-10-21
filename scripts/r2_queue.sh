#!/usr/bin/env bash
# Round 2: InjecAgent (enhanced setting) under T1/T3/T1M/T3B15; starts after the round-1 queue.
set -uo pipefail
cd "$(dirname "$0")/.."
until grep -q "QUEUE-DONE" logs/r1_queue.log 2>/dev/null; do sleep 120; done
for m in Qwen/Qwen3-4B Qwen/Qwen3-8B mlx-community/Llama-3.2-3B-Instruct mlx-community/Qwen3-14B-4bit; do
  uv run python scripts/run_injecagent.py --model "$m" --setting enhanced
done
echo R2-DONE
