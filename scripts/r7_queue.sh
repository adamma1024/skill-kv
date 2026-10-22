#!/usr/bin/env bash
# Round 7 (TMLR review M5): second sampling seed for InjecAgent at 256 tokens; starts after round 6.
set -uo pipefail
cd "$(dirname "$0")/.."
until grep -q "R6-DONE" logs/r6_queue.log 2>/dev/null; do sleep 120; done
for m in Qwen/Qwen3-4B Qwen/Qwen3-8B mlx-community/Llama-3.2-3B-Instruct mlx-community/Qwen3-14B-4bit mlx-community/Qwen3-32B-4bit; do
  uv run python scripts/run_injecagent.py --model "$m" --setting enhanced --max-tokens 256 --seed 1 \
    --out "outputs/ia256_enhanced_${m//\//--}_seed1.jsonl"
done
echo R7-DONE
