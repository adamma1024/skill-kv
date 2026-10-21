#!/usr/bin/env bash
# Round 3: attention analysis; starts after round 2.
set -uo pipefail
cd "$(dirname "$0")/.."
until grep -q "R2-DONE" logs/r2_queue.log 2>/dev/null; do sleep 120; done
for m in Qwen/Qwen3-4B Qwen/Qwen3-8B mlx-community/Llama-3.2-3B-Instruct mlx-community/Qwen3-14B-4bit; do
  uv run python scripts/run_attn.py --model "$m"
done
echo R3-DONE
