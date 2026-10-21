#!/usr/bin/env bash
# Round 6: official chat-template robustness check; starts after round 5.
set -uo pipefail
cd "$(dirname "$0")/.."
until grep -q "R5-DONE" logs/r5_queue.log 2>/dev/null; do sleep 120; done
for m in Qwen/Qwen3-4B mlx-community/Llama-3.2-3B-Instruct; do uv run python scripts/run_template_check.py --model "$m"; done
echo R6-DONE
