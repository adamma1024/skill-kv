#!/usr/bin/env bash
# W1 batch: task success C0-C3 (2 library seeds x 4 tasks/skill) and selection-vs-N, both models.
set -euo pipefail
cd "$(dirname "$0")/.."
for m in Qwen/Qwen3-4B Qwen/Qwen3-1.7B; do
  for s in 0 1; do
    uv run python scripts/run_w1.py --model "$m" --seed "$s" --tasks-per-skill 4
  done
  uv run python scripts/run_selection.py --model "$m"
done
