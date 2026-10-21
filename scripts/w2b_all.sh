#!/usr/bin/env bash
# W2 replication on Llama-3.2-3B and Qwen3-8B (same libraries, seeds and conditions as the Qwen3-4B run).
set -euo pipefail
cd "$(dirname "$0")/.."
for m in mlx-community/Llama-3.2-3B-Instruct Qwen/Qwen3-8B; do
  for s in 0 1; do uv run python scripts/run_w2.py --model "$m" --seed "$s"; done
done
