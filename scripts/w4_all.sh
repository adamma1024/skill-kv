#!/usr/bin/env bash
# W4: replicate the W3 injection test on Llama-3.2-3B (MLX mirror) and Qwen3-8B, 2 libraries each.
set -euo pipefail
cd "$(dirname "$0")/.."
for s in 0 1; do uv run python scripts/run_w3.py --model mlx-community/Llama-3.2-3B-Instruct --seed "$s"; done
until grep -q "^done" logs/dl_qwen8b.log; do sleep 30; done   # Qwen3-8B download runs separately
for s in 0 1; do uv run python scripts/run_w3.py --model Qwen/Qwen3-8B --seed "$s"; done
