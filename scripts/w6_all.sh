#!/usr/bin/env bash
# W6: RAG-layout control ([docs][question]) on the three models, 2 libraries each.
set -euo pipefail
cd "$(dirname "$0")/.."
for m in Qwen/Qwen3-4B Qwen/Qwen3-8B mlx-community/Llama-3.2-3B-Instruct; do
  for s in 0 1; do uv run python scripts/run_w6.py --model "$m" --seed "$s"; done
done
