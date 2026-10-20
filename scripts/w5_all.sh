#!/usr/bin/env bash
# W5: mitigations (selective recompute) and the T1' attention-mask control, plain system prompt.
set -euo pipefail
cd "$(dirname "$0")/.."
C=T1,T3,T1M,T3F16,T3L16,T3R15,T3B15
for m in Qwen/Qwen3-4B Qwen/Qwen3-8B mlx-community/Llama-3.2-3B-Instruct; do
  slug=${m//\//--}
  for s in 0 1; do
    uv run python scripts/run_w3.py --model "$m" --seed "$s" --defended 0 --conds "$C" \
      --out "outputs/w5_${slug}_s${s}_n12.jsonl"
  done
done
