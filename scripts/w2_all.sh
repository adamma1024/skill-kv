#!/usr/bin/env bash
# W2: C1/C2/C3 + C1M/C3M/C3G ablations, Qwen3-4B, 2 libraries x 24 tasks x 3 sampling seeds, max 512 tokens.
set -euo pipefail
cd "$(dirname "$0")/.."
for s in 0 1; do
  uv run python scripts/run_w2.py --model Qwen/Qwen3-4B --seed "$s"
done
