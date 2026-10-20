#!/usr/bin/env bash
# W3 / RQ7 injection test, Qwen3-4B, 2 libraries.
set -euo pipefail
cd "$(dirname "$0")/.."
for s in 0 1; do uv run python scripts/run_w3.py --model Qwen/Qwen3-4B --seed "$s"; done
