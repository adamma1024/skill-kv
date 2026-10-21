#!/usr/bin/env bash
# Round 1 data: recompute-ratio sweep, natural documents, larger models (4-bit).
set -uo pipefail
cd "$(dirname "$0")/.."
R=scripts/run_w3.py
slug() { echo "${1//\//--}"; }
# 1) recompute-ratio sweep (pairs with w5 T1/T3: same items, seeds and generation streams)
for m in Qwen/Qwen3-4B Qwen/Qwen3-8B mlx-community/Llama-3.2-3B-Instruct; do
  for s in 0 1; do uv run python $R --model "$m" --seed "$s" --defended 0 --conds T3B30,T3B50 \
    --out "outputs/w8_$(slug $m)_s${s}_n12.jsonl"; done
done
# 2) natural documents
for m in Qwen/Qwen3-4B Qwen/Qwen3-8B mlx-community/Llama-3.2-3B-Instruct; do
  uv run python $R --model "$m" --items natural --defended 0 --conds T1,T3,T3G,T1M,T3B15 \
    --out "outputs/w7_$(slug $m)_nat0.jsonl"
done
# 3) Qwen3-14B (4-bit): synthetic + natural
until grep -q "^done" logs/dl_14b.log; do sleep 60; done
m=mlx-community/Qwen3-14B-4bit
for s in 0 1; do uv run python $R --model $m --seed "$s" --defended 0 --conds T1,T3,T3G,T1M,T3B15 \
  --out "outputs/w5_$(slug $m)_s${s}_n12.jsonl"; done
uv run python $R --model $m --items natural --defended 0 --conds T1,T3,T3G,T1M,T3B15 --out "outputs/w7_$(slug $m)_nat0.jsonl"
# 4) Qwen3-32B (4-bit): core conditions only
m=mlx-community/Qwen3-32B-4bit
for s in 0 1; do uv run python $R --model $m --seed "$s" --defended 0 --conds T1,T3,T1M \
  --out "outputs/w5_$(slug $m)_s${s}_n12.jsonl"; done
uv run python $R --model $m --items natural --defended 0 --conds T1,T3,T1M --out "outputs/w7_$(slug $m)_nat0.jsonl"
echo QUEUE-DONE
