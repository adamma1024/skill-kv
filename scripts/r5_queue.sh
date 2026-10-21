#!/usr/bin/env bash
# Round 5 (TMLR review): 14B sweep, InjecAgent at 256 tokens, partial-mask ablation, spotlighting,
# 8B 4-bit quantization control, Llama-3.1-8B. Starts after round 4.
set -uo pipefail
cd "$(dirname "$0")/.."
R=scripts/run_w3.py
slug() { echo "${1//\//--}"; }
until grep -q "R4-DONE" logs/r4_queue.log 2>/dev/null; do sleep 120; done
m=mlx-community/Qwen3-14B-4bit
for s in 0 1; do uv run python $R --model $m --seed "$s" --defended 0 --conds T3B30,T3B50 --out "outputs/w8_$(slug $m)_s${s}_n12.jsonl"; done
for m in Qwen/Qwen3-4B Qwen/Qwen3-8B mlx-community/Llama-3.2-3B-Instruct mlx-community/Qwen3-14B-4bit; do
  uv run python scripts/run_injecagent.py --model "$m" --setting enhanced --max-tokens 256
done
for m in Qwen/Qwen3-4B Qwen/Qwen3-8B mlx-community/Llama-3.2-3B-Instruct mlx-community/Qwen3-14B-4bit; do
  for s in 0 1; do uv run python $R --model "$m" --seed "$s" --defended 0 --conds T1Mreq,T1Mcall --out "outputs/w10_$(slug $m)_s${s}_n12.jsonl"; done
  uv run python $R --model "$m" --items natural --defended 0 --conds T1Mreq,T1Mcall --out "outputs/w10_$(slug $m)_nat0.jsonl"
done
for m in Qwen/Qwen3-4B Qwen/Qwen3-8B mlx-community/Llama-3.2-3B-Instruct; do
  for s in 0 1; do uv run python $R --model "$m" --seed "$s" --defended 0 --spotlight --conds T1,T3 --out "outputs/w9_$(slug $m)_s${s}_n12.jsonl"; done
done
until grep -q "^done" logs/dl_l31.log 2>/dev/null; do sleep 60; done
for m in mlx-community/Qwen3-8B-4bit mlx-community/Llama-3.1-8B-Instruct; do
  for s in 0 1; do uv run python $R --model "$m" --seed "$s" --defended 0 --conds T1,T3,T1M --out "outputs/w5_$(slug $m)_s${s}_n12.jsonl"; done
  uv run python $R --model "$m" --items natural --defended 0 --conds T1,T3,T1M --out "outputs/w7_$(slug $m)_nat0.jsonl"
done
uv run python scripts/run_injecagent.py --model mlx-community/Llama-3.1-8B-Instruct --setting enhanced --max-tokens 256
echo R5-DONE
