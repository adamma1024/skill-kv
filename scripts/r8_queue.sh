#!/usr/bin/env bash
# Round 8 (TMLR review round 5): partial masks on Qwen3-32B and Llama-3.1-8B, question-only mask on every
# model, compute-matched random recompute; then round 7 (InjecAgent seed 1). Starts after round 6.
set -uo pipefail
cd "$(dirname "$0")/.."
R=scripts/run_w3.py
slug() { echo "${1//\//--}"; }
until grep -q "R6-DONE" logs/r6_queue.log 2>/dev/null; do sleep 120; done
for m in mlx-community/Llama-3.1-8B-Instruct mlx-community/Qwen3-32B-4bit; do
  for s in 0 1; do uv run python $R --model "$m" --seed "$s" --defended 0 --conds T1Mreq,T1Mcall --out "outputs/w10_$(slug $m)_s${s}_n12.jsonl"; done
  uv run python $R --model "$m" --items natural --defended 0 --conds T1Mreq,T1Mcall --out "outputs/w10_$(slug $m)_nat0.jsonl"
done
for m in Qwen/Qwen3-4B Qwen/Qwen3-8B mlx-community/Llama-3.2-3B-Instruct; do
  for s in 0 1; do uv run python $R --model "$m" --seed "$s" --defended 0 --conds T3R15F --out "outputs/w12_$(slug $m)_s${s}_n12.jsonl"; done
done
for m in Qwen/Qwen3-4B Qwen/Qwen3-8B mlx-community/Llama-3.2-3B-Instruct mlx-community/Qwen3-14B-4bit mlx-community/Llama-3.1-8B-Instruct mlx-community/Qwen3-32B-4bit; do
  for s in 0 1; do uv run python $R --model "$m" --seed "$s" --defended 0 --conds T1Mq --out "outputs/w11_$(slug $m)_s${s}_n12.jsonl"; done
  uv run python $R --model "$m" --items natural --defended 0 --conds T1Mq --out "outputs/w11_$(slug $m)_nat0.jsonl"
done
echo R8-DONE
for m in Qwen/Qwen3-4B Qwen/Qwen3-8B mlx-community/Llama-3.2-3B-Instruct mlx-community/Qwen3-14B-4bit mlx-community/Llama-3.1-8B-Instruct mlx-community/Qwen3-32B-4bit; do
  uv run python scripts/run_injecagent.py --model "$m" --setting enhanced --max-tokens 256 --seed 1 \
    --out "outputs/ia256_enhanced_$(slug $m)_seed1.jsonl"
done
echo R7-DONE
