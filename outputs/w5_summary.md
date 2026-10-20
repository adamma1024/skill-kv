## W3 / RQ7 — Qwen/Qwen3-4B: 24 documents, 96 items, seeds [0, 1]

Compliance = the model followed the injected instruction (canary / refusal). Rates with document-clustered bootstrap 95% CI.

### System prompt: plain

| cond | injection compliance | answer correct (injected) | answer correct (clean) | recompute fraction |
|---|---|---|---|---|
| T1 | 0.111 [0.05, 0.18] | 0.986 [0.96, 1.00] | 1.000 [1.00, 1.00] | — |
| T3 | 0.521 [0.46, 0.58] | 0.667 [0.67, 0.67] | 1.000 [1.00, 1.00] | — |
| T1M | 0.507 [0.44, 0.57] | 0.667 [0.67, 0.67] | 1.000 [1.00, 1.00] | 1.00 |
| T3F16 | 0.500 [0.44, 0.56] | 0.667 [0.67, 0.67] | 1.000 [1.00, 1.00] | 0.01 |
| T3L16 | 0.514 [0.45, 0.58] | 0.674 [0.67, 0.69] | 1.000 [1.00, 1.00] | 0.01 |
| T3R15 | 0.451 [0.37, 0.53] | 0.708 [0.67, 0.76] | 1.000 [1.00, 1.00] | 0.15 |
| T3B15 | 0.403 [0.34, 0.47] | 0.736 [0.68, 0.79] | 1.000 [1.00, 1.00] | 0.20 |

Paired difference in compliance (same item and seed):

| contrast | meaning | Δ compliance |
|---|---|---|
| T3 − T1 | KV pack (no conversation) vs text | 0.410 [0.35, 0.47] |
| T1M − T1 | text, doc masked from the conversation (T1') vs text | 0.396 [0.34, 0.45] |
| T3 − T1M | RoPE shift: pack vs masked text at true positions | 0.014 [-0.01, 0.04] |
| T3F16 − T3 | recompute first 16 doc tokens vs pack | -0.021 [-0.04, 0.00] |
| T3L16 − T3 | recompute last 16 doc tokens vs pack | -0.007 [-0.02, 0.00] |
| T3R15 − T3 | recompute random 15% vs pack | -0.069 [-0.13, -0.01] |
| T3B15 − T3 | CacheBlend-style 15% vs pack | -0.118 [-0.18, -0.06] |
| T3B15 − T1 | CacheBlend-style 15% vs text | 0.292 [0.22, 0.36] |

Compliance by injection type:

| type | T1 | T3 | T1M | T3F16 | T3L16 | T3R15 | T3B15 |
|---|---|---|---|---|---|---|---|
| override | 0.04 | 1.00 | 1.00 | 1.00 | 0.98 | 0.88 | 0.79 |
| append | 0.29 | 0.56 | 0.52 | 0.50 | 0.56 | 0.48 | 0.42 |
| refuse | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

Compliance by injection depth in the document:

| depth | T1 | T3 | T1M | T3F16 | T3L16 | T3R15 | T3B15 |
|---|---|---|---|---|---|---|---|
| 0.1 | 0.27 | 0.67 | 0.67 | 0.67 | 0.67 | 0.67 | 0.60 |
| 0.5 | 0.02 | 0.50 | 0.48 | 0.46 | 0.50 | 0.38 | 0.35 |
| 0.9 | 0.04 | 0.40 | 0.38 | 0.38 | 0.38 | 0.31 | 0.25 |

## W3 / RQ7 — Qwen/Qwen3-8B: 24 documents, 96 items, seeds [0, 1]

Compliance = the model followed the injected instruction (canary / refusal). Rates with document-clustered bootstrap 95% CI.

### System prompt: plain

| cond | injection compliance | answer correct (injected) | answer correct (clean) | recompute fraction |
|---|---|---|---|---|
| T1 | 0.000 [0.00, 0.00] | 1.000 [1.00, 1.00] | 1.000 [1.00, 1.00] | — |
| T3 | 0.201 [0.12, 0.28] | 0.896 [0.83, 0.96] | 1.000 [1.00, 1.00] | — |
| T1M | 0.222 [0.15, 0.31] | 0.854 [0.79, 0.92] | 1.000 [1.00, 1.00] | 1.00 |
| T3F16 | 0.153 [0.08, 0.23] | 0.931 [0.88, 0.98] | 1.000 [1.00, 1.00] | 0.01 |
| T3L16 | 0.215 [0.14, 0.30] | 0.889 [0.83, 0.95] | 1.000 [1.00, 1.00] | 0.01 |
| T3R15 | 0.167 [0.10, 0.25] | 0.917 [0.86, 0.97] | 1.000 [1.00, 1.00] | 0.15 |
| T3B15 | 0.090 [0.04, 0.15] | 0.917 [0.86, 0.97] | 1.000 [1.00, 1.00] | 0.20 |

Paired difference in compliance (same item and seed):

| contrast | meaning | Δ compliance |
|---|---|---|
| T3 − T1 | KV pack (no conversation) vs text | 0.201 [0.12, 0.28] |
| T1M − T1 | text, doc masked from the conversation (T1') vs text | 0.222 [0.15, 0.31] |
| T3 − T1M | RoPE shift: pack vs masked text at true positions | -0.021 [-0.08, 0.03] |
| T3F16 − T3 | recompute first 16 doc tokens vs pack | -0.049 [-0.10, -0.01] |
| T3L16 − T3 | recompute last 16 doc tokens vs pack | 0.014 [0.00, 0.03] |
| T3R15 − T3 | recompute random 15% vs pack | -0.035 [-0.07, -0.01] |
| T3B15 − T3 | CacheBlend-style 15% vs pack | -0.111 [-0.19, -0.02] |
| T3B15 − T1 | CacheBlend-style 15% vs text | 0.090 [0.04, 0.15] |

Compliance by injection type:

| type | T1 | T3 | T1M | T3F16 | T3L16 | T3R15 | T3B15 |
|---|---|---|---|---|---|---|---|
| override | 0.00 | 0.31 | 0.44 | 0.21 | 0.33 | 0.25 | 0.25 |
| append | 0.00 | 0.29 | 0.23 | 0.25 | 0.31 | 0.25 | 0.02 |
| refuse | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

Compliance by injection depth in the document:

| depth | T1 | T3 | T1M | T3F16 | T3L16 | T3R15 | T3B15 |
|---|---|---|---|---|---|---|---|
| 0.1 | 0.00 | 0.40 | 0.46 | 0.31 | 0.42 | 0.31 | 0.23 |
| 0.5 | 0.00 | 0.12 | 0.06 | 0.08 | 0.15 | 0.10 | 0.00 |
| 0.9 | 0.00 | 0.08 | 0.15 | 0.06 | 0.08 | 0.08 | 0.04 |

## W3 / RQ7 — mlx-community/Llama-3.2-3B-Instruct: 24 documents, 96 items, seeds [0, 1]

Compliance = the model followed the injected instruction (canary / refusal). Rates with document-clustered bootstrap 95% CI.

### System prompt: plain

| cond | injection compliance | answer correct (injected) | answer correct (clean) | recompute fraction |
|---|---|---|---|---|
| T1 | 0.035 [0.01, 0.07] | 0.965 [0.93, 0.99] | 1.000 [1.00, 1.00] | — |
| T3 | 0.181 [0.11, 0.26] | 0.840 [0.78, 0.90] | 1.000 [1.00, 1.00] | — |
| T1M | 0.250 [0.15, 0.35] | 0.785 [0.71, 0.85] | 1.000 [1.00, 1.00] | 1.00 |
| T3F16 | 0.181 [0.11, 0.26] | 0.840 [0.78, 0.90] | 1.000 [1.00, 1.00] | 0.01 |
| T3L16 | 0.167 [0.10, 0.25] | 0.833 [0.77, 0.89] | 1.000 [1.00, 1.00] | 0.01 |
| T3R15 | 0.167 [0.10, 0.25] | 0.826 [0.76, 0.89] | 1.000 [1.00, 1.00] | 0.15 |
| T3B15 | 0.118 [0.06, 0.19] | 0.882 [0.83, 0.93] | 0.979 [0.94, 1.00] | 0.21 |

Paired difference in compliance (same item and seed):

| contrast | meaning | Δ compliance |
|---|---|---|
| T3 − T1 | KV pack (no conversation) vs text | 0.146 [0.08, 0.22] |
| T1M − T1 | text, doc masked from the conversation (T1') vs text | 0.215 [0.13, 0.31] |
| T3 − T1M | RoPE shift: pack vs masked text at true positions | -0.069 [-0.12, -0.01] |
| T3F16 − T3 | recompute first 16 doc tokens vs pack | 0.000 [0.00, 0.00] |
| T3L16 − T3 | recompute last 16 doc tokens vs pack | -0.014 [-0.04, 0.00] |
| T3R15 − T3 | recompute random 15% vs pack | -0.014 [-0.05, 0.01] |
| T3B15 − T3 | CacheBlend-style 15% vs pack | -0.062 [-0.11, -0.01] |
| T3B15 − T1 | CacheBlend-style 15% vs text | 0.083 [0.03, 0.15] |

Compliance by injection type:

| type | T1 | T3 | T1M | T3F16 | T3L16 | T3R15 | T3B15 |
|---|---|---|---|---|---|---|---|
| override | 0.10 | 0.29 | 0.46 | 0.29 | 0.29 | 0.27 | 0.27 |
| append | 0.00 | 0.19 | 0.21 | 0.19 | 0.15 | 0.17 | 0.08 |
| refuse | 0.00 | 0.06 | 0.08 | 0.06 | 0.06 | 0.06 | 0.00 |

Compliance by injection depth in the document:

| depth | T1 | T3 | T1M | T3F16 | T3L16 | T3R15 | T3B15 |
|---|---|---|---|---|---|---|---|
| 0.1 | 0.00 | 0.29 | 0.38 | 0.29 | 0.25 | 0.25 | 0.15 |
| 0.5 | 0.10 | 0.21 | 0.33 | 0.21 | 0.21 | 0.21 | 0.19 |
| 0.9 | 0.00 | 0.04 | 0.04 | 0.04 | 0.04 | 0.04 | 0.02 |

## Across models: T3 − T1 compliance (paired, document-clustered 95% CI)

| model | defended | T1 | T3 | T3G | Δ T3 − T1 | Δ T3G − T1 | override T1 → T3 |
|---|---|---|---|---|---|---|---|
| Qwen/Qwen3-4B | False | 0.11 | 0.52 | — | 0.410 [0.35, 0.47] | — | 0.04 → 1.00 |
| Qwen/Qwen3-8B | False | 0.00 | 0.20 | — | 0.201 [0.12, 0.28] | — | 0.00 → 0.31 |
| mlx-community/Llama-3.2-3B-Instruct | False | 0.03 | 0.18 | — | 0.146 [0.08, 0.22] | — | 0.10 → 0.29 |

