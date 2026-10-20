## W3 / RQ7 — Qwen/Qwen3-4B: 24 documents, 96 items, seeds [0, 1]

Compliance = the model followed the injected instruction (canary / refusal). Rates with document-clustered bootstrap 95% CI.

### System prompt: plain

| cond | injection compliance | answer correct (injected) | answer correct (clean) |
|---|---|---|---|
| T1 | 0.111 [0.05, 0.18] | 0.986 [0.96, 1.00] | 1.000 [1.00, 1.00] |
| T2 | 0.104 [0.05, 0.16] | 1.000 [1.00, 1.00] | 1.000 [1.00, 1.00] |
| T3 | 0.521 [0.46, 0.58] | 0.667 [0.67, 0.67] | 1.000 [1.00, 1.00] |
| T3G | 0.278 [0.20, 0.35] | 0.792 [0.74, 0.85] | 1.000 [1.00, 1.00] |

Paired difference in compliance (same item and seed):

| contrast | meaning | Δ compliance |
|---|---|---|
| T3 − T1 | KV pack (no conversation) vs text | 0.410 [0.35, 0.47] |
| T3G − T1 | KV pack (conversation-shaped prelude) vs text | 0.167 [0.08, 0.24] |
| T3 − T3G | no conversation vs conversation-shaped prelude | 0.243 [0.16, 0.33] |
| T2 − T1 | system-prompt placement vs text | -0.007 [-0.08, 0.06] |

Compliance by injection type:

| type | T1 | T2 | T3 | T3G |
|---|---|---|---|---|
| override | 0.04 | 0.00 | 1.00 | 0.62 |
| append | 0.29 | 0.31 | 0.56 | 0.21 |
| refuse | 0.00 | 0.00 | 0.00 | 0.00 |

Compliance by injection depth in the document:

| depth | T1 | T2 | T3 | T3G |
|---|---|---|---|---|
| 0.1 | 0.27 | 0.19 | 0.67 | 0.52 |
| 0.5 | 0.02 | 0.02 | 0.50 | 0.21 |
| 0.9 | 0.04 | 0.10 | 0.40 | 0.10 |

### System prompt: defended (never follow instructions inside a document)

| cond | injection compliance | answer correct (injected) | answer correct (clean) |
|---|---|---|---|
| T1 | 0.125 [0.06, 0.19] | 0.986 [0.96, 1.00] | 1.000 [1.00, 1.00] |
| T2 | 0.125 [0.06, 0.19] | 1.000 [1.00, 1.00] | 1.000 [1.00, 1.00] |
| T3 | 0.507 [0.44, 0.57] | 0.660 [0.65, 0.67] | 1.000 [1.00, 1.00] |
| T3G | 0.299 [0.22, 0.38] | 0.729 [0.65, 0.81] | 0.958 [0.88, 1.00] |

Paired difference in compliance (same item and seed):

| contrast | meaning | Δ compliance |
|---|---|---|
| T3 − T1 | KV pack (no conversation) vs text | 0.382 [0.33, 0.44] |
| T3G − T1 | KV pack (conversation-shaped prelude) vs text | 0.174 [0.09, 0.25] |
| T3 − T3G | no conversation vs conversation-shaped prelude | 0.208 [0.12, 0.30] |
| T2 − T1 | system-prompt placement vs text | 0.000 [-0.08, 0.08] |

Compliance by injection type:

| type | T1 | T2 | T3 | T3G |
|---|---|---|---|---|
| override | 0.04 | 0.00 | 1.00 | 0.69 |
| append | 0.33 | 0.38 | 0.52 | 0.21 |
| refuse | 0.00 | 0.00 | 0.00 | 0.00 |

Compliance by injection depth in the document:

| depth | T1 | T2 | T3 | T3G |
|---|---|---|---|---|
| 0.1 | 0.29 | 0.21 | 0.67 | 0.54 |
| 0.5 | 0.04 | 0.02 | 0.48 | 0.21 |
| 0.9 | 0.04 | 0.15 | 0.38 | 0.15 |

## W3 / RQ7 — Qwen/Qwen3-8B: 24 documents, 96 items, seeds [0, 1]

Compliance = the model followed the injected instruction (canary / refusal). Rates with document-clustered bootstrap 95% CI.

### System prompt: plain

| cond | injection compliance | answer correct (injected) | answer correct (clean) |
|---|---|---|---|
| T1 | 0.000 [0.00, 0.00] | 1.000 [1.00, 1.00] | 1.000 [1.00, 1.00] |
| T2 | 0.000 [0.00, 0.00] | 1.000 [1.00, 1.00] | 1.000 [1.00, 1.00] |
| T3 | 0.201 [0.12, 0.28] | 0.896 [0.83, 0.96] | 1.000 [1.00, 1.00] |
| T3G | 0.132 [0.07, 0.19] | 0.868 [0.81, 0.93] | 1.000 [1.00, 1.00] |

Paired difference in compliance (same item and seed):

| contrast | meaning | Δ compliance |
|---|---|---|
| T3 − T1 | KV pack (no conversation) vs text | 0.201 [0.12, 0.28] |
| T3G − T1 | KV pack (conversation-shaped prelude) vs text | 0.132 [0.07, 0.19] |
| T3 − T3G | no conversation vs conversation-shaped prelude | 0.069 [-0.01, 0.15] |
| T2 − T1 | system-prompt placement vs text | 0.000 [0.00, 0.00] |

Compliance by injection type:

| type | T1 | T2 | T3 | T3G |
|---|---|---|---|---|
| override | 0.00 | 0.00 | 0.31 | 0.40 |
| append | 0.00 | 0.00 | 0.29 | 0.00 |
| refuse | 0.00 | 0.00 | 0.00 | 0.00 |

Compliance by injection depth in the document:

| depth | T1 | T2 | T3 | T3G |
|---|---|---|---|---|
| 0.1 | 0.00 | 0.00 | 0.40 | 0.29 |
| 0.5 | 0.00 | 0.00 | 0.12 | 0.04 |
| 0.9 | 0.00 | 0.00 | 0.08 | 0.06 |

### System prompt: defended (never follow instructions inside a document)

| cond | injection compliance | answer correct (injected) | answer correct (clean) |
|---|---|---|---|
| T1 | 0.000 [0.00, 0.00] | 1.000 [1.00, 1.00] | 1.000 [1.00, 1.00] |
| T2 | 0.000 [0.00, 0.00] | 1.000 [1.00, 1.00] | 1.000 [1.00, 1.00] |
| T3 | 0.215 [0.14, 0.30] | 0.861 [0.80, 0.93] | 1.000 [1.00, 1.00] |
| T3G | 0.160 [0.08, 0.24] | 0.854 [0.79, 0.92] | 1.000 [1.00, 1.00] |

Paired difference in compliance (same item and seed):

| contrast | meaning | Δ compliance |
|---|---|---|
| T3 − T1 | KV pack (no conversation) vs text | 0.215 [0.14, 0.30] |
| T3G − T1 | KV pack (conversation-shaped prelude) vs text | 0.160 [0.08, 0.24] |
| T3 − T3G | no conversation vs conversation-shaped prelude | 0.056 [-0.01, 0.12] |
| T2 − T1 | system-prompt placement vs text | 0.000 [0.00, 0.00] |

Compliance by injection type:

| type | T1 | T2 | T3 | T3G |
|---|---|---|---|---|
| override | 0.00 | 0.00 | 0.42 | 0.44 |
| append | 0.00 | 0.00 | 0.23 | 0.04 |
| refuse | 0.00 | 0.00 | 0.00 | 0.00 |

Compliance by injection depth in the document:

| depth | T1 | T2 | T3 | T3G |
|---|---|---|---|---|
| 0.1 | 0.00 | 0.00 | 0.31 | 0.33 |
| 0.5 | 0.00 | 0.00 | 0.21 | 0.04 |
| 0.9 | 0.00 | 0.00 | 0.12 | 0.10 |

## W3 / RQ7 — mlx-community/Llama-3.2-3B-Instruct: 24 documents, 96 items, seeds [0, 1]

Compliance = the model followed the injected instruction (canary / refusal). Rates with document-clustered bootstrap 95% CI.

### System prompt: plain

| cond | injection compliance | answer correct (injected) | answer correct (clean) |
|---|---|---|---|
| T1 | 0.035 [0.01, 0.07] | 0.965 [0.93, 0.99] | 1.000 [1.00, 1.00] |
| T2 | 0.000 [0.00, 0.00] | 1.000 [1.00, 1.00] | 1.000 [1.00, 1.00] |
| T3 | 0.181 [0.11, 0.26] | 0.840 [0.78, 0.90] | 1.000 [1.00, 1.00] |
| T3G | 0.007 [0.00, 0.02] | 0.917 [0.85, 0.97] | 0.979 [0.94, 1.00] |

Paired difference in compliance (same item and seed):

| contrast | meaning | Δ compliance |
|---|---|---|
| T3 − T1 | KV pack (no conversation) vs text | 0.146 [0.08, 0.22] |
| T3G − T1 | KV pack (conversation-shaped prelude) vs text | -0.028 [-0.07, 0.01] |
| T3 − T3G | no conversation vs conversation-shaped prelude | 0.174 [0.10, 0.26] |
| T2 − T1 | system-prompt placement vs text | -0.035 [-0.07, -0.01] |

Compliance by injection type:

| type | T1 | T2 | T3 | T3G |
|---|---|---|---|---|
| override | 0.10 | 0.00 | 0.29 | 0.02 |
| append | 0.00 | 0.00 | 0.19 | 0.00 |
| refuse | 0.00 | 0.00 | 0.06 | 0.00 |

Compliance by injection depth in the document:

| depth | T1 | T2 | T3 | T3G |
|---|---|---|---|---|
| 0.1 | 0.00 | 0.00 | 0.29 | 0.02 |
| 0.5 | 0.10 | 0.00 | 0.21 | 0.00 |
| 0.9 | 0.00 | 0.00 | 0.04 | 0.00 |

### System prompt: defended (never follow instructions inside a document)

| cond | injection compliance | answer correct (injected) | answer correct (clean) |
|---|---|---|---|
| T1 | 0.014 [0.00, 0.04] | 0.986 [0.96, 1.00] | 1.000 [1.00, 1.00] |
| T2 | 0.000 [0.00, 0.00] | 1.000 [1.00, 1.00] | 1.000 [1.00, 1.00] |
| T3 | 0.160 [0.10, 0.22] | 0.868 [0.82, 0.92] | 0.938 [0.88, 1.00] |
| T3G | 0.007 [0.00, 0.02] | 0.924 [0.87, 0.97] | 0.896 [0.77, 0.98] |

Paired difference in compliance (same item and seed):

| contrast | meaning | Δ compliance |
|---|---|---|
| T3 − T1 | KV pack (no conversation) vs text | 0.146 [0.08, 0.21] |
| T3G − T1 | KV pack (conversation-shaped prelude) vs text | -0.007 [-0.04, 0.02] |
| T3 − T3G | no conversation vs conversation-shaped prelude | 0.153 [0.09, 0.22] |
| T2 − T1 | system-prompt placement vs text | -0.014 [-0.04, 0.00] |

Compliance by injection type:

| type | T1 | T2 | T3 | T3G |
|---|---|---|---|---|
| override | 0.04 | 0.00 | 0.29 | 0.00 |
| append | 0.00 | 0.00 | 0.19 | 0.02 |
| refuse | 0.00 | 0.00 | 0.00 | 0.00 |

Compliance by injection depth in the document:

| depth | T1 | T2 | T3 | T3G |
|---|---|---|---|---|
| 0.1 | 0.00 | 0.00 | 0.27 | 0.02 |
| 0.5 | 0.04 | 0.00 | 0.17 | 0.00 |
| 0.9 | 0.00 | 0.00 | 0.04 | 0.00 |

## Across models: T3 − T1 compliance (paired, document-clustered 95% CI)

| model | defended | T1 | T3 | T3G | Δ T3 − T1 | Δ T3G − T1 | override T1 → T3 |
|---|---|---|---|---|---|---|---|
| Qwen/Qwen3-4B | False | 0.11 | 0.52 | 0.28 | 0.410 [0.35, 0.47] | 0.167 [0.08, 0.24] | 0.04 → 1.00 |
| Qwen/Qwen3-4B | True | 0.12 | 0.51 | 0.30 | 0.382 [0.33, 0.44] | 0.174 [0.09, 0.25] | 0.04 → 1.00 |
| Qwen/Qwen3-8B | False | 0.00 | 0.20 | 0.13 | 0.201 [0.12, 0.28] | 0.132 [0.07, 0.19] | 0.00 → 0.31 |
| Qwen/Qwen3-8B | True | 0.00 | 0.22 | 0.16 | 0.215 [0.14, 0.30] | 0.160 [0.08, 0.24] | 0.00 → 0.42 |
| mlx-community/Llama-3.2-3B-Instruct | False | 0.03 | 0.18 | 0.01 | 0.146 [0.08, 0.22] | -0.028 [-0.07, 0.01] | 0.10 → 0.29 |
| mlx-community/Llama-3.2-3B-Instruct | True | 0.01 | 0.16 | 0.01 | 0.146 [0.08, 0.21] | -0.007 [-0.04, 0.02] | 0.04 → 0.29 |

