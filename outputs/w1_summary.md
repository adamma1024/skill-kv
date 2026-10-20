## Qwen/Qwen3-1.7B  (96 tasks, libraries s0_n12, s1_n12)

Rates with skill-clustered bootstrap 95% CI.

| cond | all-5 pass (oracle skill) | constraint score | end-to-end (selection × pass) | median TTFT s | median gen tok |
|---|---|---|---|---|---|
| C0 | 0.052 [0.00, 0.12] | 0.592 [0.53, 0.65] | 0.052 | 0.223 | 158 |
| C1 | 0.000 [0.00, 0.00] | 0.294 [0.23, 0.35] | 0.000 | 0.738 | 43 |
| C2 | 0.031 [0.00, 0.09] | 0.281 [0.16, 0.41] | 0.031 | 0.194 | 24 |
| C3 | 0.000 [0.00, 0.00] | 0.306 [0.23, 0.39] | 0.000 | 0.174 | 102 |

Paired difference vs C1 (same task), skill-clustered bootstrap 95% CI:

| cond − C1 | Δ all-5 pass | Δ constraint score | TTFT speedup vs C1 (median of per-task ratios) |
|---|---|---|---|
| C0 | 0.052 [0.00, 0.12] | 0.298 [0.20, 0.39] | 3.3× |
| C2 | 0.031 [0.00, 0.09] | -0.013 [-0.13, 0.12] | 3.7× |
| C3 | 0.000 [0.00, 0.00] | 0.012 [-0.09, 0.11] | 4.6× |

Constraint pass rate by planted depth (rows) and condition (columns):

| depth | C0 | C1 | C2 | C3 |
|---|---|---|---|---|
| 0.05 | 0.55 | 0.22 | 0.29 | 0.36 |
| 0.25 | 0.65 | 0.30 | 0.19 | 0.34 |
| 0.45 | 0.55 | 0.34 | 0.38 | 0.34 |
| 0.65 | 0.58 | 0.35 | 0.31 | 0.28 |
| 0.85 | 0.62 | 0.25 | 0.24 | 0.20 |

Constraint pass rate by type:

| type | C0 | C1 | C2 | C3 |
|---|---|---|---|---|
| code | 0.81 | 0.00 | 0.12 | 0.04 |
| filename | 0.60 | 0.65 | 0.33 | 0.60 |
| param | 0.90 | 0.69 | 0.44 | 0.56 |
| heading | 0.45 | 0.05 | 0.35 | 0.12 |
| signoff | 0.20 | 0.08 | 0.16 | 0.20 |

Selection on the W1 libraries: lenient 0.979, strict format 0.698

## Qwen/Qwen3-4B  (96 tasks, libraries s0_n12, s1_n12)

Rates with skill-clustered bootstrap 95% CI.

| cond | all-5 pass (oracle skill) | constraint score | end-to-end (selection × pass) | median TTFT s | median gen tok |
|---|---|---|---|---|---|
| C0 | 0.021 [0.00, 0.05] | 0.625 [0.56, 0.68] | 0.021 | 0.659 | 163 |
| C1 | 0.021 [0.00, 0.06] | 0.685 [0.65, 0.72] | 0.021 | 1.434 | 160 |
| C2 | 0.448 [0.27, 0.64] | 0.846 [0.78, 0.91] | 0.448 | 0.336 | 201 |
| C3 | 0.552 [0.39, 0.72] | 0.856 [0.78, 0.92] | 0.552 | 0.178 | 214 |

Paired difference vs C1 (same task), skill-clustered bootstrap 95% CI:

| cond − C1 | Δ all-5 pass | Δ constraint score | TTFT speedup vs C1 (median of per-task ratios) |
|---|---|---|---|
| C0 | 0.000 [-0.03, 0.03] | -0.060 [-0.13, 0.00] | 3.5× |
| C2 | 0.427 [0.25, 0.60] | 0.160 [0.11, 0.22] | 5.1× |
| C3 | 0.531 [0.38, 0.69] | 0.171 [0.11, 0.22] | 8.7× |

Constraint pass rate by planted depth (rows) and condition (columns):

| depth | C0 | C1 | C2 | C3 |
|---|---|---|---|---|
| 0.05 | 0.50 | 0.71 | 0.92 | 0.90 |
| 0.25 | 0.65 | 0.67 | 0.82 | 0.86 |
| 0.45 | 0.76 | 0.71 | 0.80 | 0.84 |
| 0.65 | 0.72 | 0.71 | 0.83 | 0.88 |
| 0.85 | 0.50 | 0.64 | 0.85 | 0.80 |

Constraint pass rate by type:

| type | C0 | C1 | C2 | C3 |
|---|---|---|---|---|
| code | 0.40 | 0.04 | 0.55 | 0.81 |
| filename | 0.85 | 1.00 | 0.91 | 0.90 |
| param | 0.90 | 1.00 | 0.90 | 0.86 |
| heading | 0.92 | 1.00 | 1.00 | 1.00 |
| signoff | 0.06 | 0.39 | 0.88 | 0.71 |

Selection on the W1 libraries: lenient 1.000, strict format 1.000

## Selection accuracy vs library size N

| model | N | catalog tokens | catalog prefill s | accuracy | strict format | median selection s |
|---|---|---|---|---|---|---|
| Qwen/Qwen3-1.7B | 12 | 369 | 0.1 | 1.000 | 0.583 | 0.39 |
| Qwen/Qwen3-1.7B | 25 | 682 | 0.1 | 1.000 | 0.680 | 0.36 |
| Qwen/Qwen3-1.7B | 50 | 1245 | 0.3 | 1.000 | 0.700 | 0.34 |
| Qwen/Qwen3-1.7B | 100 | 2389 | 0.5 | 0.980 | 0.590 | 0.36 |
| Qwen/Qwen3-4B | 12 | 369 | 0.2 | 1.000 | 1.000 | 0.73 |
| Qwen/Qwen3-4B | 25 | 682 | 0.4 | 1.000 | 1.000 | 0.82 |
| Qwen/Qwen3-4B | 50 | 1245 | 0.6 | 1.000 | 1.000 | 0.81 |
| Qwen/Qwen3-4B | 100 | 2389 | 1.4 | 0.910 | 0.910 | 0.96 |
