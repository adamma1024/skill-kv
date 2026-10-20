## W2 — Qwen/Qwen3-4B: 24 skills, 48 tasks × seeds [0, 1, 2], T=0.7 top-p=0.8 top-k=20, max_tokens=512

Rates with skill-clustered bootstrap 95% CI.

| cond | n | all-5 pass | constraint score | truncated | all-5 pass, untruncated only | median gen tok | median TTFT s |
|---|---|---|---|---|---|---|---|
| C1 | 144 | 0.014 [0.00, 0.04] | 0.667 [0.64, 0.70] | 0.00 | 0.014 [0.00, 0.04] | 156 | 1.216 |
| C2 | 144 | 0.493 [0.29, 0.67] | 0.894 [0.85, 0.93] | 0.00 | 0.493 [0.29, 0.67] | 203 | 0.154 |
| C3 | 144 | 0.701 [0.56, 0.83] | 0.926 [0.90, 0.96] | 0.00 | 0.701 [0.56, 0.83] | 210 | 0.089 |
| C1M | 144 | 0.875 [0.75, 0.99] | 0.975 [0.95, 1.00] | 0.00 | 0.875 [0.75, 0.99] | 134 | 1.165 |
| C3M | 144 | 0.938 [0.83, 1.00] | 0.988 [0.97, 1.00] | 0.00 | 0.938 [0.83, 1.00] | 188 | 0.086 |
| C3G | 144 | 0.035 [0.01, 0.07] | 0.662 [0.62, 0.70] | 0.00 | 0.035 [0.01, 0.07] | 176 | 0.087 |

Paired differences (same task and sampling seed), skill-clustered bootstrap 95% CI:

| contrast | meaning | n pairs | Δ all-5 pass | Δ constraint score | Δ all-5 pass, both untruncated |
|---|---|---|---|---|---|
| C3 − C1 | approximate splice vs exact text | 144 | 0.688 [0.55, 0.81] | 0.260 [0.23, 0.29] | 0.688 [0.55, 0.81] |
| C2 − C1 | exact sys-prefix vs exact text | 144 | 0.479 [0.31, 0.66] | 0.228 [0.18, 0.28] | 0.479 [0.31, 0.66] |
| C3 − C2 | approximate splice vs exact sys-prefix | 144 | 0.208 [0.01, 0.41] | 0.032 [-0.01, 0.08] | 0.208 [0.01, 0.41] |
| C1M − C1 | naming every item in the question (text) | 144 | 0.861 [0.73, 0.98] | 0.308 [0.28, 0.34] | 0.861 [0.73, 0.98] |
| C3M − C1M | splice vs text when every item is named | 144 | 0.062 [-0.10, 0.22] | 0.012 [-0.02, 0.04] | 0.062 [-0.10, 0.22] |
| C3G − C1 | pack saw a content-free question | 144 | 0.021 [-0.03, 0.07] | -0.004 [-0.05, 0.05] | 0.021 [-0.03, 0.07] |
| C3 − C3G | no question at all vs content-free question | 144 | 0.667 [0.53, 0.78] | 0.264 [0.22, 0.31] | 0.667 [0.53, 0.78] |

Share of the C3−C1 gap closed by naming every item (C1M): 125%

Constraint pass rate by type:

| type | C1 | C2 | C3 | C1M | C3M | C3G |
|---|---|---|---|---|---|---|
| code | 0.03 | 0.55 | 0.84 | 0.88 | 1.00 | 0.22 |
| filename | 1.00 | 0.99 | 0.93 | 0.99 | 1.00 | 0.98 |
| param | 1.00 | 0.97 | 0.93 | 1.00 | 0.94 | 0.98 |
| heading | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.75 |
| signoff | 0.31 | 0.97 | 0.93 | 1.00 | 1.00 | 0.38 |
