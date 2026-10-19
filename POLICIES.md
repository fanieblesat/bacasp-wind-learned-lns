# Trained policies

Every learned result in the paper comes from 35 trained policies. Their evaluations are recorded in
`scripts/results/runs.csv`, and every table and figure is regenerated from that record (`./reproduce.sh record`); the
policy files are not needed for that. They are provided so that any logged evaluation can be re-run.

**Originals (20 files).** These are the files that produced the record. `python tools/verify_policies.py` re-evaluates
each on logged (instance, wind realization) pairs and checks that it reproduces the logged cost exactly; evaluation of a
fixed policy is deterministic on any machine.

**Retrained (15 files).** The originals of these 15 policies were lost before this repository was assembled. They were
retrained with the same code, seed and settings (`scripts/3_benchmark.py`, `4_ablations_and_masking.py`,
`5_param_sensitivity.py`), on a different machine. Training is deterministic on a given machine and software stack, but
not across machines: floating-point differences in the neural-network library change an occasional greedy decision
during training, after which the run follows a different trajectory. A retrained file is therefore a new draw from the
same training procedure, not the file that produced the record, and it does not reproduce the logged costs.
`python tools/compare_retrained.py --summary` evaluates each retrained policy on every (point, instance, realization) on
which its original was logged and compares the two against the recorded ALNS baseline (section below).

| File | Used for | Status | SHA-256 (first 16) |
|---|---|---|---|
| `policy_full_seed0.pkl` | DQN-LNS (main method) | present, verified against the record | `37662850ed880d3b` |
| `policy_full_seed1.pkl` | DQN-LNS (main method) | present, verified against the record | `1774f51de0666b83` |
| `policy_full_seed2.pkl` | DQN-LNS (main method) | **retrained** (original lost; same code, seed and settings) | `05f7a82612c9380a` |
| `policy_full_seed3.pkl` | DQN-LNS (main method) | **retrained** (original lost; same code, seed and settings) | `69c467071e034a8f` |
| `policy_full_seed4.pkl` | DQN-LNS (main method) | present, verified against the record | `3e44f2f44d898b94` |
| `policy_fullnodecay_seed0.pkl` | constant learning rate (training-curve comparison only) | present, verified against the record | `f25b01f7ac8457fc` |
| `policy_fullnodecay_seed1.pkl` | constant learning rate (training-curve comparison only) | present, verified against the record | `5ede94f8741408b2` |
| `policy_op_seed0.pkl` | operator-only controller (tab:control) | present, verified against the record | `18ef34247d21dde4` |
| `policy_op_seed1.pkl` | operator-only controller (tab:control) | present, verified against the record | `c0db71a2037e5fe4` |
| `policy_op_seed2.pkl` | operator-only controller (tab:control) | **retrained** (original lost; same code, seed and settings) | `84eaab50881777ae` |
| `policy_op_seed3.pkl` | operator-only controller (tab:control) | **retrained** (original lost; same code, seed and settings) | `0ad72985ee45daa8` |
| `policy_op_seed4.pkl` | operator-only controller (tab:control) | **retrained** (original lost; same code, seed and settings) | `c8764675fa3715a6` |
| `policy_masked_seed0.pkl` | masked controller (tab:masking) | **retrained** (original lost; same code, seed and settings) | `58dcc62b62967eb1` |
| `policy_masked_seed1.pkl` | masked controller (tab:masking) | **retrained** (original lost; same code, seed and settings) | `7a7ff323fe34882f` |
| `policy_masked_seed2.pkl` | masked controller (tab:masking) | **retrained** (original lost; same code, seed and settings) | `5cbead55f4c2c527` |
| `policy_masked_seed3.pkl` | masked controller (tab:masking) | **retrained** (original lost; same code, seed and settings) | `8c54505893a3aaf0` |
| `policy_masked_seed4.pkl` | masked controller (tab:masking) | **retrained** (original lost; same code, seed and settings) | `af544f3cd8a947e4` |
| `policy_noweather_seed0.pkl` | no-weather state (tab:weather_ablation) | present, verified against the record | `6233232720189ce8` |
| `policy_noweather_seed1.pkl` | no-weather state (tab:weather_ablation) | present, verified against the record | `c13cc49493c01153` |
| `policy_noweather_seed2.pkl` | no-weather state (tab:weather_ablation) | present, verified against the record | `a6a1035668b2e76f` |
| `policy_noweather_seed3.pkl` | no-weather state (tab:weather_ablation) | present, verified against the record | `7dfab405c58adce1` |
| `policy_noweather_seed4.pkl` | no-weather state (tab:weather_ablation) | present, verified against the record | `3e65f32f2b18c24e` |
| `policy_broad_seed0.pkl` | broad training distribution (tab:distribution) | **retrained** (original lost; same code, seed and settings) | `16e8659361e8c447` |
| `policy_broad_seed1.pkl` | broad training distribution (tab:distribution) | **retrained** (original lost; same code, seed and settings) | `adf2c420a87b656c` |
| `policy_broad_seed2.pkl` | broad training distribution (tab:distribution) | present, verified against the record | `6b5446d48cc1eb59` |
| `policy_param_r_0.1.pkl` | parameter sensitivity (tab:params) | present, verified against the record | `37bac1a7d8e0d9f6` |
| `policy_param_r_0.3.pkl` | parameter sensitivity (tab:params) | present, verified against the record | `9fd153aa07ef07b3` |
| `policy_param_r_0.4.pkl` | parameter sensitivity (tab:params) | present, verified against the record | `27306228f7ec4363` |
| `policy_param_theta_s_0.1.pkl` | parameter sensitivity (tab:params) | **retrained** (original lost; same code, seed and settings) | `9606490552c443f3` |
| `policy_param_theta_s_0.2.pkl` | parameter sensitivity (tab:params) | **retrained** (original lost; same code, seed and settings) | `b4f2384376a863f8` |
| `policy_param_theta_s_0.5.pkl` | parameter sensitivity (tab:params) | **retrained** (original lost; same code, seed and settings) | `1bd6472d3f29c50f` |
| `policy_param_tau_(0.002, 0.01).pkl` | parameter sensitivity (tab:params) | present, verified against the record | `054a3bcea7011c43` |
| `policy_param_tau_(0.01, 0.05).pkl` | parameter sensitivity (tab:params) | present, verified against the record | `30a77a188eb6b939` |
| `policy_param_alpha_0.25.pkl` | parameter sensitivity (tab:params) | present, verified against the record | `269c397f0b4b99b3` |
| `policy_param_alpha_0.75.pkl` | parameter sensitivity (tab:params) | present, verified against the record | `f99d7ce08a044868` |

## Retrained against original

Each retrained policy was evaluated on every (point, instance, realization) on which its original was logged (6,480
runs, written to `scripts/results/retrained_runs.csv` when the tool is run; the record is not modified). Gaps are on the
penalized cost $Z_\Omega$ used by the paper's tests, so a single infeasible return dominates its row.

* **Single policies.** A retrained policy differs from its original by about as much as the original policies of the
  same variant differ among themselves. At 3B-6QC, for example, the five original DQN-LNS policies are 2.8% to 5.4% below
  ALNS; the two retrained ones (seeds 2 and 3) are 4.1% and 6.6% below. The largest single difference is an infeasible
  return of the retrained `masked` seed-0 policy at sigma_w = 8.0; the record contains the same kind of event for the
  original DQN-LNS seed-0 policy at that point.
* **The quantities the paper reports** (mean over seeds, Wilcoxon test against ALNS). Substituting the retrained files for
  the lost ones changes the sign of one of the 36 seed-mean gaps listed below (DQN-masked at sigma_w = 8.0, the infeasible
  return above) and the verdict at the 5% level in four: DQN-LNS at 5B-10QC (p = 0.097 -> 0.026) and at |V| = 20
  (p = 0.048 -> 0.097), the single-seed setting theta_s = 0.5 (p = 0.001 -> 0.114), and DQN-masked at sigma_w = 8.0
  (p < 0.001 -> 0.165). All other signs and verdicts are unchanged.

Output of `python tools/compare_retrained.py --summary`:

```
policy        point                    n  orig vs ALNS retr vs ALNS  retr vs orig  p(retr=orig)
LNS s2        2B-4QC                  50        -1.90%       +1.61%        +3.58%  0.000
LNS s3        2B-4QC                  50        -1.21%       -3.82%        -2.64%  0.003
broad s0      2B-4QC                  50        +0.08%       -1.53%        -1.60%  0.297
broad s1      2B-4QC                  50       +16.63%       +9.63%        -6.01%  0.000
LNS s2        3B-6QC                  50        -4.48%       -4.06%        +0.44%  0.776
LNS s3        3B-6QC                  50        -5.40%       -6.64%        -1.30%  0.034
broad s0      3B-6QC                  50        -4.93%       -4.34%        +0.61%  0.197
broad s1      3B-6QC                  50        +0.71%       -5.11%        -5.79%  0.000
LNS s2        4B-8QC                  50        -1.49%       -3.61%        -2.15%  0.000
LNS s3        4B-8QC                  50        -3.04%       -3.11%        -0.07%  0.889
broad s0      4B-8QC                  50        -2.61%       -1.88%        +0.75%  0.389
broad s1      4B-8QC                  50        -1.58%       -3.54%        -1.98%  0.000
LNS s2        5B-10QC                 50        +0.78%       -1.22%        -1.99%  0.000
LNS s3        5B-10QC                 50        -1.58%       -0.98%        +0.60%  0.037
broad s0      5B-10QC                 50        -0.77%       -0.90%        -0.13%  0.031
broad s1      5B-10QC                 50        -0.30%       -1.87%        -1.57%  0.000
LNS s2        V100                    20        -6.03%       -7.48%        -1.54%  0.247
LNS s3        V100                    20        -8.89%      -10.01%        -1.24%  0.135
broad s0      V100                    20        -7.84%       -7.16%        +0.74%  0.622
broad s1      V100                    20        -1.89%       -6.20%        -4.39%  0.000
LNS s2        V20                     20        +1.73%       +1.10%        -0.61%  0.641
LNS s3        V20                     20        +2.05%       +1.73%        -0.31%  0.422
broad s0      V20                     20        +3.06%       +1.24%        -1.76%  0.023
broad s1      V20                     20        +1.50%       +2.05%        +0.54%  0.324
LNS s2        V30                     20        -3.03%       -3.09%        -0.07%  0.926
LNS s3        V30                     20        -2.19%       -3.80%        -1.64%  0.091
broad s0      V30                     20        -1.87%       -3.18%        -1.33%  0.121
broad s1      V30                     20        -2.56%       -1.70%        +0.88%  0.351
LNS s2        V50                     20        -2.92%       -4.37%        -1.50%  0.083
LNS s3        V50                     20        -6.15%       -6.68%        -0.56%  0.433
broad s0      V50                     20        -5.13%       -5.36%        -0.24%  0.627
broad s1      V50                     20        -0.21%       -4.95%        -4.76%  0.002
LNS s2        V75                     20        -5.09%       -7.72%        -2.77%  0.007
LNS s3        V75                     20        -8.98%       -8.84%        +0.16%  0.881
broad s0      V75                     20        -8.90%       -7.27%        +1.80%  0.165
broad s1      V75                     20        -0.14%       -7.37%        -7.24%  0.000
LNS s2        mu10                    20        -2.92%       -4.37%        -1.50%  0.083
LNS s3        mu10                    20        -6.15%       -6.68%        -0.56%  0.433
masked s0     mu10                    20        -6.66%       -4.86%        +1.93%  0.145
masked s1     mu10                    20        -6.63%       -6.84%        -0.22%  0.779
masked s2     mu10                    20        -3.69%       +0.90%        +4.77%  0.052
masked s3     mu10                    20        -6.51%       -5.55%        +1.03%  0.330
masked s4     mu10                    20        -1.45%       -3.03%        -1.60%  0.189
op s2         mu10                    20        -3.87%       -4.06%        -0.20%  0.756
op s3         mu10                    20        -5.19%       -4.07%        +1.18%  0.330
op s4         mu10                    20        -4.85%       -0.66%        +4.40%  0.002
LNS s2        mu15                    20        -3.01%       -1.93%        +1.12%  0.498
LNS s3        mu15                    20        -5.26%       -4.97%        +0.31%  0.898
masked s0     mu15                    20        -5.54%       -0.72%        +5.11%  0.001
masked s1     mu15                    20        -2.18%       -3.68%        -1.54%  0.261
masked s2     mu15                    20        -0.98%       +2.93%        +3.94%  0.048
masked s3     mu15                    20        -5.62%       -4.93%        +0.73%  0.279
masked s4     mu15                    20        -2.04%       -3.00%        -0.98%  0.409
op s2         mu15                    20        -1.77%       -2.14%        -0.38%  0.723
op s3         mu15                    20        -3.86%       -2.44%        +1.48%  0.143
op s4         mu15                    20        -2.66%       -1.10%        +1.60%  0.198
LNS s2        mu20                    20        +0.10%       -1.37%        -1.46%  0.076
LNS s3        mu20                    20        -4.03%       -5.53%        -1.57%  0.179
masked s0     mu20                    20        -3.90%       -2.42%        +1.54%  0.322
masked s1     mu20                    20        -5.03%       -3.60%        +1.51%  0.147
masked s2     mu20                    20        -3.04%       -5.53%        -2.57%  0.024
masked s3     mu20                    20        -1.29%       -2.65%        -1.38%  0.090
masked s4     mu20                    20        -1.13%       -1.05%        +0.08%  0.927
op s2         mu20                    20        -0.74%       -0.80%        -0.06%  0.927
op s3         mu20                    20        -2.62%       -2.72%        -0.10%  0.841
op s4         mu20                    20        -2.15%       -1.30%        +0.86%  0.513
LNS s2        mu5                     20        -2.18%       -3.61%        -1.46%  0.057
LNS s3        mu5                     20        -3.36%       -4.32%        -0.99%  0.522
LNS s0        param:theta_s=0.1       20        -6.70%       -6.51%        +0.21%  0.732
LNS s0        param:theta_s=0.2       20        -5.83%       -6.14%        -0.34%  0.648
LNS s0        param:theta_s=0.5       20        -4.22%       -1.85%        +2.47%  0.003
LNS s2        sig1.5                  20        -1.88%       -3.36%        -1.51%  0.177
LNS s3        sig1.5                  20        -2.25%       -2.57%        -0.33%  0.825
LNS s2        sig3.5                  20        -2.92%       -4.37%        -1.50%  0.083
LNS s3        sig3.5                  20        -6.15%       -6.68%        -0.56%  0.433
LNS s2        sig5.5                  20        -6.35%       -5.15%        +1.28%  0.231
LNS s3        sig5.5                  20        -6.76%       -7.49%        -0.78%  0.546
LNS s2        sig8.0                  20        -5.08%       -4.26%        +0.87%  0.349
LNS s3        sig8.0                  20        -3.75%       -3.72%        +0.03%  0.898
masked s0     sig8.0                  20        -6.53%     +100.15%      +114.13%  0.001
masked s1     sig8.0                  20        -5.65%       -7.11%        -1.55%  0.048
masked s2     sig8.0                  20        -5.41%       -5.15%        +0.28%  0.701
masked s3     sig8.0                  20        -4.68%       -5.06%        -0.39%  0.614
masked s4     sig8.0                  20        -1.49%       -4.22%        -2.77%  0.027

seed-mean gap to ALNS (as reported in the paper)  original -> with retrained files substituted
  DQN-LNS    2B-4QC                  -0.86% (p=0.347)  ->   -0.68% (p=0.426)
  DQN-LNS    3B-6QC                  -4.36% (p=7.16e-09)  ->   -4.52% (p=2.24e-12)
  DQN-LNS    4B-8QC                  -2.38% (p=2.16e-05)  ->   -2.82% (p=1.14e-06)
  DQN-LNS    5B-10QC                 -0.64% (p=0.0968)  ->   -0.92% (p=0.0264)
  DQN-LNS    V100                    -7.54% (p=1.91e-06)  ->   -8.06% (p=1.91e-06)
  DQN-LNS    V20                     +1.47% (p=0.0484)  ->   +1.29% (p=0.0973)
  DQN-LNS    V30                     -3.05% (p=0.00143)  ->   -3.39% (p=0.00116)
  DQN-LNS    V50                     -4.54% (p=9.54e-06)  ->   -4.94% (p=4.77e-05)
  DQN-LNS    V75                     -6.83% (p=1.34e-05)  ->   -7.33% (p=1.91e-06)
  DQN-LNS    mu10                    -4.54% (p=9.54e-06)  ->   -4.94% (p=4.77e-05)
  DQN-LNS    mu15                    -2.93% (p=0.00199)  ->   -2.65% (p=0.00422)
  DQN-LNS    mu20                    -1.08% (p=0.245)  ->   -1.67% (p=0.123)
  DQN-LNS    mu5                     -2.78% (p=0.00233)  ->   -3.25% (p=0.00102)
  DQN-LNS    param:theta_s=0.1       -6.70% (p=3.62e-05)  ->   -6.51% (p=0.000832)
  DQN-LNS    param:theta_s=0.2       -5.83% (p=1.91e-06)  ->   -6.14% (p=5.72e-06)
  DQN-LNS    param:theta_s=0.5       -4.22% (p=0.00102)  ->   -1.85% (p=0.114)
  DQN-LNS    sig1.5                  -2.38% (p=0.00639)  ->   -2.74% (p=0.000586)
  DQN-LNS    sig3.5                  -4.54% (p=9.54e-06)  ->   -4.94% (p=4.77e-05)
  DQN-LNS    sig5.5                  +8.11% (p=0.000851)  ->   +8.20% (p=0.000708)
  DQN-LNS    sig8.0                 +16.55% (p=0.0583)  ->  +16.72% (p=0.123)
  DQN-op     mu10                    -4.51% (p=1.91e-05)  ->   -3.48% (p=0.00148)
  DQN-op     mu15                    -2.51% (p=0.0121)  ->   -1.99% (p=0.024)
  DQN-op     mu20                    +8.71% (p=0.648)  ->   +8.85% (p=0.956)
  DQN-masked mu10                    -4.99% (p=1.91e-06)  ->   -3.88% (p=6.29e-05)
  DQN-masked mu15                    -3.27% (p=0.00422)  ->   -1.88% (p=0.0206)
  DQN-masked mu20                    -2.88% (p=0.00558)  ->   -3.05% (p=0.024)
  DQN-masked sig8.0                  -4.75% (p=0.000134)  ->  +15.72% (p=0.165)
  DQN-broad  2B-4QC                  +5.48% (p=6.34e-11)  ->   +2.61% (p=0.000691)
  DQN-broad  3B-6QC                  -2.84% (p=6.35e-07)  ->   -4.59% (p=5.61e-11)
  DQN-broad  4B-8QC                  -1.88% (p=0.000261)  ->   -2.28% (p=2.03e-05)
  DQN-broad  5B-10QC                 -0.13% (p=0.515)  ->   -0.69% (p=0.128)
  DQN-broad  V100                    -5.25% (p=1.91e-06)  ->   -6.46% (p=1.91e-06)
  DQN-broad  V20                     +1.84% (p=0.0419)  ->   +1.42% (p=0.0484)
  DQN-broad  V30                     -2.31% (p=0.0333)  ->   -2.46% (p=0.000483)
  DQN-broad  V50                     -3.23% (p=0.000851)  ->   -4.89% (p=0.000851)
  DQN-broad  V75                     -5.09% (p=9.54e-06)  ->   -6.96% (p=3.81e-06)
```

