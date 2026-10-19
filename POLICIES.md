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

[[PENDING: output of `python tools/compare_retrained.py --summary`, added before the repository is made public]]
