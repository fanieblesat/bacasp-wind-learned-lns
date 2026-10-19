# Learned search control for integrated berth allocation and quay crane assignment under wind-induced disruptions

Code, data, experimental record and trained policies for the paper

> F. Niebles-Atencio and S. Voß. *Learned search control for integrated berth allocation and quay crane assignment
> under wind-induced disruptions.* Submitted to the Journal of Scheduling, 2026.

Every number, table and figure in the paper is produced by the scripts in `scripts/` from the record in
`scripts/results/`. The record was written by the same scripts from seeded runs; nothing in it was edited by hand.

**Reproducibility.** Every evaluation of a trained policy is deterministic and reproduces the logged costs exactly.
Training is deterministic on a given machine and software stack; across machines, floating-point differences in the
neural-network library lead to different but equally distributed policies. Twenty of the 35 policy files are the ones
that produced the record; fifteen were lost and are provided retrained, marked as such (POLICIES.md).

## Contents

```
scripts/
  jos_core/        the model package: instance and wind generators, the period-indexed simulator (the executable form
                   of the model), the exact models (CP-SAT aggregated form; crane-specific form), the six operators,
                   the classical baselines, the controller and its training (double DQN, validation-based checkpoint selection)
  interface.py     the single seam between the drivers and jos_core
  common.py        configurations, instance pools, seeds, the results logger
  0_agreement.py ... 7_figures.py   one driver per table or figure (map below)
  paper_rows.py    every LaTeX table row, from results/runs.csv
  stats_tools.py   paired tests, equivalence tests, Friedman/Nemenyi, training curves
  data/            the vessel and berth records the instances are resampled from
  results/         the experimental record (see below) and the trained policies (see POLICIES.md)
tools/verify_policies.py   checks that each original policy file reproduces the record
tools/compare_retrained.py compares the 15 retrained policies with the record (POLICIES.md)
reproduce.sh               regenerates tables and figures from the record, or verifies the policies
```

`scripts/results/` holds
`runs.csv` (one row per method x seed x instance x wind realization: cost, turnaround, violations, time; 26,520 rows),
`validation.csv` (exact validation), `train_curves.csv` (validation score of every trained policy every 200 episodes),
`anytime_traces.json` (per-iteration best-so-far traces), `behaviour.csv` (every decision of the seed-0 policy and of the
random policy), `agreement.txt` (simulator-model agreement check), and `paper_rows.tex` / `usage_rows.tex` (the generated rows).

## Paper element -> script

| Paper | Produced by | From |
|---|---|---|
| Simulator-model agreement (Sec. 5) | `0_agreement.py` | runs the exact model |
| Table `tab:validation` | `1_validate_multiberth.py` | `results/validation.csv` |
| Table `tab:time`, Fig. `fig_anytime` | `2_anytime.py` (`--from-json` to rebuild from the record) | `results/anytime_traces.json` |
| Tables `tab:main`, `tab:wind`, `tab:fleet`, `tab:ablation`, `tab:weather_ablation`, `tab:control`, `tab:masking`, `tab:distribution`, `tab:seeds`, `tab:params` | `paper_rows.py` | `results/runs.csv` |
| Table `tab:friedman`, Figs. `fig_regime`, `fig_decomp`, `fig_cd` | `stats_tools.py --all`, `7_figures.py` | `results/runs.csv` |
| Table `tab:usage`, Fig. `fig_action_usage` | `6_behaviour.py` (`--from-csv`) | `results/behaviour.csv` |
| Fig. `fig_training` | `stats_tools.py --curves` | `results/train_curves.csv` |
| Runs behind the record | `3_benchmark.py`, `4_ablations_and_masking.py`, `5_param_sensitivity.py` | — |

## Installation

Python 3.10-3.13. Tested with numpy 2.4, scipy 1.17, pandas 3.0, matplotlib 3.10, torch 2.11-2.14 (CPU), ortools 9.15.6755.

```
pip install -r requirements.txt
```

## Reproducing the paper

```
./reproduce.sh record   # minutes: every table row and every figure, regenerated from scripts/results/
./reproduce.sh verify   # ~30 min: each trained policy re-evaluated on logged runs; must print ALL POLICIES REPRODUCE THE RECORD
```

To re-run the experiments themselves (about 120 CPU-thread-hours; one thread per job, several jobs in parallel), work in
a fresh clone or move `scripts/results/runs.csv` aside first, since the drivers append to it and skip logged runs:

```
cd scripts
python 0_agreement.py                                        # every line must read AGREE
python 3_benchmark.py --train --seed k                       # k = 0..4, the five DQN-LNS policies
python 3_benchmark.py --baselines                            # Greedy, Std-LNS, ALNS, ALNS-SA, DQN-R
python 3_benchmark.py --eval --seed k                        # k = 0..4
python 4_ablations_and_masking.py --variant {noweather,op,masked} --seed k     # k = 0..4 (train, then evaluate)
python 4_ablations_and_masking.py --variant broad --seed k                     # k = 0..2
python 5_param_sensitivity.py                                # parameter study (one seed per setting)
python 1_validate_multiberth.py --workers 8                  # exact validation (CP-SAT, up to 1 h per instance)
python 2_anytime.py; python 6_behaviour.py                   # need policy_full_seed0.pkl
```

Evaluation of a given policy is deterministic: the same policy file gives the same cost on any machine we used.
Training is seeded, but floating-point non-determinism in PyTorch means that retraining on a different machine or
library version can give a slightly different policy; the trained policies are therefore provided (POLICIES.md).

## Data

`scripts/data/` contains the vessel-call record (arrival times, terminal, vessel class) from which instances are
resampled, and the berth data. [[FILL: source of the arrival records and its terms of use.]] Handling times and all
other instance attributes are generated (see Sec. 5 of the paper and `jos_core/instances.py`).

## License and citation

Code: MIT (see `LICENSE`). Please cite the paper (see `CITATION.cff`).
