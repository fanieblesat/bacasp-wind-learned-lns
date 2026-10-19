"""
4_ablations_and_masking.py — the policy variants:
  noweather  11-feature state (R2-10, R1-6)          -> tab:weather_ablation   (wind sweep)
  op         operator-only control (6 actions)        -> tab:control            (mu 10/15/20)
  masked     feasibility-aware action masking (R3-4)  -> tab:masking            (mu 10/15/20, sig 8)
  broad      full controller trained on the broad distribution -> tab:distribution (configs + fleet sweep)
Resumable. Each variant: 5 seeds (3 for broad). Run --train jobs in parallel across machines.

Usage:  python 4_ablations_and_masking.py --variant noweather [--train|--eval] [--seed k] [--fast]
Requires 3_benchmark.py to have logged ALNS / DQN-LNS first (for the comparison columns).
"""
import argparse, pickle
import torch
from common import *
from interface import make_instance, make_wind, train_policy, timed_evaluate

EPISODES = 4000
METHOD = {"noweather": "DQN-noweather", "op": "DQN-op", "masked": "DQN-masked", "broad": "DQN-broad"}
POINTS = {"noweather": ["mu5", "mu10", "mu15", "mu20"],
          "op":        ["mu10", "mu15", "mu20"],
          "masked":    ["mu10", "mu15", "mu20", "sig8.0"],
          "broad":     list(CONFIGS) + ["V20", "V30", "V50", "V75", "V100"]}
SEEDS = {"broad": TRAIN_SEEDS[:3]}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--variant", required=True, choices=METHOD)
    ap.add_argument("--train", action="store_true"); ap.add_argument("--eval", action="store_true")
    ap.add_argument("--seed", type=int, default=None); ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--fast", action="store_true"); a = ap.parse_args(); torch.set_num_threads(a.threads)
    method = METHOD[a.variant]; everything = not (a.train or a.eval)
    seeds = [a.seed] if a.seed is not None else (TRAIN_SEEDS[:1] if a.fast else SEEDS.get(a.variant, TRAIN_SEEDS))
    for sd in seeds:
        pth = RESULTS / f"policy_{a.variant}_seed{sd}.pkl"
        if a.train or everything:
            if pth.exists(): print(f"{pth} exists")
            else:
                print(f"training {a.variant} seed {sd} ...", flush=True)
                kw = dict(broad=True, tag="broad") if a.variant == "broad" else {}
                variant = "full" if a.variant == "broad" else a.variant
                policy = train_policy(sd, REF, variant=variant, episodes=(100 if a.fast else EPISODES),
                                      probe_csv=str(RESULTS / "train_curves.csv"), **kw)
                pickle.dump(policy, open(pth, "wb"))
        if a.eval or everything:
            if not pth.exists(): print(f"missing {pth}"); continue
            policy = pickle.load(open(pth, "rb"))
            eval_method = "DQN-LNS" if a.variant == "broad" else method
            for point in (POINTS[a.variant][:1] if a.fast else POINTS[a.variant]):
                cfg = {**CONFIGS, **SWEEP}[point]; done = already_done(point, method, sd)
                inst_seeds = BENCH_SEEDS if point in CONFIGS else SWEEP_SEEDS
                for inst_seed in (inst_seeds[:3] if a.fast else inst_seeds):
                    inst = make_instance(inst_seed, cfg)
                    for real in range(1 if a.fast else N_REAL):
                        if (inst_seed, real) in done: continue
                        wind = make_wind(wind_seed(inst_seed, real), cfg)
                        log_run(point, method, sd, inst_seed, real, timed_evaluate(eval_method, policy, inst, wind, seed=real))
                print(f"  {point:<8} {method:<14} seed {sd}: done", flush=True)

if __name__ == "__main__":
    main()
