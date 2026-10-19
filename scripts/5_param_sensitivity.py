"""
5_param_sensitivity.py — parameter sensitivity (R1-5.4, R1-4.2). Fills tab:params.

One setting at a time, everything else at the default; one training seed per setting;
evaluated at the reference point (3B-6QC, mu=10: SWEEP_SEEDS x 3 realizations) for DQN-LNS
and ALNS (ALNS re-run because the operators / the problem change with the parameter).
  r        in {0.1, 0.2*, 0.3, 0.4}          (kappa_0 removal fraction)
  theta_s  in {0.1, 0.2, 0.3*, 0.5}          (kappa_1 threshold)
  tau grid in {(0.002,0.01), (0.005,0.02)*, (0.01,0.05)}
  alpha    in {0.25, 0.5*, 0.75}             (productivity loss at the threshold; changes the problem)
(* = default; the default row is taken from 3_benchmark.py seed 0 and is not re-run.)
Points are logged as "param:<name>=<value>" in results/runs.csv. Resumable.

Usage: python 5_param_sensitivity.py [--only r,theta_s] [--fast]
"""
import argparse, pickle, csv
import numpy as np, torch
from common import *
from interface import make_instance, make_wind, train_policy, timed_evaluate
from jos_core import search, controller, params, instances

SETTINGS = {
    "r":       [0.1, 0.3, 0.4],
    "theta_s": [0.1, 0.2, 0.5],
    "tau":     [(0.002, 0.01), (0.01, 0.05)],
    "alpha":   [0.25, 0.75],
}
EPISODES = 4000

def apply_setting(name, val):
    if name == "r": search.R_FRAC = val
    elif name == "theta_s": search.THETA_S = val
    elif name == "tau": controller.TAU[:] = [0.0, val[0], val[1]]
    elif name == "alpha": params.ALPHA = val          # instances.make_wind reads params.mu_of at construction

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--only", default=None); ap.add_argument("--fast", action="store_true")
    ap.add_argument("--threads", type=int, default=1); ap.add_argument("--values", default=None); a = ap.parse_args(); torch.set_num_threads(a.threads)
    names = a.only.split(",") if a.only else list(SETTINGS)
    cfg = SWEEP["mu10"]
    for name in names:
        for val in [v for v in SETTINGS[name] if not a.values or str(v) in a.values.split(";")]:
            point = f"param:{name}={val}"; apply_setting(name, val)
            pth = RESULTS / f"policy_{point.replace(':','_').replace('=','_')}.pkl"
            if pth.exists(): policy = pickle.load(open(pth, "rb"))
            else:
                print(f"training {point} ...", flush=True)
                policy = train_policy(0, REF, variant="full", episodes=(100 if a.fast else EPISODES))
                pickle.dump(policy, open(pth, "wb"))
            for method, pol in (("ALNS", None), ("DQN-LNS", policy)):
                done = already_done(point, method, 0)
                for inst_seed in (SWEEP_SEEDS[:3] if a.fast else SWEEP_SEEDS):
                    inst = make_instance(inst_seed, cfg)
                    for real in range(1 if a.fast else N_REAL):
                        if (inst_seed, real) in done: continue
                        wind = make_wind(wind_seed(inst_seed, real), cfg)
                        log_run(point, method, 0, inst_seed, real, timed_evaluate(method, pol, inst, wind, seed=real))
            print(f"  {point}: done", flush=True)
            # restore defaults
            search.R_FRAC, search.THETA_S, params.ALPHA = 0.2, 0.3, 0.5; controller.TAU[:] = [0.0, 0.005, 0.02]
    # table
    rows = list(csv.DictReader(open(RUNS_CSV)))
    print("\n%% tab:params rows: setting & ALNS mean & DQN-LNS mean & gap")
    pts = sorted({r["point"] for r in rows if r["point"].startswith("param:")}) + ["mu10"]
    for p in pts:
        g = lambda m: np.mean([float(r["cost"]) for r in rows if r["point"] == p and r["method"] == m and int(r["seed"]) == 0])
        try: al, dq = g("ALNS"), g("DQN-LNS"); print(f"{p} & {al:,.0f} & {dq:,.0f} & {(dq-al)/al*100:+.2f}\\% \\\\")
        except Exception: pass

if __name__ == "__main__":
    main()
