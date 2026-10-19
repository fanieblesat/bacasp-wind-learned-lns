"""
3_benchmark.py — main campaign: five training seeds x full evaluation.
Fills: tab:main, tab:wind, tab:fleet, tab:ablation, tab:seeds, tab:time (time_s column).
Resumable: re-run after interruption; logged (instance, realization) pairs are skipped.

Usage:
  python 3_benchmark.py --fast                 # smoke test (minutes): 1 seed, 2 points, 3 instances
  python 3_benchmark.py --train --seed 0       # train ONE full policy (~5 h, 1 thread) -> results/policy_full_seed0.pkl
  python 3_benchmark.py --baselines            # classical baselines + DQN-R on every point (~2 h)
  python 3_benchmark.py --eval --seed 0        # evaluate policy seed 0 on every point (~40 min)
  python 3_benchmark.py                        # everything, sequentially (train 5 seeds, baselines, eval)
Run several --train / --eval jobs in parallel on the laptops (each uses one thread).
"""
import argparse, pickle, sys
from pathlib import Path
import torch
from common import *
from interface import make_instance, make_wind, train_policy, timed_evaluate

CLASSICAL = ["Greedy", "Std-LNS", "ALNS", "ALNS-SA"]
EPISODES = 4000

def points(fast):
    pts = {**CONFIGS, **SWEEP}
    return {k: v for k, v in list(pts.items())[:2]} if fast else pts

def seeds_for(point, fast):
    s = BENCH_SEEDS if point in CONFIGS else SWEEP_SEEDS
    return s[:3] if fast else s

def run_method(point, cfg, method, seed, policy, fast):
    done = already_done(point, method, seed)
    for inst_seed in seeds_for(point, fast):
        inst = make_instance(inst_seed, cfg)
        for real in range(1 if fast else N_REAL):
            if (inst_seed, real) in done: continue
            wind = make_wind(wind_seed(inst_seed, real), cfg)
            r = timed_evaluate(method, policy, inst, wind, budget=500, seed=real)
            log_run(point, method, seed, inst_seed, real, r)
    print(f"  {point:<8} {method:<10} seed {seed}: done", flush=True)

def train_one(sd, fast, nodecay=False):
    pth = RESULTS / (f"policy_fullnodecay_seed{sd}.pkl" if nodecay else f"policy_full_seed{sd}.pkl")
    if pth.exists():
        print(f"policy seed {sd} exists: {pth}"); return pickle.load(open(pth, "rb"))
    print(f"training full policy seed {sd} ({EPISODES} episodes{', constant LR' if nodecay else ''}) ...", flush=True)
    kw = dict(lr_final=None, tag="full-nodecay") if nodecay else {}
    policy = train_policy(sd, REF, variant="full", episodes=(100 if fast else EPISODES),
                          probe_csv=str(RESULTS / "train_curves.csv"), **kw)
    pickle.dump(policy, open(pth, "wb")); return policy

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true"); ap.add_argument("--train", action="store_true")
    ap.add_argument("--baselines", action="store_true"); ap.add_argument("--eval", action="store_true")
    ap.add_argument("--seed", type=int, default=None); ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--no-decay", action="store_true", help="constant learning rate (the pre-revision recipe; for the stability comparison only). Files: policy_fullnodecay_seed*.pkl, method DQN-LNS-nodecay")
    a = ap.parse_args(); torch.set_num_threads(a.threads)
    pts = points(a.fast); tseeds = [a.seed] if a.seed is not None else (TRAIN_SEEDS[:1] if a.fast else TRAIN_SEEDS)
    everything = not (a.train or a.baselines or a.eval)

    if a.train or everything:
        for sd in tseeds: train_one(sd, a.fast, a.no_decay)
    if a.baselines or everything:
        for point, cfg in pts.items():
            for m in CLASSICAL: run_method(point, cfg, m, 0, None, a.fast)
            for sd in tseeds: run_method(point, cfg, "DQN-R", sd, ("random", sd), a.fast)
    if a.eval or everything:
        for sd in tseeds:
            pth = RESULTS / (f"policy_fullnodecay_seed{sd}.pkl" if a.no_decay else f"policy_full_seed{sd}.pkl")
            if not pth.exists(): print(f"missing {pth}: run --train --seed {sd} first"); continue
            policy = pickle.load(open(pth, "rb"))
            print(f"policy seed {sd}: best validation {policy['best_val']:.4f} at episode {policy['best_episode']} "
                  f"(trained {policy['train_time_s']/3600:.1f} h)")
            for point, cfg in pts.items(): run_method(point, cfg, "DQN-LNS-nodecay" if a.no_decay else "DQN-LNS", sd, policy, a.fast)
    print(f"\nDone. Now: python stats_tools.py {RUNS_CSV} --all > tables.tex")

if __name__ == "__main__":
    main()
