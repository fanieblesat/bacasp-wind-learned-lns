"""
2_anytime.py — wall-clock comparison (R2-11). Fills tab:time and fig:anytime.
Runs every method at 3B-6QC, mu=10 with per-iteration (elapsed_s, best_cost) traces,
then computes (a) mean time per run, (b) cost each method reaches within the
wall-clock budget ALNS consumed, (c) the anytime figure.

Usage: python 2_anytime.py [--fast]     (needs results/policy_full_seed0.pkl from 3_benchmark.py)
"""
import argparse, pickle, json
import numpy as np
from common import *
from interface import make_instance, make_wind, timed_evaluate

METHODS = ["Std-LNS", "ALNS", "ALNS-SA", "DQN-LNS"]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--fast", action="store_true")
    ap.add_argument("--from-json", action="store_true", help="rebuild the table rows and the figure from results/anytime_traces.json "
                    "without re-running (wall-clock times are machine-dependent; re-running reproduces costs, not times)")
    a = ap.parse_args()
    traces = {m: [] for m in METHODS}; times = {m: [] for m in METHODS}
    if a.from_json:
        traces = json.load(open(RESULTS / "anytime_traces.json"))
        times = {m: [tr[-1][0] for tr in traces[m]] for m in METHODS}
    else:
        cfg = REF; policy = pickle.load(open(RESULTS / "policy_full_seed0.pkl", "rb"))
        for inst_seed in (BENCH_SEEDS[:3] if a.fast else BENCH_SEEDS):
            inst = make_instance(inst_seed, cfg)
            for real in range(1 if a.fast else N_REAL):
                wind = make_wind(wind_seed(inst_seed, real), cfg)
                for m in METHODS:
                    r = timed_evaluate(m, policy if m == "DQN-LNS" else None, inst, wind, trace=True)
                    traces[m].append(r["trace"]); times[m].append(r["time_s"])
        json.dump(traces, open(RESULTS / "anytime_traces.json", "w"))

    # (a) time per run
    print("\n%% tab:time rows")
    t_alns = float(np.mean(times["ALNS"]))
    grid = np.linspace(0, max(np.max(v) for v in times.values()), 200)
    curves = {}
    for m in METHODS:
        # best-so-far at each grid time, averaged over runs (step-function interpolation)
        rows = []
        for tr in traces[m]:
            ts = np.array([p[0] for p in tr]); cs = np.array([p[1] for p in tr])
            rows.append([cs[ts <= g][-1] if (ts <= g).any() else cs[0] for g in grid])
        curves[m] = np.mean(rows, axis=0)
        cost_at_alns_time = float(np.interp(t_alns, grid, curves[m]))
        n_iter_at = np.mean([sum(1 for p in tr if p[0] <= t_alns) for tr in traces[m]])
        print(f"{m} & {np.mean(times[m]):.2f} $\\pm$ {np.std(times[m]):.2f} & {n_iter_at:.0f} & {cost_at_alns_time:,.0f} \\\\")

    # (c) figure
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    for m in METHODS: ax.plot(grid, curves[m], label=m, lw=1.4)
    ax.axvline(t_alns, ls="--", color="gray", lw=0.8); ax.set_xlabel("elapsed wall-clock (s)")
    ax.set_ylabel("mean best-so-far cost"); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig("fig_anytime.pdf"); print("wrote fig_anytime.pdf")

if __name__ == "__main__":
    main()
