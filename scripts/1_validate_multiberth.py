"""
1_validate_multiberth.py — exact validation on multi-berth instances (R2-7, R3-3). Fills tab:validation.

For |V| in {6, 8, 10}, 10 instances each (VALID_SEEDS), |B|=3, |K|=6, T=48, one wind realization:
  1. solve the exact (aggregated) model with CP-SAT, warm-started from Greedy; record status, bound, gap;
  2. replay the solver's variables in the simulator (agreement must be exact);
  3. run Greedy, Std-LNS, ALNS, ALNS-SA and DQN-LNS (every trained seed in results/) at I=500;
     report the gap of each to the best known solver value (optimal when proven).
Resumable per instance (results/validation.csv).

Usage: python 1_validate_multiberth.py [--time-limit 3600] [--workers 8] [--fast]
"""
import argparse, pickle, csv, math
import numpy as np
from common import *
from interface import make_instance, make_wind, timed_evaluate, schedule_from_solution
from jos_core.simulator import to_mip_variables
from jos_core.search import greedy_initial
from jos_core.exact_cpsat import solve_cpsat

OUT = RESULTS / "validation.csv"
FIELDS = ["nV", "seed", "status", "obj", "bound", "gap_pct", "time_s", "agree", "Greedy", "Std-LNS", "ALNS", "ALNS-SA", "DQN_mean", "DQN_worst", "DQN_n"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--time-limit", type=float, default=3600)
    ap.add_argument("--workers", type=int, default=8); ap.add_argument("--fast", action="store_true"); a = ap.parse_args()
    T = 48
    done = set()
    if OUT.exists():
        for row in csv.DictReader(open(OUT)): done.add((int(row["nV"]), int(row["seed"])))
    policies = {sd: pickle.load(open(RESULTS / f"policy_full_seed{sd}.pkl", "rb")) for sd in TRAIN_SEEDS
                if (RESULTS / f"policy_full_seed{sd}.pkl").exists()}
    if not policies: print("NOTE: no trained policies in results/ — DQN-LNS columns will be empty; rerun after 3_benchmark.py")
    for nV in ([6] if a.fast else [6, 8, 10]):
        for seed in (VALID_SEEDS[:2] if a.fast else VALID_SEEDS):
            if (nV, seed) in done: continue
            cfg = dict(REF, n_vessels=nV, horizon=T); inst = make_instance(seed, cfg); wind = make_wind(wind_seed(seed, 0), cfg)
            x0 = greedy_initial(inst, wind); hx, hs, hq, r0 = to_mip_variables(inst, wind, x0)
            ex = solve_cpsat(inst, wind, T, time_limit_s=(60 if a.fast else a.time_limit), workers=a.workers,
                             hint=(hx, hs, hq) if r0.nu == 0 and max(r0.C.values()) <= T else None)
            if ex["status"] not in ("OPTIMAL", "FEASIBLE"):
                print(f"  |V|={nV} seed {seed}: solver found nothing ({ex['status']}) — increase time limit"); continue
            Zs, nus = schedule_from_solution(inst, wind, ex["x"], ex["sigma"], ex["q"])
            agree = abs(Zs - ex["obj"]) < 1e-6 and nus == 0
            if not agree: print(f"  !!! AGREEMENT FAILURE |V|={nV} seed {seed}: solver {ex['obj']} vs simulator {Zs} (nu={nus})")
            ref = ex["obj"]; row = dict(nV=nV, seed=seed, status=ex["status"], obj=ref, bound=ex["bound"],
                                        gap_pct=(ref - ex["bound"]) / ref * 100, time_s=ex["time_s"], agree=int(agree))
            for mth in ["Greedy", "Std-LNS", "ALNS", "ALNS-SA"]:
                row[mth] = (timed_evaluate(mth, None, inst, wind)["cost"] - ref) / ref * 100
            dq = [(timed_evaluate("DQN-LNS", p, inst, wind)["cost"] - ref) / ref * 100 for p in policies.values()]
            row.update(DQN_mean=np.mean(dq) if dq else math.nan, DQN_worst=max(dq) if dq else math.nan, DQN_n=len(dq))
            new = not OUT.exists()
            with open(OUT, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS); (w.writeheader() if new else None); w.writerow(row)
            print(f"  |V|={nV} seed {seed}: {ex['status']} obj={ref:.0f} gap={row['gap_pct']:.2f}% {ex['time_s']:.0f}s agree={agree} "
                  f"Greedy {row['Greedy']:+.1f}% ALNS {row['ALNS']:+.1f}% DQN {row['DQN_mean']:+.1f}%", flush=True)
    # table rows
    rows = list(csv.DictReader(open(OUT)))
    print("\n%% tab:validation rows: |V| & proven optimal & mean solver gap & time & agreement & Greedy & Std-LNS & ALNS & ALNS-SA & DQN-LNS (mean / worst seed)")
    for nV in sorted({int(r["nV"]) for r in rows}):
        rs = [r for r in rows if int(r["nV"]) == nV]; f = lambda k: np.mean([float(r[k]) for r in rs])
        nopt = sum(r["status"] == "OPTIMAL" for r in rs)
        print(f"{nV} & {nopt}/{len(rs)} & {f('gap_pct'):.2f}\\% & {f('time_s'):.0f} & {int(all(int(r['agree']) for r in rs))} & "
              f"{f('Greedy'):+.2f}\\% & {f('Std-LNS'):+.2f}\\% & {f('ALNS'):+.2f}\\% & {f('ALNS-SA'):+.2f}\\% & {f('DQN_mean'):+.2f}\\% / {f('DQN_worst'):+.2f}\\% \\\\")

if __name__ == "__main__":
    main()
