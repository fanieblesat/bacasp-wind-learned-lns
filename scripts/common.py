"""common.py — configurations, pools, and the single results logger shared by all drivers."""
from __future__ import annotations
import csv, os, json, sys
from pathlib import Path

RESULTS = Path("results"); RESULTS.mkdir(exist_ok=True)
RUNS_CSV = RESULTS / "runs.csv"
FIELDS = ["point", "method", "seed", "instance", "realization", "cost", "Z", "viol", "time_s"]

REF = dict(n_berths=3, n_cranes=6, beta=[1.0, 0.7, 0.4], n_vessels=50, horizon=168,
           mu_w=10.0, sigma_w=3.5, theta_w=0.12)

CONFIGS = {
    "2B-4QC":  dict(REF, n_berths=2, n_cranes=4,  beta=[1.0, 0.4]),
    "3B-6QC":  dict(REF),
    "4B-8QC":  dict(REF, n_berths=4, n_cranes=8,  beta=[1.0, 0.8, 0.6, 0.4]),
    "5B-10QC": dict(REF, n_berths=5, n_cranes=10, beta=[1.0, 0.85, 0.7, 0.55, 0.4]),
}
SWEEP = {**{f"mu{m}": dict(REF, mu_w=float(m)) for m in (5, 10, 15, 20)},
         **{f"sig{s}": dict(REF, sigma_w=float(s)) for s in (1.5, 3.5, 5.5, 8.0)},
         **{f"V{v}": dict(REF, n_vessels=v) for v in (20, 30, 50, 75, 100)}}

BENCH_SEEDS = list(range(2000, 2050))   # 50 benchmark instances
SWEEP_SEEDS = list(range(1000, 1020))   # 20 sweep instances
VALID_SEEDS = list(range(3000, 3010))   # 10 validation instances per size
N_REAL = 3                              # wind realizations per instance
TRAIN_SEEDS = [0, 1, 2, 3, 4]           # five independently trained policies

def wind_seed(instance_seed: int, realization: int) -> int:
    return 10_000_000 + 100 * instance_seed + realization   # joint, reproducible

def log_run(point, method, seed, instance, realization, r: dict):
    new = not RUNS_CSV.exists()
    with open(RUNS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new: w.writeheader()
        w.writerow(dict(point=point, method=method, seed=seed, instance=instance,
                        realization=realization, cost=r["cost"], Z=r.get("Z", r["cost"]), viol=r["viol"], time_s=r["time_s"]))

def already_done(point, method, seed) -> set:
    """Resume support: (instance, realization) pairs already logged for this key."""
    if not RUNS_CSV.exists(): return set()
    done = set()
    with open(RUNS_CSV) as f:
        for row in csv.DictReader(f):
            if row["point"] == point and row["method"] == method and int(row["seed"]) == seed:
                done.add((int(row["instance"]), int(row["realization"])))
    return done

def save_policy_marker(name: str, meta: dict):
    (RESULTS / f"policy_{name}.json").write_text(json.dumps(meta, indent=1))
