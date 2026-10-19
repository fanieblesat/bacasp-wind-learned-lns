"""compare_retrained.py — how much do the paper's numbers depend on the particular policy files?

Fifteen of the 35 trained policies of the campaign were lost before this repository was assembled and were retrained
with the same code, seed and settings on a different machine (POLICIES.md). Training is deterministic on a given
machine and software stack but not across machines, so a retrained policy is a new draw from the same training
procedure, not the original file. This tool evaluates each retrained policy on EVERY (point, instance, realization)
on which its original was logged in scripts/results/runs.csv, writes the costs to scripts/results/retrained_runs.csv
(the record itself is never modified), and summarizes original vs retrained against the ALNS rows of the record.

Usage (from the repository root):
  python tools/compare_retrained.py scripts/results/policy_full_seed2.pkl     # evaluate one policy (resumable)
  python tools/compare_retrained.py --summary                                  # the comparison
"""
import sys, os, csv, pickle, argparse
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); SCRIPTS = os.path.join(ROOT, "scripts")
ap = argparse.ArgumentParser(); ap.add_argument("pkls", nargs="*"); ap.add_argument("--summary", action="store_true"); a = ap.parse_args()
pkls = [os.path.abspath(p) for p in a.pkls]
sys.path.insert(0, SCRIPTS); os.chdir(SCRIPTS)
import numpy as np, pandas as pd
from scipy import stats
ALLOWED = {("torch._utils", "_rebuild_tensor_v2"), ("torch.storage", "_load_from_bytes"), ("collections", "OrderedDict")}
class Safe(pickle.Unpickler):
    def find_class(self, m, n):
        if (m, n) in ALLOWED: return super().find_class(m, n)
        raise pickle.UnpicklingError(f"blocked global {m}.{n}")
from common import *
from interface import make_instance, make_wind, timed_evaluate
from jos_core import search, controller, params
OUT = os.path.join("results", "retrained_runs.csv")
runs = pd.read_csv("results/runs.csv")
RETRAINED = ["policy_full_seed2.pkl", "policy_full_seed3.pkl", "policy_op_seed2.pkl", "policy_op_seed3.pkl", "policy_op_seed4.pkl",
             *[f"policy_masked_seed{k}.pkl" for k in range(5)], "policy_broad_seed0.pkl", "policy_broad_seed1.pkl",
             *[f"policy_param_theta_s_{v}.pkl" for v in ("0.1", "0.2", "0.5")]]

def setting(point=""):
    search.R_FRAC, search.THETA_S, params.ALPHA = 0.2, 0.3, 0.5; controller.TAU[:] = [0.0, 0.005, 0.02]
    if point.startswith("param:"):
        name, val = point[6:].split("=")
        if name == "r": search.R_FRAC = float(val)
        elif name == "theta_s": search.THETA_S = float(val)
        elif name == "alpha": params.ALPHA = float(val)
        elif name == "tau":
            v = tuple(float(x) for x in val.strip("()").split(",")); controller.TAU[:] = [0.0, v[0], v[1]]

def target(fn):
    b = os.path.basename(fn)[:-4].replace("policy_", "")
    if b.startswith("param_"):
        rest = b[6:]
        if rest.startswith("theta_s_"): return "param:theta_s=" + rest[8:], "DQN-LNS", "DQN-LNS", 0
        name, val = rest.split("_", 1); return f"param:{name}={val}", "DQN-LNS", "DQN-LNS", 0
    var, sd = b.rsplit("_seed", 1)
    m = {"full": ("DQN-LNS", "DQN-LNS"), "op": ("DQN-op", "DQN-op"), "masked": ("DQN-masked", "DQN-masked"),
         "broad": ("DQN-broad", "DQN-LNS"), "noweather": ("DQN-noweather", "DQN-noweather")}[var]
    return None, m[0], m[1], int(sd)

if not a.summary:
    done = set()
    if os.path.exists(OUT):
        for r in csv.DictReader(open(OUT)): done.add((r["point"], r["method"], int(r["seed"]), int(r["instance"]), int(r["realization"])))
    for fn in pkls:
        pol = Safe(open(fn, "rb")).load(); point, logged, ev, sd = target(fn)
        rows = runs[(runs.method == logged) & (runs.seed == sd)]
        if point: rows = rows[rows.point == point]
        for pt, grp in rows.groupby("point"):
            cfg = {**CONFIGS, **SWEEP}.get(pt, REF); setting(pt)
            for _, r in grp.iterrows():
                key = (pt, logged, sd, int(r.instance), int(r.realization))
                if key in done: continue
                inst = make_instance(int(r.instance), cfg); wind = make_wind(wind_seed(int(r.instance), int(r.realization)), cfg)
                res = timed_evaluate(ev, pol, inst, wind, budget=500, seed=int(r.realization))
                new = not os.path.exists(OUT)
                with open(OUT, "a", newline="") as f:
                    w = csv.writer(f)
                    if new: w.writerow(["point", "method", "seed", "instance", "realization", "cost", "Z", "viol", "time_s"])
                    w.writerow([pt, logged, sd, int(r.instance), int(r.realization), res["cost"], res.get("Z", res["cost"]), res["viol"], res["time_s"]])
            setting(); print(f"{os.path.basename(fn):<34} {pt:<22} done", flush=True)
    sys.exit(0)

# ---------------- summary
new = pd.read_csv(OUT)
def inst_mean(d): return d.groupby("instance").cost.mean()
print(f"{'policy':<14}{'point':<22}{'n':>4}  {'orig vs ALNS':>12} {'retr vs ALNS':>12}  {'retr vs orig':>12}  p(retr=orig)")
rows = []
for (pt, m, sd), g in new.groupby(["point", "method", "seed"]):
    o = runs[(runs.point == pt) & (runs.method == m) & (runs.seed == sd)]
    al = runs[(runs.point == pt) & (runs.method == "ALNS")]
    oi, ri, ai = inst_mean(o), inst_mean(g), inst_mean(al); idx = oi.index.intersection(ri.index).intersection(ai.index)
    go = (oi[idx].mean() - ai[idx].mean()) / ai[idx].mean() * 100; gr = (ri[idx].mean() - ai[idx].mean()) / ai[idx].mean() * 100
    d = (ri[idx].mean() - oi[idx].mean()) / oi[idx].mean() * 100
    p = stats.wilcoxon(ri[idx], oi[idx]).pvalue if not np.allclose(ri[idx], oi[idx]) else 1.0
    rows.append((m, sd, pt, go, gr)); print(f"{m[4:]+' s'+str(sd):<14}{pt:<22}{len(idx):>4}  {go:>+11.2f}% {gr:>+11.2f}%  {d:>+11.2f}%  {p:.3f}")
# the paper-level quantity: the seed-mean gap of a learned method to ALNS, with the original files and with the
# retrained files substituted for the lost ones (the other seeds unchanged)
print("\nseed-mean gap to ALNS (as reported in the paper)  original -> with retrained files substituted")
for m, pts in (("DQN-LNS", sorted(new[new.method == "DQN-LNS"].point.unique())), ("DQN-op", ["mu10", "mu15", "mu20"]),
               ("DQN-masked", ["mu10", "mu15", "mu20", "sig8.0"]), ("DQN-broad", sorted(new[new.method == "DQN-broad"].point.unique()))):
    for pt in pts:
        o = runs[(runs.point == pt) & (runs.method == m)]
        if o.empty: continue
        r = new[(new.point == pt) & (new.method == m)]
        sub = pd.concat([o[~o.seed.isin(r.seed.unique())], r])
        al = inst_mean(runs[(runs.point == pt) & (runs.method == "ALNS")])
        g = lambda d: (d.groupby("instance").cost.mean().mean() - al.mean()) / al.mean() * 100
        pa = lambda d: stats.wilcoxon(d.groupby("instance").cost.mean().loc[al.index], al).pvalue
        print(f"  {m:<11}{pt:<22} {g(o):+6.2f}% (p={pa(o):.3g})  ->  {g(sub):+6.2f}% (p={pa(sub):.3g})")
