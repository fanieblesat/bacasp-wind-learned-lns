"""verify_policies.py — check that every trained policy in scripts/results/ is the one that produced the record.

For each policy_*.pkl the script finds the method/seed (and, for the parameter study, the setting) it is credited with in
scripts/results/runs.csv, re-evaluates it on the first K logged (instance, wind realization) pairs, and compares the
returned penalized cost with the logged value. Evaluation of a fixed policy is deterministic, so the expected outcome is
exact equality ("K/K reproduce"). The two constant-learning-rate policies (fullnodecay) are used only for the training-curve
comparison and have no runs.csv rows; for them the stored best validation score is compared with train_curves.csv.

Policies are loaded with a restricted unpickler that admits only tensor reconstruction and OrderedDict.
Usage (from the repository root):  python tools/verify_policies.py [--k 3] [policy files ...]
"""
import sys, os, glob, pickle, argparse
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); SCRIPTS = os.path.join(ROOT, "scripts")
ap = argparse.ArgumentParser(); ap.add_argument("pkls", nargs="*"); ap.add_argument("--k", type=int, default=3); a = ap.parse_args()
RETRAINED = {"policy_full_seed2.pkl", "policy_full_seed3.pkl", "policy_op_seed2.pkl", "policy_op_seed3.pkl", "policy_op_seed4.pkl",
             *{f"policy_masked_seed{k}.pkl" for k in range(5)}, "policy_broad_seed0.pkl", "policy_broad_seed1.pkl",
             *{f"policy_param_theta_s_{v}.pkl" for v in ("0.1", "0.2", "0.5")}}   # see POLICIES.md: not the files that produced the record
pkls = [os.path.abspath(p) for p in a.pkls] or sorted(glob.glob(os.path.join(SCRIPTS, "results", "policy_*.pkl")))
skipped = [p for p in pkls if os.path.basename(p) in RETRAINED]; pkls = [p for p in pkls if os.path.basename(p) not in RETRAINED]
sys.path.insert(0, SCRIPTS); os.chdir(SCRIPTS)
import pandas as pd
ALLOWED = {("torch._utils", "_rebuild_tensor_v2"), ("torch.storage", "_load_from_bytes"), ("collections", "OrderedDict")}
class Safe(pickle.Unpickler):
    def find_class(self, m, n):
        if (m, n) in ALLOWED: return super().find_class(m, n)
        raise pickle.UnpicklingError(f"blocked global {m}.{n}")
from common import *
from interface import make_instance, make_wind, timed_evaluate
from jos_core import search, controller, params
runs = pd.read_csv("results/runs.csv"); curves = pd.read_csv("results/train_curves.csv", header=None)

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
    m = {"full": ("DQN-LNS", "DQN-LNS"), "fullnodecay": ("DQN-LNS-nodecay", None), "op": ("DQN-op", "DQN-op"),
         "noweather": ("DQN-noweather", "DQN-noweather"), "masked": ("DQN-masked", "DQN-masked"), "broad": ("DQN-broad", "DQN-LNS")}[var]
    return None, m[0], m[1], int(sd)

bad = 0
for fn in pkls:
    pol = Safe(open(fn, "rb")).load(); name = os.path.basename(fn)
    point, logged, ev, sd = target(fn)
    if ev is None:   # constant-learning-rate policies: compare with the training curve
        c = curves[(curves[0] == "full-nodecay") & (curves[1] == sd)]
        ok = abs(c[3].min() - pol["best_val"]) < 1e-4
        print(f"{name:<36} best validation {pol['best_val']:.5f} vs curve minimum {c[3].min():.5f}  {'OK' if ok else 'MISMATCH'}"); bad += not ok; continue
    rows = runs[(runs.method == logged) & (runs.seed == sd)]
    if point: rows = rows[rows.point == point]
    if rows.empty: print(f"{name:<36} no logged rows for {logged} seed {sd} {point or ''}"); bad += 1; continue
    pt = point or ("3B-6QC" if "3B-6QC" in set(rows.point) else sorted(set(rows.point))[0])
    rows = rows[rows.point == pt].head(a.k); cfg = {**CONFIGS, **SWEEP}.get(pt, REF); setting(pt); hits = 0
    for _, r in rows.iterrows():
        inst = make_instance(int(r.instance), cfg); wind = make_wind(wind_seed(int(r.instance), int(r.realization)), cfg)
        hits += abs(timed_evaluate(ev, pol, inst, wind, budget=500, seed=int(r.realization))["cost"] - r.cost) < 1e-6
    setting(); bad += hits < len(rows)
    print(f"{name:<36} {logged:<15} seed {sd} @ {pt:<22} {hits}/{len(rows)} reproduce")
for p in skipped: print(f"{os.path.basename(p):<36} retrained (original lost): not expected to reproduce the record; see tools/compare_retrained.py")
print(f"ALL {len(pkls)} ORIGINAL POLICIES REPRODUCE THE RECORD" if not bad else f"{bad} POLICY FILE(S) DO NOT REPRODUCE THE RECORD")
sys.exit(1 if bad else 0)
