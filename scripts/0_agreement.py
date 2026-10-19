"""
0_agreement.py — simulator <-> exact model agreement (R2-7). Run FIRST; takes ~5 min.

Three checks, all must print AGREE:
  (A) small instances: the exact optimum's decision variables, replayed by the simulator's
      rules (evaluate_fixed), reproduce the solver objective and violate no constraint;
  (B) full-size instances (|V| = 50 and 100, all wind regimes): schedules produced by the
      simulator (Greedy and a 300-iteration LNS) are FEASIBLE in the exact model when their
      variables are fixed, with identical objective — i.e. every schedule the heuristics
      report is a feasible solution of the model with exactly the reported cost;
  (C) corollary check: the crane-specific model (xi, non-crossing) and the aggregated count
      model have the same optimum on a tiny instance.
Writes results/agreement.txt.
"""
import sys, time, random, math
from common import *
from interface import make_instance, make_wind
from jos_core.simulator import evaluate, evaluate_fixed, to_mip_variables
from jos_core.exact_cpsat import solve_cpsat
from jos_core.exact import solve as solve_scip
from jos_core import search

out = []
def say(s): print(s, flush=True); out.append(s)

say("(A) exact optimum replayed by the simulator")
for nV, seed, mu in [(6, 3000, 10.0), (6, 3001, 15.0), (6, 3002, 20.0), (8, 3003, 10.0)]:
    cfg = dict(REF, n_vessels=nV, horizon=48, mu_w=mu)
    inst = make_instance(seed, cfg); wind = make_wind(wind_seed(seed, 0), cfg)
    ex = solve_cpsat(inst, wind, 48, time_limit_s=120, workers=4)
    Z, nu, prob = evaluate_fixed(inst, wind, ex["x"], ex["sigma"], ex["q"])
    ok = abs(Z - ex["obj"]) < 1e-6 and not prob
    say(f"  |V|={nV} seed {seed} mu={mu}: solver {ex['status']} obj={ex['obj']:.0f} | simulator replay Z={Z:.0f} nu={nu} -> {'AGREE' if ok else 'MISMATCH ' + str(prob[:2])}")

say("(B) full-size simulator schedules fixed in the exact model")
for name, cfg, seed in [("3B-6QC mu10", REF, 2000), ("mu20", SWEEP["mu20"], 2003), ("sig8", SWEEP["sig8.0"], 2004),
                        ("5B-10QC", CONFIGS["5B-10QC"], 2001), ("V100 mu15", dict(SWEEP["mu15"], n_vessels=100), 2002)]:
    inst = make_instance(seed, cfg); wind = make_wind(wind_seed(seed, 0), cfg)
    rng = random.Random(0); s = search.SearchRun(inst, wind, 300, False)
    for i in range(300):
        cand = search.apply(rng.randrange(6), s.cur, inst, wind, rng); rr = s.consider(cand)
        if rr.Z_omega <= s.cur_r.Z_omega: s.cur, s.cur_r = cand, rr
    for label, sol in (("greedy", s.x0), ("LNS-300", s.best)):
        r = evaluate(inst, wind, sol)
        if r.nu > 0:
            say(f"  {name} {label}: nu={r.nu} (infeasible by construction; not fixable) Z={r.Z:.0f}"); continue
        T = max(r.C.values()) + 1
        x, sig, q, _ = to_mip_variables(inst, wind, sol)
        fx = solve_cpsat(inst, wind, T, time_limit_s=300, workers=4, fix=(x, sig, q))
        ok = fx["status"] in ("OPTIMAL", "FEASIBLE") and abs(fx["obj"] - r.Z) < 1e-6
        say(f"  {name} {label}: |V|={inst.n} T={T} simulator Z={r.Z:.0f} | model {fx['status']} obj={fx['obj']:.0f} -> {'AGREE' if ok else 'MISMATCH'}")

say("(C) crane-specific vs aggregated model")
cfg = dict(REF, n_vessels=3, horizon=24, mu_w=15.0); inst = make_instance(3010, cfg); wind = make_wind(wind_seed(3010, 0), cfg)
a = solve_cpsat(inst, wind, 24, 120, 4); b = solve_scip(inst, wind, 24, 600, aggregated=False)
say(f"  |V|=3 T=24: aggregated {a['status']} {a['obj']:.0f} | crane-specific {b['status']} {b['obj']:.0f} -> {'AGREE' if abs(a['obj']-b['obj'])<1e-6 else 'MISMATCH'}")
(RESULTS / "agreement.txt").write_text("\n".join(out) + "\n")
say(f"written {RESULTS/'agreement.txt'}")
