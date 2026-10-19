"""
paper_rows.py — the exact LaTeX rows of the manuscript tables, from results/runs.csv.

Convention used in every table (stated once in Section 5.2 of the paper):
  * "cost" is Z, the turnaround cost of the schedule a method returns;
  * infeasible returns (nu > 0) are counted separately (column "Inf.");
  * p-values are Wilcoxon signed-rank tests on the penalized objective Z_Omega, paired on
    (instance, realization) and averaged over seeds, so that an infeasible return is ranked
    as the worst outcome of its instance;
  * "wins" counts instances on which DQN-LNS has lower Z_Omega than the baseline.
Usage: python paper_rows.py results/runs.csv > results/paper_rows.tex
"""
import sys, math
import numpy as np, pandas as pd
from scipy import stats
import stats_tools as S

df = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "results/runs.csv")
BASE = "ALNS"

def Z(m, pt):  return S.per_instance_Z(df, m, pt)
def ZO(m, pt): return S.per_instance(df, m, pt)
def ninf(m, pt):
    d = df[(df.point == pt) & (df.method == m)]; return int((d.viol > 0).sum()), len(d)
def gapZ(m, pt, b=BASE):
    a, bb = Z(m, pt), Z(b, pt); return (a.mean() - bb.mean()) / bb.mean() * 100
def pw(m, pt, b=BASE):
    return S.paired_stats(ZO(m, pt), ZO(b, pt))
def seed_gapsZ(m, pt, b=BASE):
    base = Z(b, pt); out = {}
    for sd, g in df[(df.point == pt) & (df.method == m)].groupby("seed"):
        s = g.groupby("instance").Z.mean(); idx = s.index.intersection(base.index)
        out[int(sd)] = float((s.loc[idx].mean() - base.loc[idx].mean()) / base.loc[idx].mean() * 100)
    return out
def inf_str(m, pt):
    k, n = ninf(m, pt); return f"{k}/{n}" if k else "0"

def main_rows(pt):
    rows = []
    for m in ["DQN-LNS", "ALNS", "ALNS-SA", "Std-LNS", "DQN-R", "Greedy"]:
        if m not in set(df[df.point == pt].method): continue
        z = Z(m, pt).mean(); cell = S.fc(z)
        d = df[(df.point == pt) & (df.method == m)]
        if m in ("DQN-LNS", "DQN-R"):
            sm = d.groupby("seed").Z.mean(); cell += f" [{sm.std(ddof=1):,.0f}]".replace(",", "{,}")
        if m == BASE:
            rows.append(f" & {m} & {cell} & {inf_str(m, pt)} & --- & --- & --- \\\\")
        else:
            st = pw(m, pt)
            extra = f"{S.fp(st['p'])} & {st['wins']}/{st['n']}" if m == "DQN-LNS" else "--- & ---"
            rows.append(f" & {m} & {cell} & {inf_str(m, pt)} & {S.fg(gapZ(m, pt))} & {extra} \\\\")
    return rows

def sweep_row(pt, label):
    st = pw("DQN-LNS", pt); sl = pw("DQN-LNS", pt, "DQN-R")
    eq = S.tost(ZO("DQN-LNS", pt), ZO(BASE, pt))
    eqs = "equiv." if eq["equivalent"] else ("n.s." if st["p"] >= 0.05 else "")
    return (f"\\quad {label} & {S.fc(Z('DQN-LNS', pt).mean())} & {S.fc(Z('DQN-R', pt).mean())} & {S.fc(Z(BASE, pt).mean())} & "
            f"{inf_str('DQN-LNS', pt)} & {S.fg(gapZ('DQN-LNS', pt))} & {S.fp(st['p'])} & {st['wins']}/{st['n']} & "
            f"{S.fg(gapZ('DQN-LNS', pt, 'DQN-R'))} & {S.fp(sl['p'])} & {eqs} \\\\")

def fleet_row(pt, label):
    st = pw("DQN-LNS", pt); eq = S.tost(ZO("DQN-LNS", pt), ZO(BASE, pt))
    eqs = "equiv." if eq["equivalent"] else ("n.s." if st["p"] >= 0.05 else "")
    return (f"{label} & {S.fc(Z('DQN-LNS', pt).mean())} & {S.fc(Z('DQN-R', pt).mean())} & {S.fc(Z(BASE, pt).mean())} & "
            f"{S.fc(Z('Greedy', pt).mean())} & {S.fg(gapZ('DQN-LNS', pt))} & {S.fp(st['p'])} & {st['wins']}/{st['n']} & {eqs} \\\\")

def ablation_row(pt, label):
    env = gapZ("DQN-R", pt); learn = gapZ("DQN-LNS", pt, "DQN-R"); tot = gapZ("DQN-LNS", pt)
    sl = pw("DQN-LNS", pt, "DQN-R"); eq = S.tost(ZO("DQN-LNS", pt), ZO("DQN-R", pt))
    eqs = "equiv." if eq["equivalent"] else ("n.s." if sl["p"] >= 0.05 else "")
    return f"{label} & {S.fg(env)} & {S.fg(learn)} & {S.fg(tot)} & {S.fp(sl['p'])} & {eqs} \\\\"

def two_row(pt, label, m2):
    s1 = pw("DQN-LNS", pt); s2 = pw(m2, pt); s12 = pw(m2, pt, "DQN-LNS")
    return (f"{label} & {S.fg(gapZ('DQN-LNS', pt))} ({inf_str('DQN-LNS', pt)}) & {S.fg(gapZ(m2, pt))} ({inf_str(m2, pt)}) & "
            f"{S.fg(gapZ(m2, pt, 'DQN-LNS'))} & {S.fp(s12['p'])} \\\\")

def seeds_row(pt, label):
    g = seed_gapsZ("DQN-LNS", pt); v = np.array([g[s] for s in sorted(g)])
    p = float(stats.ttest_1samp(v, 0).pvalue)
    return f"{label} & " + " & ".join(S.fg(g[s]) for s in sorted(g)) + f" & {S.fg(v.mean())} $\\pm$ {v.std(ddof=1):.2f} & {S.fp(p)} \\\\"

def param_rows(name, vals, default):
    for v in vals:
        pt = "mu10" if v == default else f"param:{name}={v}"
        if pt not in set(df.point): print(f" & {v} & \\todo{{}} & \\todo{{}} & \\todo{{}} \\\\"); continue
        st = pw("DQN-LNS", pt); g = S.fg(gapZ("DQN-LNS", pt))
        lab = f"\\textbf{{{v}}}" if v == default else f"{v}"
        print(f" & {lab} & {g} & {S.fp(st['p'])} & {st['wins']}/{st['n']} \\\\")

def pr(x=""): print(x)

pr("%% ---- tab:main"); [pr(r) for pt in ["2B-4QC", "3B-6QC", "4B-8QC", "5B-10QC"] for r in [f"%% {pt}"] + main_rows(pt)]
pr("%% ---- tab:wind")
for pt, lab in [("mu5", "5"), ("mu10", "10"), ("mu15", "15"), ("mu20", "20")]: pr(sweep_row(pt, lab))
for pt, lab in [("sig1.5", "1.5"), ("sig3.5", "3.5"), ("sig5.5", "5.5"), ("sig8.0", "8.0")]: pr(sweep_row(pt, lab))
pr("%% ---- tab:fleet")
for pt, lab in [("V20", "20"), ("V30", "30"), ("V50", "50"), ("V75", "75"), ("V100", "100")]: pr(fleet_row(pt, lab))
pr("%% ---- tab:ablation")
for pt, lab in [("mu5", "$\\mu_w = 5$"), ("mu10", "$\\mu_w = 10$"), ("mu15", "$\\mu_w = 15$"), ("mu20", "$\\mu_w = 20$"),
                ("sig1.5", "$\\sigma_w = 1.5$"), ("sig5.5", "$\\sigma_w = 5.5$"), ("sig8.0", "$\\sigma_w = 8.0$"),
                ("3B-6QC", "3B-6QC"), ("V100", "$|V| = 100$")]: pr(ablation_row(pt, lab))
pr("%% ---- tab:weather_ablation")
for pt, lab in [("mu5", "$\\mu_w = 5$"), ("mu10", "$\\mu_w = 10$"), ("mu15", "$\\mu_w = 15$"), ("mu20", "$\\mu_w = 20$")]: pr(two_row(pt, lab, "DQN-noweather"))
pr("%% ---- tab:control")
for pt, lab in [("mu10", "10"), ("mu15", "15"), ("mu20", "20")]: pr(two_row(pt, lab, "DQN-op"))
pr("%% ---- tab:masking")
for pt, lab in [("mu10", "$\\mu_w = 10$"), ("mu15", "$\\mu_w = 15$"), ("mu20", "$\\mu_w = 20$"), ("sig8.0", "$\\sigma_w = 8.0$")]: pr(two_row(pt, lab, "DQN-masked"))
pr("%% ---- tab:distribution")
for pt, lab in [("2B-4QC", "2B-4QC"), ("3B-6QC", "3B-6QC (reference)"), ("4B-8QC", "4B-8QC"), ("5B-10QC", "5B-10QC"),
                ("V20", "$|V| = 20$"), ("V30", "$|V| = 30$"), ("V75", "$|V| = 75$"), ("V100", "$|V| = 100$")]: pr(two_row(pt, lab, "DQN-broad"))
pr("%% ---- tab:seeds")
for pt, lab in [("3B-6QC", "3B-6QC"), ("mu10", "$\\mu_w = 10$"), ("mu15", "$\\mu_w = 15$"), ("mu20", "$\\mu_w = 20$"),
                ("sig8.0", "$\\sigma_w = 8.0$"), ("5B-10QC", "5B-10QC"), ("V20", "$|V| = 20$"), ("V100", "$|V| = 100$")]: pr(seeds_row(pt, lab))
pr("%% ---- tab:params (one row per setting; default = the mu10 five-seed value)")
param_rows("r", [0.1, 0.2, 0.3, 0.4], 0.2); param_rows("theta_s", [0.1, 0.2, 0.3, 0.5], 0.3)
param_rows("tau", ["(0.002, 0.01)", "(0.005, 0.02)", "(0.01, 0.05)"], "(0.005, 0.02)"); param_rows("alpha", [0.25, 0.5, 0.75], 0.5)
pr("%% ---- tab:time (per-run seconds at 3B-6QC)")
for m in ["Greedy", "Std-LNS", "ALNS", "ALNS-SA", "DQN-R", "DQN-LNS"]:
    d = df[(df.point == "3B-6QC") & (df.method == m)]; pr(f"{m} & {d.time_s.mean():.2f} $\\pm$ {d.time_s.std():.2f} \\\\")
pr("%% ---- infeasible returns, all search methods")
d = df[df.method != "Greedy"]; pr(f"%% {int((d.viol>0).sum())} of {len(d)} runs; Greedy: {int((df[df.method=='Greedy'].viol>0).sum())} of {len(df[df.method=='Greedy'])}")
for _, r in d[d.viol > 0].iterrows(): pr(f"%%   {r.point} {r.method} seed {r.seed} inst {r.instance} real {r.realization} Z={r.Z:.0f}")
