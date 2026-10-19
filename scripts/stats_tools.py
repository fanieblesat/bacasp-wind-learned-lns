"""
stats_tools.py — statistics and LaTeX emitters for the revised manuscript.

Fully runnable and independent of the simulator. Input is a long-format
CSV of per-run records with columns

    point, method, seed, instance, realization, cost, viol, time_s

where `point` is an evaluation point label ("3B-6QC", "mu5", "sig8.0",
"V75", ...), `method` in {Greedy, Std-LNS, ALNS, ALNS-SA, DQN-LNS, DQN-R,
DQN-op, DQN-noweather, DQN-masked, ...}, `seed` is the training seed for
learned methods (use 0 for non-learned), `realization` in {0,1,2}.

Implements the protocol of Section 5.2:
  * mean over realizations per (method, seed, instance)      [R2-6]
  * mean over seeds for learned methods; across-seed sd       [R2-8, R3-2]
  * paired Wilcoxon signed-rank vs ALNS, paired Cohen's d, wins
  * TOST equivalence test at +-margin of ALNS mean            [R2-9]
  * per-seed table with one-sample t-test across seeds        [Sec 5.13]
  * LaTeX row emitters for every table in Section 5

Usage:
    python stats_tools.py results.csv --point 3B-6QC
    python stats_tools.py results.csv --all
    python stats_tools.py --selftest        # runs on synthetic data
"""
from __future__ import annotations
import argparse
import math
import sys
import numpy as np
import pandas as pd
from scipy import stats

BASELINE = "ALNS"
EQUIV_MARGIN = 0.01      # +-1% of baseline mean cost


# ---------------------------------------------------------------- aggregation
def per_instance(df: pd.DataFrame, method: str, point: str) -> pd.Series:
    """Mean cost over realizations AND seeds, indexed by instance."""
    d = df[(df.method == method) & (df.point == point)]
    if d.empty:
        raise KeyError(f"no rows for method={method} point={point}")
    return d.groupby("instance").cost.mean().sort_index()


def per_instance_Z(df, method, point) -> pd.Series:
    """Mean UNPENALIZED turnaround Z over realizations and seeds (what the tables display);
    tests and rankings use per_instance (Z_Omega), which equals Z whenever nu = 0."""
    d = df[(df.method == method) & (df.point == point)]
    col = "Z" if "Z" in d.columns else "cost"
    return d.groupby("instance")[col].mean().sort_index()


def per_instance_by_seed(df, method, point) -> pd.DataFrame:
    """Rows: instance; columns: seed; value: mean over realizations."""
    d = df[(df.method == method) & (df.point == point)]
    return d.groupby(["instance", "seed"]).cost.mean().unstack("seed").sort_index()


def viol_mean(df, method, point) -> float:
    d = df[(df.method == method) & (df.point == point)]
    return float(d.viol.mean())


# ---------------------------------------------------------------- tests
def paired_stats(a: pd.Series, b: pd.Series) -> dict:
    """a = method, b = baseline, aligned on instance. Negative gap = a better."""
    idx = a.index.intersection(b.index)
    a, b = a.loc[idx].values, b.loc[idx].values
    diff = a - b
    gap = float(diff.mean() / b.mean() * 100)
    if np.allclose(diff, 0):
        p = 1.0
    else:
        p = float(stats.wilcoxon(a, b).pvalue)
    d = float(diff.mean() / diff.std(ddof=1)) if diff.std(ddof=1) > 0 else 0.0
    wins = int((a < b).sum())
    return dict(gap=gap, p=p, d=d, wins=wins, n=len(idx), mean_a=float(a.mean()), mean_b=float(b.mean()))


def tost(a: pd.Series, b: pd.Series, margin_frac: float = EQUIV_MARGIN) -> dict:
    """Two one-sided tests for equivalence of paired means within +-margin of
    the baseline mean. Equivalent iff BOTH one-sided p-values < 0.05."""
    idx = a.index.intersection(b.index)
    diff = (a.loc[idx] - b.loc[idx]).values
    margin = margin_frac * float(b.loc[idx].mean())
    n = len(diff)
    if n < 3 or diff.std(ddof=1) == 0:
        return dict(equivalent=False, p_lower=float("nan"), p_upper=float("nan"), margin=margin)
    se = diff.std(ddof=1) / math.sqrt(n)
    t_low = (diff.mean() + margin) / se     # H0: diff <= -margin
    t_up = (diff.mean() - margin) / se      # H0: diff >= +margin
    p_low = 1 - stats.t.cdf(t_low, n - 1)
    p_up = stats.t.cdf(t_up, n - 1)
    return dict(equivalent=bool(p_low < 0.05 and p_up < 0.05),
                p_lower=float(p_low), p_upper=float(p_up), margin=margin)


def seed_table(df, method, point, baseline=BASELINE) -> dict:
    """Per-seed gap to baseline, mean +- sd across seeds, one-sample t vs 0."""
    base = per_instance(df, baseline, point)
    byseed = per_instance_by_seed(df, method, point)
    gaps = {}
    for sd in byseed.columns:
        s = byseed[sd].dropna()
        idx = s.index.intersection(base.index)
        gaps[int(sd)] = float((s.loc[idx] - base.loc[idx]).mean() / base.loc[idx].mean() * 100)
    g = np.array(list(gaps.values()))
    p = float(stats.ttest_1samp(g, 0).pvalue) if len(g) > 1 and g.std(ddof=1) > 0 else float("nan")
    return dict(per_seed=gaps, mean=float(g.mean()), sd=float(g.std(ddof=1)) if len(g) > 1 else 0.0, p=p)


# ---------------------------------------------------------------- formatting
def fp(p: float) -> str:
    if isinstance(p, float) and math.isnan(p): return "n/a"
    if p >= 0.001: return f"{p:.3f}"
    e = int(math.floor(math.log10(p))); m = p / 10 ** e
    return f"${m:.1f}\\!\\times\\!10^{{{e}}}$"

def fc(x: float) -> str:
    return f"{x:,.0f}".replace(",", "{,}")

def fg(g: float) -> str:
    return f"${g:+.2f}\\%$"


def row_main(df, point, method, baseline=BASELINE) -> str:
    """One row of Table main: method, cost [sd across seeds], viol, gap, p, wins."""
    a = per_instance(df, method, point)
    cost = fc(per_instance_Z(df, method, point).mean())
    byseed = per_instance_by_seed(df, method, point)
    if byseed.shape[1] > 1:
        seed_means = byseed.mean(axis=0)
        cost += f" [{seed_means.std(ddof=1):,.0f}]".replace(",", "{,}")
    v = f"{viol_mean(df, method, point):.2f}"
    if method == baseline:
        return f" & {method} & {cost} & {v} & --- & --- & --- \\\\"
    b = per_instance(df, baseline, point)
    st = paired_stats(a, b)
    za, zb = per_instance_Z(df, method, point), per_instance_Z(df, baseline, point)
    st["gap"] = (za.mean() - zb.mean()) / zb.mean() * 100      # displayed gap on turnaround Z; p and wins on Z_Omega
    if method.startswith("DQN-LNS"):
        return f" & {method} & {cost} & {v} & {fg(st['gap'])} & {fp(st['p'])} & {st['wins']}/{st['n']} \\\\"
    return f" & {method} & {cost} & {v} & {fg(st['gap'])} & --- & --- \\\\"


def row_wind(df, point, label) -> str:
    """One row of Table wind: cost DQN / DQN-R / ALNS, gap+p vs ALNS, learning gap+p, TOST."""
    a = per_instance(df, "DQN-LNS", point)
    r = per_instance(df, "DQN-R", point)
    b = per_instance(df, BASELINE, point)
    s1 = paired_stats(a, b); s2 = paired_stats(a, r)
    eq = tost(a, r)
    eqs = "equiv." if eq["equivalent"] else ("n.s." if s2["p"] >= 0.05 else "")
    return (f"\\quad {label} & {fc(a.mean())} & {fc(r.mean())} & {fc(b.mean())} & "
            f"{fg(s1['gap'])} & {fp(s1['p'])} & {fg(s2['gap'])} & {fp(s2['p'])} & {eqs} \\\\")


def row_ablation(df, point, label) -> str:
    a = per_instance(df, "DQN-LNS", point); r = per_instance(df, "DQN-R", point); b = per_instance(df, BASELINE, point)
    env = paired_stats(r, b)["gap"]; learn = paired_stats(a, r); tot = paired_stats(a, b)["gap"]
    eq = tost(a, r)
    eqs = "equiv." if eq["equivalent"] else ("n.s." if learn["p"] >= 0.05 else "")
    return f"{label} & {fg(env)} & {fg(learn['gap'])} & {fg(tot)} & {fp(learn['p'])} & {eqs} \\\\"


def row_two_methods(df, point, label, m1, m2) -> str:
    """Generic: gap+p of m1 and m2 vs ALNS, and their direct difference (Tables weather/control/masking)."""
    b = per_instance(df, BASELINE, point)
    a1 = per_instance(df, m1, point); a2 = per_instance(df, m2, point)
    s1 = paired_stats(a1, b); s2 = paired_stats(a2, b); s12 = paired_stats(a2, a1)
    return (f"{label} & {fg(s1['gap'])} ({viol_mean(df, m1, point):.2f}) & {fg(s2['gap'])} "
            f"({viol_mean(df, m2, point):.2f}) & {fg(s12['gap'])} & {fp(s12['p'])} \\\\")


def row_seeds(df, point, label, method="DQN-LNS") -> str:
    st = seed_table(df, method, point)
    cells = " & ".join(fg(st["per_seed"][s]) for s in sorted(st["per_seed"]))
    return f"{label} & {cells} & {fg(st['mean'])} $\\pm$ {st['sd']:.2f} & {fp(st['p'])} \\\\"




# ---------------------------------------------------------------- multi-method comparison
def friedman_nemenyi(df, point, methods=("Greedy", "Std-LNS", "ALNS", "ALNS-SA", "DQN-LNS"), alpha=0.05):
    """Friedman test across methods on per-instance costs, mean ranks, and the
    Nemenyi critical difference (Demsar 2006). Returns dict; prints a LaTeX line."""
    mats = [per_instance(df, m, point) for m in methods]
    idx = mats[0].index
    for s in mats: idx = idx.intersection(s.index)
    X = np.column_stack([s.loc[idx].values for s in mats])      # n_instances x k
    n, k = X.shape
    stat, p = stats.friedmanchisquare(*[X[:, j] for j in range(k)])
    ranks = np.apply_along_axis(stats.rankdata, 1, X).mean(axis=0)
    # Nemenyi CD at alpha=0.05: q_alpha * sqrt(k(k+1)/(6n)); q from studentized range / sqrt(2)
    q_alpha = stats.studentized_range.ppf(1 - alpha, k, np.inf) / math.sqrt(2)
    cd = q_alpha * math.sqrt(k * (k + 1) / (6 * n))
    line = " & ".join(f"{m} ({r:.2f})" for m, r in zip(methods, ranks))
    print(f"%% Friedman @{point}: chi2={stat:.2f}, p={fp(p)}, n={n}, k={k}; CD(0.05)={cd:.2f}")
    print(f"{point} & {line} & {fp(p)} & {cd:.2f} \\\\")
    return dict(stat=stat, p=p, ranks=dict(zip(methods, ranks)), cd=cd, n=n)

# ---------------------------------------------------------------- self-test
def _synthetic(seed=0) -> pd.DataFrame:
    """Synthetic long-format data with a known structure: DQN-LNS 2.5% better
    than ALNS at mu10, equivalent to DQN-R at mu20; five seeds; three realizations."""
    rng = np.random.default_rng(seed)
    rows = []
    for point, base_cost, dqn_gap, r_gap in [("mu10", 38000, -0.025, +0.02), ("mu20", 48000, +0.08, +0.085)]:
        for inst in range(20):
            ibase = base_cost * rng.normal(1, 0.08)
            for real in range(3):
                noise = rng.normal(1, 0.01)
                rows.append(dict(point=point, method="ALNS", seed=0, instance=inst, realization=real,
                                 cost=ibase * noise, viol=0, time_s=3.0))
                rows.append(dict(point=point, method="DQN-R", seed=0, instance=inst, realization=real,
                                 cost=ibase * (1 + r_gap) * rng.normal(1, 0.01), viol=0, time_s=3.5))
                for sd in range(5):
                    rows.append(dict(point=point, method="DQN-LNS", seed=sd, instance=inst, realization=real,
                                     cost=ibase * (1 + dqn_gap + rng.normal(0, 0.004)) * rng.normal(1, 0.01),
                                     viol=0, time_s=3.6))
    return pd.DataFrame(rows)


def selftest():
    df = _synthetic()
    print("--- Table wind rows ---")
    print(row_wind(df, "mu10", "10")); print(row_wind(df, "mu20", "20"))
    print("--- Table ablation rows ---")
    print(row_ablation(df, "mu10", "$\\mu_w=10$")); print(row_ablation(df, "mu20", "$\\mu_w=20$"))
    print("--- Table seeds rows ---")
    print(row_seeds(df, "mu10", "$\\mu_w=10$")); print(row_seeds(df, "mu20", "$\\mu_w=20$"))
    print("--- Table main rows (mu10 as a pseudo-config) ---")
    for m in ["DQN-LNS", "ALNS", "DQN-R"]:
        print(row_main(df, "mu10", m))
    # assertions on known structure
    a = per_instance(df, "DQN-LNS", "mu10"); b = per_instance(df, "ALNS", "mu10")
    st = paired_stats(a, b); assert -3.5 < st["gap"] < -1.5 and st["p"] < 0.01, st
    a20 = per_instance(df, "DQN-LNS", "mu20"); r20 = per_instance(df, "DQN-R", "mu20")
    eq = tost(a20, r20); assert eq["equivalent"], eq
    friedman_nemenyi(df, "mu10", methods=("ALNS", "DQN-R", "DQN-LNS"))
    print("\nSELF-TEST PASSED: known -2.5% effect recovered with p<0.01; equivalence at mu20 detected by TOST.")


# ---------------------------------------------------------------- CLI
# ---------------------------------------------------------------- training curves (fig:training)
def training_curves(path="results/train_curves.csv", out="fig_training.pdf"):
    """Validation score vs episode for every (variant, seed); selected checkpoint = minimum."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    d = pd.read_csv(path, header=None, names=["variant", "seed", "episode", "val", "eps", "elapsed"])
    variants = list(dict.fromkeys(d.variant))
    fig, axes = plt.subplots(1, len(variants), figsize=(3.1 * len(variants), 2.8), squeeze=False)
    for ax, var in zip(axes[0], variants):
        for sd, g in d[d.variant == var].groupby("seed"):
            g = g.sort_values("episode"); ax.plot(g.episode, g.val, lw=1.1, label=f"seed {sd}")
            i = g.val.idxmin(); ax.plot(g.episode[i], g.val[i], "o", ms=4, color="k")
        ax.set_title(var, fontsize=9); ax.set_xlabel("episode"); ax.set_ylabel("validation $Z_\\Omega(x^*)/Z(x_0)$")
        ax.axhline(1.0, ls=":", color="gray", lw=0.8)
    axes[0][0].legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(out); print(f"wrote {out}")




if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?")
    ap.add_argument("--point"); ap.add_argument("--all", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--curves", action="store_true")
    a = ap.parse_args()
    if a.curves:
        training_curves(); sys.exit(0)
    if a.selftest or not a.csv:
        selftest(); sys.exit(0)
    df = pd.read_csv(a.csv)
    points = sorted(df.point.unique()) if a.all else [a.point]
    for pt in points:
        print(f"\n%% ===== point {pt} =====")
        methods = [m for m in ["DQN-LNS", "DQN-R", "ALNS", "ALNS-SA", "Std-LNS", "Greedy"] if m in set(df[df.point == pt].method)]
        for m in methods: print(row_main(df, pt, m))
        if "DQN-R" in methods: print(row_ablation(df, pt, pt)); print(row_seeds(df, pt, pt))
        try: friedman_nemenyi(df, pt)
        except KeyError: pass
        for extra in ["DQN-noweather", "DQN-op", "DQN-masked", "DQN-broad"]:
            if extra in set(df[df.point == pt].method): print(row_two_methods(df, pt, pt, "DQN-LNS", extra))


