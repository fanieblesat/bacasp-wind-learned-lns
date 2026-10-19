"""
7_figures.py — the three summary figures of Section 5, from results/runs.csv only.

  fig_regime.pdf   : gap to ALNS on Z with 95% CI vs mu_w, sigma_w and |V| (DQN-LNS, DQN-R,
                     DQN-masked / DQN-noweather / DQN-op where evaluated)
  fig_decomp.pdf   : (a) environment / learning / total decomposition per point;
                     (b) per-instance gap dispersion with per-seed means, main points
  fig_cd.pdf       : Friedman/Nemenyi critical-difference diagrams at six points

Convention: gaps are on the turnaround Z of the returned schedule (as in every table);
the CI is a t-interval over instances of the per-instance paired relative difference
(mean over realizations and seeds), so it is the dispersion of the quantity the Wilcoxon
test ranks, expressed in percent of the ALNS mean.  CD diagrams rank on Z_Omega.
Usage: python 7_figures.py [results/runs.csv]
"""
import sys, math
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import stats_tools as S

df = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "results/runs.csv")
BASE = "ALNS"
COL = {"DQN-LNS": "C0", "DQN-R": "0.55", "DQN-masked": "C3", "DQN-noweather": "C2", "DQN-op": "C1",
       "ALNS": "k", "ALNS-SA": "C4", "Std-LNS": "C5", "Greedy": "C6"}
MK = {"DQN-LNS": "o", "DQN-R": "s", "DQN-masked": "^", "DQN-noweather": "v", "DQN-op": "D"}

def inst_gaps(m, pt, b=BASE):
    """Per-instance relative difference (%) of m to b on Z, aligned on instance."""
    a, bb = S.per_instance_Z(df, m, pt), S.per_instance_Z(df, b, pt)
    idx = a.index.intersection(bb.index)
    return (a.loc[idx].values - bb.loc[idx].values) / bb.loc[idx].mean() * 100

def gap_ci(m, pt, b=BASE):
    g = inst_gaps(m, pt, b); n = len(g)
    h = stats.t.ppf(0.975, n - 1) * g.std(ddof=1) / math.sqrt(n)
    return g.mean(), h

def has(m, pt): return m in set(df[df.point == pt].method)

# ------------------------------------------------------------------ fig_regime
def fig_regime():
    panels = [("mu", [("mu5", 5), ("mu10", 10), ("mu15", 15), ("mu20", 20)], "wind intensity $\\mu_w$ (m/s)"),
              ("sig", [("sig1.5", 1.5), ("sig3.5", 3.5), ("sig5.5", 5.5), ("sig8.0", 8.0)], "wind volatility $\\sigma_w$"),
              ("V", [("V20", 20), ("V30", 30), ("V50", 50), ("V75", 75), ("V100", 100)], "fleet size $|V|$")]
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.1), sharey=True)
    for ax, (_, pts, xl) in zip(axs, panels):
        for m in ["DQN-LNS", "DQN-R", "DQN-masked", "DQN-noweather", "DQN-op"]:
            xs, ys, hs = [], [], []
            for pt, x in pts:
                if not has(m, pt): continue
                y, h = gap_ci(m, pt); xs.append(x); ys.append(y); hs.append(h)
            if not xs: continue
            off = {"DQN-LNS": 0, "DQN-R": 0, "DQN-masked": 0.015, "DQN-noweather": -0.015, "DQN-op": 0.03}[m]
            xs = np.array(xs, float) * (1 + off)
            ax.errorbar(xs, ys, yerr=hs, color=COL[m], marker=MK[m], ms=4, lw=1.2, capsize=2.5, label=m,
                        ls="-" if m in ("DQN-LNS", "DQN-R") else "--")
        ax.axhline(0, color="k", lw=0.8, ls=":"); ax.set_xlabel(xl); ax.grid(alpha=0.25)
        if xl.startswith("fleet"): ax.set_xscale("log"); ax.set_xticks([20, 30, 50, 75, 100]); ax.set_xticklabels(["20", "30", "50", "75", "100"])
    axs[0].set_ylabel("gap to ALNS on $Z$ (%)")
    h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, ncol=5, fontsize=7.5, frameon=False, loc="lower center")
    for ax, t in zip(axs, "abc"): ax.set_title(f"({t})", fontsize=9, loc="left")
    axs[2].minorticks_off()
    fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig("fig_regime.pdf"); print("wrote fig_regime.pdf")

# ------------------------------------------------------------------ fig_decomp
def fig_decomp():
    pts = [("2B-4QC", "2B-4QC"), ("3B-6QC", "3B-6QC"), ("4B-8QC", "4B-8QC"), ("5B-10QC", "5B-10QC"),
           ("mu5", "$\\mu_w{=}5$"), ("mu15", "$\\mu_w{=}15$"), ("mu20", "$\\mu_w{=}20$"),
           ("sig1.5", "$\\sigma_w{=}1.5$"), ("sig5.5", "$\\sigma_w{=}5.5$"), ("sig8.0", "$\\sigma_w{=}8$"),
           ("V20", "$|V|{=}20$"), ("V30", "$|V|{=}30$"), ("V75", "$|V|{=}75$"), ("V100", "$|V|{=}100$")]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 3.4), gridspec_kw=dict(width_ratios=[1.15, 1]))
    x = np.arange(len(pts)); w = 0.27
    env = [inst_gaps("DQN-R", p).mean() for p, _ in pts]
    lrn = [(S.per_instance_Z(df, "DQN-LNS", p).mean() - S.per_instance_Z(df, "DQN-R", p).mean())
           / S.per_instance_Z(df, BASE, p).mean() * 100 for p, _ in pts]        # learning, in % of ALNS mean
    tot = [inst_gaps("DQN-LNS", p).mean() for p, _ in pts]
    a1.bar(x - w, env, w, color="0.6", label="environment (DQN-R $-$ ALNS)")
    a1.bar(x, lrn, w, color="C0", alpha=0.55, label="learning (DQN-LNS $-$ DQN-R)")
    a1.bar(x + w, tot, w, color="C0", label="total (DQN-LNS $-$ ALNS)")
    a1.axhline(0, color="k", lw=0.8); a1.set_xticks(x); a1.set_xticklabels([l for _, l in pts], rotation=60, fontsize=7, ha="right")
    a1.set_ylabel("gap in % of ALNS mean $Z$"); a1.set_ylim(top=6.2); a1.legend(fontsize=7, frameon=False, loc="upper right"); a1.grid(axis="y", alpha=0.25)
    # (b) dispersion strips
    pts2 = [("2B-4QC", "2B-4QC"), ("3B-6QC", "3B-6QC"), ("4B-8QC", "4B-8QC"), ("5B-10QC", "5B-10QC"),
            ("mu20", "$\\mu_w{=}20$"), ("sig8.0", "$\\sigma_w{=}8$"), ("V20", "$|V|{=}20$"), ("V100", "$|V|{=}100$")]
    rng = np.random.default_rng(0)
    for k, (p, l) in enumerate(pts2):
        g = inst_gaps("DQN-LNS", p)
        a2.scatter(k + rng.uniform(-0.18, 0.18, len(g)), g, s=7, color="C0", alpha=0.45, lw=0)
        base = S.per_instance_Z(df, BASE, p)
        for sd, gg in df[(df.point == p) & (df.method == "DQN-LNS")].groupby("seed"):
            s = gg.groupby("instance").Z.mean(); idx = s.index.intersection(base.index)
            a2.plot(k + 0.3, (s.loc[idx].mean() - base.loc[idx].mean()) / base.loc[idx].mean() * 100, "_", color="C3", ms=9, mew=1.3)
        a2.plot([k - 0.25, k + 0.25], [g.mean()] * 2, color="k", lw=1.5)
    a2.axhline(0, color="k", lw=0.8, ls=":"); a2.set_xticks(range(len(pts2))); a2.set_xticklabels([l for _, l in pts2], rotation=60, fontsize=7, ha="right")
    a2.set_ylabel("per-instance gap of DQN-LNS to ALNS (%)"); a2.grid(axis="y", alpha=0.25)
    a2.plot([], [], "o", color="C0", alpha=0.6, ms=3, label="instance (mean of seeds, realizations)")
    a2.plot([], [], "_", color="C3", ms=9, mew=1.3, label="training seed (mean of instances)")
    a2.plot([], [], color="k", lw=1.5, label="mean"); a2.set_ylim(-15.5, 7); a2.legend(fontsize=6.5, frameon=False, loc="lower left")
    a1.set_title("(a)", fontsize=9, loc="left"); a2.set_title("(b)", fontsize=9, loc="left")
    fig.tight_layout(); fig.savefig("fig_decomp.pdf"); print("wrote fig_decomp.pdf")

# ------------------------------------------------------------------ fig_cd
def cd_diagram(ax, ranks, cd, title):
    """Demsar-style CD diagram. ranks: dict method -> mean rank (1 = best)."""
    k = len(ranks); lo, hi = 1, k
    ax.set_xlim(hi + 1.9, lo - 1.9); ax.set_ylim(-3.6, 1.2); ax.axis("off")
    ax.plot([lo, hi], [0, 0], color="k", lw=1)
    for r in range(lo, hi + 1):
        ax.plot([r, r], [0, 0.15], color="k", lw=1); ax.text(r, 0.3, str(r), ha="center", va="bottom", fontsize=7)
    ax.plot([lo, lo + cd], [0.85, 0.85], color="k", lw=1.5); ax.text(lo + cd / 2, 0.95, f"CD = {cd:.2f}", ha="center", va="bottom", fontsize=7)
    order = sorted(ranks, key=ranks.get)
    for i, m in enumerate(order):
        r = ranks[m]; left = i >= math.ceil(k / 2)          # worst-ranked methods labelled on the left (high ranks)
        y = -0.9 - (k - 1 - i if left else i) * 0.55
        xt = hi + 0.25 if left else lo - 0.25
        ax.plot([r, r], [0, y], color="k", lw=0.7); ax.plot([r, xt], [y, y], color="k", lw=0.7)
        ax.text(xt, y, f"{m} ({r:.2f})", ha="right" if left else "left", va="center", fontsize=7)
    # cliques: maximal groups of consecutive methods within CD
    y = -0.25; drawn = []
    for i in range(k):
        j = i
        while j + 1 < k and ranks[order[j + 1]] - ranks[order[i]] <= cd: j += 1
        if j > i and not any(i >= a and j <= b for a, b in drawn):
            ax.plot([ranks[order[i]] - 0.05, ranks[order[j]] + 0.05], [y, y], color="k", lw=2.5); drawn.append((i, j)); y -= 0.15
    ax.set_title(title, fontsize=8)

def fpm(p):
    if p >= 0.001: return f"{p:.3f}"
    e = int(math.floor(math.log10(p))); return f"{p / 10 ** e:.1f}\\times 10^{{{e}}}"

def fig_cd():
    pts = [("3B-6QC", "3B-6QC ($n{=}50$)"), ("5B-10QC", "5B-10QC ($n{=}50$)"), ("mu20", "$\\mu_w = 20$ ($n{=}20$)"),
           ("sig8.0", "$\\sigma_w = 8.0$ ($n{=}20$)"), ("V20", "$|V| = 20$ ($n{=}20$)"), ("V100", "$|V| = 100$ ($n{=}20$)")]
    methods = ("Greedy", "Std-LNS", "ALNS", "ALNS-SA", "DQN-LNS")
    fig, axs = plt.subplots(2, 3, figsize=(10.5, 3.8))
    import io, contextlib
    for ax, (p, t) in zip(axs.flat, pts):
        with contextlib.redirect_stdout(io.StringIO()):
            r = S.friedman_nemenyi(df, p, methods)
        cd_diagram(ax, r["ranks"], r["cd"], f"{t}, Friedman $p = {fpm(r['p'])}$")
    fig.tight_layout(); fig.savefig("fig_cd.pdf"); print("wrote fig_cd.pdf")

if __name__ == "__main__":
    fig_regime(); fig_decomp(); fig_cd()
