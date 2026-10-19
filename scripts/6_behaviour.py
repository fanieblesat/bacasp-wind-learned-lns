"""
6_behaviour.py — what the learned policy does (Section 5, "What the learned policy does").
Records every decision of the seed-0 full policy and of the random policy over the wind sweep
(mu5, mu10, mu15, mu20; 20 instances x 3 realizations; 500 iterations) together with the search
state before the decision and the outcome of the move. Fills tab:usage and fig_action_usage.pdf.

Usage: python 6_behaviour.py [--seed 0]      -> results/behaviour.csv, results/usage_rows.tex, fig_action_usage.pdf
"""
import argparse, pickle, csv
import numpy as np
from common import *
from interface import make_instance, make_wind
from jos_core import controller
from jos_core.controller import Env, QNet, greedy_action, decode, n_actions, state_dim, TAU, SEV

OPS = ["random reassign", "vessel swap", "storm-aware", "consolidate", "delay storm", "crane rebalance"]
POINTS = ["mu5", "mu10", "mu15", "mu20"]

def run(policy, tag, rows):
    import random
    if policy is None:
        net = None
    else:
        net = QNet(state_dim("full"), n_actions("full")); net.load_state_dict(policy["state_dict"]); net.eval()
    for pt in POINTS:
        cfg = SWEEP[pt]
        for isd in SWEEP_SEEDS:
            inst = make_instance(isd, cfg)
            for real in range(N_REAL):
                wind = make_wind(wind_seed(isd, real), cfg)
                env = Env(inst, wind, "full", 500, seed=real); s = env.reset(); done = False
                prng = random.Random(1_000_003 + real)
                while not done:
                    i_frac = env.i / env.I; stag = env.stag; s_hat = env.s_hat
                    a = prng.randrange(n_actions("full")) if net is None else greedy_action(net, s)
                    op, sev, tau = decode(a, "full")
                    s, _, done = env.step(a)
                    rows.append((tag, pt, isd, real, env.i, op, sev, tau, int(env.last_acc), int(env.last_imp),
                                 round(i_frac, 3), stag, round(s_hat, 4)))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--from-csv", action="store_true", help="only rebuild table/figure from results/behaviour.csv"); a = ap.parse_args()
    import pandas as pd
    if not a.from_csv:
        policy = pickle.load(open(RESULTS / f"policy_full_seed{a.seed}.pkl", "rb"))
        rows = []
        run(policy, "trained", rows); run(None, "random", rows)
        with open(RESULTS / "behaviour.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow("policy point instance real iter op sev tau acc newbest ifrac stag shat".split()); w.writerows(rows)
    df = pd.read_csv(RESULTS / "behaviour.csv")
    out = []
    def block(name, col, labels):
        out.append(f"\\multicolumn{{7}}{{@{{}}l}}{{\\emph{{{name}}}}} \\\\")
        for val, lab in labels:
            cells = []
            for pol in ("trained", "random"):
                d = df[df.policy == pol]; sel = d[d[col] == val]
                share = f"{len(sel)/len(d)*100:.1f}\\%"
                if len(sel) < 100: cells += [share, "---", "---"]           # too few selections for a rate
                else: cells += [share, f"{sel.acc.mean()*100:.1f}\\%", f"{sel.newbest.mean()*100:.1f}\\%"]
            out.append(f"\\quad {lab} & " + " & ".join(cells) + " \\\\")
    block("Operator", "op", list(enumerate(OPS)))
    block("Severity", "sev", [(s, f"${s}\\times$") for s in SEV])
    block("Acceptance", "tau", [(t, f"$\\tau = {t}$") for t in TAU])
    open(RESULTS / "usage_rows.tex", "w").write("\n".join(out) + "\n"); print("\n".join(out))
    n = len(df[df.policy == "trained"]); print(f"decisions per policy: {n}")

    # figure: (a) greedy share & mean severity vs horizon, (b) mean tau vs stagnation, (c) operator mix vs storm share
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.0))
    tr = df[df.policy == "trained"]; rd = df[df.policy == "random"]
    bins = np.linspace(0, 1, 11); mids = (bins[:-1] + bins[1:]) / 2
    for d, c, lab in ((tr, "C0", "trained"), (rd, "0.6", "random")):
        g = d.groupby(pd.cut(d.ifrac, bins, include_lowest=True), observed=True)
        axs[0].plot(mids, g.tau.apply(lambda x: (x == 0).mean()).values, color=c, label=f"greedy share ({lab})")
        axs[0].plot(mids, g.sev.mean().values / 3, color=c, ls="--", label=f"mean severity/3 ({lab})")
    axs[0].set_xlabel("search progress $i/I$"); axs[0].set_ylim(0, 1); axs[0].legend(fontsize=6, frameon=False)
    sb = [0, 1, 2, 5, 10, 20, 35, 51]; smid = [(sb[k] + sb[k + 1]) / 2 for k in range(len(sb) - 1)]
    for d, c, lab in ((tr, "C0", "trained"), (rd, "0.6", "random")):
        g = d.groupby(pd.cut(d.stag, sb, right=False), observed=True)
        axs[1].plot(smid, g.tau.mean().values, color=c, marker="o", ms=3, label=lab)
    axs[1].set_xlabel("iterations since last improvement"); axs[1].set_ylabel("mean $\\tau$"); axs[1].legend(fontsize=7, frameon=False)
    qs = np.quantile(tr.shat, [0, .2, .4, .6, .8, 1.0]); qs[-1] += 1e-9
    g = tr.groupby(pd.cut(tr.shat, qs, include_lowest=True), observed=True)
    mix = np.array([g.op.apply(lambda x: (x == k).mean()).values for k in range(6)])
    xm = [(qs[k] + qs[k + 1]) / 2 for k in range(5)]
    axs[2].stackplot(xm, mix, labels=OPS, alpha=0.85)
    axs[2].set_xlabel("storm share $\\hat s$ of the window"); axs[2].set_ylabel("operator mix (trained)"); axs[2].set_ylim(0, 1)
    axs[2].legend(fontsize=5.5, frameon=False, loc="upper right")
    for ax, t in zip(axs, "abc"): ax.set_title(f"({t})", fontsize=9, loc="left")
    fig.tight_layout(); fig.savefig("fig_action_usage.pdf"); print("wrote fig_action_usage.pdf")

if __name__ == "__main__":
    main()
