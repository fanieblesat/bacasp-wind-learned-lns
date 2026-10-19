"""Exact model of Section 3 in OR-Tools/SCIP, reading params.py and the same
Instance/Wind objects as the simulator.

Two forms:
  solve(..., aggregated=True)   count model: n_vt = sum_k q_vkt (Corollary to Prop. noncross:
                                 any per-period count vector admits a rail-ordered, hence
                                 non-crossing, crane labelling, so the crane-specific model and
                                 the count model have the same optimal value). This is the form
                                 solved in the validation experiment.
  solve(..., aggregated=False)  the crane-specific model exactly as written (xi, noncross);
                                 used only to confirm the corollary numerically on tiny instances.
fix=(x_vb, sigma_vbt, q_vkt) fixes a simulator schedule inside the model (feasibility +
objective agreement). hint=(x,sigma,q) warm-starts the solver.
"""
from __future__ import annotations
from ortools.linear_solver import pywraplp
from .params import RHO

STATUS = {pywraplp.Solver.OPTIMAL: "OPTIMAL", pywraplp.Solver.FEASIBLE: "FEASIBLE",
          pywraplp.Solver.INFEASIBLE: "INFEASIBLE"}


def solve(inst, wind, T: int, time_limit_s: float = 3600, fix=None, hint=None,
          aggregated=True, threads=None):
    V = list(range(inst.n)); B = list(range(inst.n_berths)); K = list(range(inst.n_cranes)); nK = len(K)
    Ts = list(range(1, T + 1))
    A, h, qmin, qmax = inst.A, inst.h, inst.qmin, inst.qmax
    S = {(b, t): wind.S[b][t] for b in B for t in Ts}
    mu = {(b, t): wind.mu[b][t] for b in B for t in Ts}
    phi = {(b, t): wind.phi[b][t] for b in B for t in Ts}
    s = pywraplp.Solver.CreateSolver("SCIP"); INF = s.infinity()
    if threads:
        s.SetNumThreads(threads)
    Z0 = s.IntVar(0, 0, "zero")
    # variables; everything before arrival is fixed to zero (not created)
    x = {(v, b): s.BoolVar(f"x{v}_{b}") for v in V for b in B}
    sig = {(v, b, t): (s.BoolVar("") if t >= A[v] and not S[b, t] else Z0) for v in V for b in B for t in Ts}
    dlt = {(v, t): (s.BoolVar("") if t >= A[v] else Z0) for v in V for t in Ts}
    zeta = {(v, b, t): (s.BoolVar("") if t >= A[v] else Z0) for v in V for b in B for t in Ts}
    C = {v: s.NumVar(0, INF, f"C{v}") for v in V}
    m = lambda v, t: sum(sig[v, b, tau] for b in B for tau in Ts if tau <= t < tau + phi[b, tau] and tau >= A[v])
    if aggregated:
        nq = {(v, t): (s.IntVar(0, nK, "") if t >= A[v] else Z0) for v in V for t in Ts}
        y = {(v, b, t): (s.IntVar(0, nK, "") if t >= A[v] else Z0) for v in V for b in B for t in Ts}
    else:
        q = {(v, k, t): (s.BoolVar("") if t >= A[v] else Z0) for v in V for k in K for t in Ts}
        xi = {(v, b, k, t): (s.BoolVar("") if t >= A[v] else Z0) for v in V for b in B for k in K for t in Ts}
    for v in V:
        s.Add(sum(sig[v, b, t] for b in B for t in Ts) == 1)                       # one_berthing (+arrival, no_berth_storm by construction)
        for b in B:
            s.Add(x[v, b] == sum(sig[v, b, t] for t in Ts))                        # x_def
        for t in Ts:
            if t < A[v]:
                continue
            s.Add(dlt[v, t] >= sum(sig[v, b, t] for b in B))                       # occ_start
            s.Add(dlt[v, t] <= (dlt[v, t - 1] if t > 1 else 0) + sum(sig[v, b, t] for b in B))  # occ_contig
            mvt = m(v, t)
            s.Add(dlt[v, t] >= mvt)                                                # stay_mooring
            for b in B:                                                            # zeta_lin
                s.Add(zeta[v, b, t] <= x[v, b]); s.Add(zeta[v, b, t] <= dlt[v, t])
                s.Add(zeta[v, b, t] >= x[v, b] + dlt[v, t] - 1)
            storm_here = sum(S[b, t] * x[v, b] for b in B)
            if aggregated:
                n = nq[v, t]
                s.Add(n <= nK * (dlt[v, t] - mvt))                                 # work_when_berthed
                s.Add(n <= nK * (1 - storm_here))                                  # no_work_storm
                s.Add(n <= qmax[v] * (dlt[v, t] - mvt))                            # gang_max
                s.Add(n >= qmin[v] * (dlt[v, t] - mvt - storm_here))               # gang_min
                for b in B:                                                        # y = x * n (exact for integer n)
                    s.Add(y[v, b, t] <= nK * x[v, b]); s.Add(y[v, b, t] <= n)
                    s.Add(y[v, b, t] >= n - nK * (1 - x[v, b]))
            else:
                for k in K:
                    s.Add(q[v, k, t] <= dlt[v, t] - mvt)                           # work_when_berthed
                    for b in B:                                                    # xi_lin, no_work_storm
                        s.Add(xi[v, b, k, t] <= x[v, b]); s.Add(xi[v, b, k, t] <= q[v, k, t])
                        s.Add(xi[v, b, k, t] >= x[v, b] + q[v, k, t] - 1)
                        if S[b, t]: s.Add(xi[v, b, k, t] == 0)
                s.Add(sum(q[v, k, t] for k in K) <= qmax[v] * (dlt[v, t] - mvt))                # gang_max
                s.Add(sum(q[v, k, t] for k in K) >= qmin[v] * (dlt[v, t] - mvt - storm_here))   # gang_min
        if aggregated:
            s.Add(sum(RHO * mu[b, t] * y[v, b, t] for t in Ts for b in B if t >= A[v]) >= h[v])          # workload
        else:
            s.Add(sum(RHO * mu[b, t] * xi[v, b, k, t] for t in Ts for k in K for b in B if t >= A[v]) >= h[v])
        s.Add(C[v] == sum((t - 1) * sig[v, b, t] for b in B for t in Ts if t >= A[v]) + sum(dlt[v, t] for t in Ts if t >= A[v]))  # completion
    for b in B:
        for t in Ts: s.Add(sum(zeta[v, b, t] for v in V) <= 1)                     # berth_cap
    if aggregated:
        for t in Ts: s.Add(sum(nq[v, t] for v in V) <= nK)                         # crane capacity (= crane_excl aggregated)
    else:
        for k in K:
            for t in Ts: s.Add(sum(q[v, k, t] for v in V) <= 1)                    # crane_excl
        for t in Ts:                                                               # noncross
            for k in K:
                for kp in K:
                    if kp <= k: continue
                    for b in B:
                        for bp in B:
                            if bp < b:
                                s.Add(sum(xi[v, b, k, t] for v in V) + sum(xi[u, bp, kp, t] for u in V) <= 1)
    if fix is not None:
        fx, fsig, fq = fix
        for key, var in x.items(): s.Add(var == fx.get(key, 0))
        for key, var in sig.items():
            if var is not Z0: s.Add(var == fsig.get(key, 0))
            elif fsig.get(key, 0): return dict(status="INFEASIBLE", obj=float("nan"), bound=float("nan"), time_s=0.0, why=f"sigma{key} fixed to 1 but inadmissible")
        if aggregated:
            cnt = {}
            for (v, k, t), val in fq.items():
                if val: cnt[(v, t)] = cnt.get((v, t), 0) + 1
            for key, var in nq.items():
                if var is not Z0: s.Add(var == cnt.get(key, 0))
        else:
            for key, var in q.items():
                if var is not Z0: s.Add(var == fq.get(key, 0))
    if hint is not None:
        hx, hsig, hq = hint
        vars_, vals = [], []
        for key, var in x.items(): vars_.append(var); vals.append(hx.get(key, 0))
        for key, var in sig.items():
            if var is not Z0: vars_.append(var); vals.append(hsig.get(key, 0))
        if aggregated:
            cnt = {}
            for (v, k, t), val in hq.items():
                if val: cnt[(v, t)] = cnt.get((v, t), 0) + 1
            for key, var in nq.items():
                if var is not Z0: vars_.append(var); vals.append(cnt.get(key, 0))
        s.SetHint(vars_, vals)
    s.Minimize(sum(C[v] - A[v] for v in V)); s.SetTimeLimit(int(time_limit_s * 1000))
    st = s.Solve()
    status = STATUS.get(st, "NONE")
    out = dict(status=status, time_s=s.wall_time() / 1000, obj=float("nan"), bound=float("nan"))
    if status in ("OPTIMAL", "FEASIBLE"):
        out.update(obj=s.Objective().Value(), bound=s.Objective().BestBound())
        iv = lambda var: int(round(var.solution_value())) if var is not Z0 else 0
        out["x"] = {k: iv(var) for k, var in x.items()}
        out["sigma"] = {k: iv(var) for k, var in sig.items()}
        out["C"] = {v: C[v].solution_value() for v in V}
        if aggregated:
            # recover crane identities by rail order (Corollary): berths in quay order, cranes consecutive
            qd = {}
            berth_of = {v: next(b for b in B if out["x"][v, b]) for v in V}
            for t in Ts:
                k0 = 0
                for b in B:
                    for v in V:
                        if berth_of[v] == b:
                            cnt = iv(nq[v, t])
                            for k in range(k0, k0 + cnt): qd[(v, k, t)] = 1
                            k0 += cnt
            out["q"] = qd
        else:
            out["q"] = {k: iv(var) for k, var in q.items()}
    return out
