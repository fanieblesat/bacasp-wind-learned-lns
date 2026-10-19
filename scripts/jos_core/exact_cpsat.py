"""Aggregated (count) exact model in CP-SAT — same constraints as exact.solve(aggregated=True),
exact integer arithmetic (mu has 3 decimals, RHO*mu*1000 is an integer), multi-threaded,
with optimality proof and bound. This is the solver used for tab:validation.

solve_cpsat(inst, wind, T, time_limit_s, workers, hint=(x,sigma,q), fix=(x,sigma,q))
returns the same dict as exact.solve (status, obj, bound, time_s, x, sigma, q, C).
"""
from __future__ import annotations
from ortools.sat.python import cp_model
from .params import RHO

SCALE = 1000


def solve_cpsat(inst, wind, T: int, time_limit_s: float = 3600, workers: int = 8,
                hint=None, fix=None):
    V = list(range(inst.n)); B = list(range(inst.n_berths)); nK = inst.n_cranes
    Ts = list(range(1, T + 1))
    A, h, qmin, qmax = inst.A, inst.h, inst.qmin, inst.qmax
    S = wind.S; phi = wind.phi
    w_int = {(b, t): int(round(RHO * wind.mu[b][t] * SCALE)) for b in B for t in Ts}
    h_int = {v: int(round(h[v] * SCALE)) for v in V}
    m = cp_model.CpModel()
    x = {(v, b): m.NewBoolVar(f"x{v}_{b}") for v in V for b in B}
    sig = {(v, b, t): m.NewBoolVar(f"s{v}_{b}_{t}") for v in V for b in B for t in Ts if t >= A[v] and not S[b][t]}
    dlt = {(v, t): m.NewBoolVar(f"d{v}_{t}") for v in V for t in Ts if t >= A[v]}
    moor = {(v, t): m.NewBoolVar(f"m{v}_{t}") for v in V for t in Ts if t >= A[v]}
    nq = {(v, t): m.NewIntVar(0, nK, f"n{v}_{t}") for v in V for t in Ts if t >= A[v]}
    y = {(v, b, t): m.NewIntVar(0, nK, f"y{v}_{b}_{t}") for v in V for b in B for t in Ts if t >= A[v]}
    C = {v: m.NewIntVar(0, T, f"C{v}") for v in V}
    for v in V:
        svs = [var for (vv, b, t), var in sig.items() if vv == v]
        m.AddExactlyOne(svs)                                                       # one_berthing
        for b in B:
            m.Add(x[v, b] == sum(var for (vv, bb, t), var in sig.items() if vv == v and bb == b))   # x_def
        for t in Ts:
            if t < A[v]:
                continue
            st = [sig[v, b, t] for b in B if (v, b, t) in sig]
            m.Add(dlt[v, t] >= sum(st))                                            # occ_start
            prev = dlt[v, t - 1] if (v, t - 1) in dlt else 0
            m.Add(dlt[v, t] <= prev + sum(st))                                     # occ_contig
            # mooring indicator m_vt = sum of sigma at (b,tau) with tau <= t < tau + phi
            m.Add(moor[v, t] == sum(sig[v, b, tau] for b in B for tau in Ts
                                    if (v, b, tau) in sig and tau <= t < tau + phi[b][tau]))
            m.Add(dlt[v, t] >= moor[v, t])                                         # stay_mooring
            storm_here = sum(S[b][t] * x[v, b] for b in B)
            m.Add(nq[v, t] <= nK * (dlt[v, t] - moor[v, t]))                       # work_when_berthed
            m.Add(nq[v, t] <= nK * (1 - storm_here))                               # no_work_storm
            m.Add(nq[v, t] <= qmax[v] * (dlt[v, t] - moor[v, t]))                  # gang_max
            m.Add(nq[v, t] >= qmin[v] * (dlt[v, t] - moor[v, t] - storm_here))     # gang_min
            for b in B:                                                            # y = x*n
                m.Add(y[v, b, t] <= nK * x[v, b]); m.Add(y[v, b, t] <= nq[v, t])
                m.Add(y[v, b, t] >= nq[v, t] - nK * (1 - x[v, b]))
        m.Add(sum(w_int[b, t] * y[v, b, t] for b in B for t in Ts if t >= A[v]) >= h_int[v])   # workload
        m.Add(C[v] == sum((t - 1) * var for (vv, b, t), var in sig.items() if vv == v)
              + sum(dlt[v, t] for t in Ts if t >= A[v]))                           # completion
    # redundant scheduling view (strengthens propagation; implied by the constraints above):
    # one optional interval per (v,b) with start = berthing period, size = occupation length,
    # NoOverlap per berth; size >= minimum service length at full productivity and q_max.
    import math as _math
    for v in V:
        pmin = int(_math.ceil(h[v] / (RHO * qmax[v]) - 1e-9))
        occ = sum(dlt[v, t] for t in Ts if t >= A[v])
        m.Add(occ >= pmin)
        start_v = sum(t * var for (vv, b, t), var in sig.items() if vv == v)
        for b in B:
            st_vb = m.NewIntVar(A[v], T, f"st{v}_{b}"); sz_vb = m.NewIntVar(1, T, f"sz{v}_{b}")
            en_vb = m.NewIntVar(A[v] + 1, T + 1, f"en{v}_{b}")
            itv = m.NewOptionalIntervalVar(st_vb, sz_vb, en_vb, x[v, b], f"itv{v}_{b}")
            m.Add(st_vb == start_v).OnlyEnforceIf(x[v, b])
            m.Add(sz_vb == occ).OnlyEnforceIf(x[v, b])
            m.Add(en_vb <= T + 1)
            m.Add(C[v] == en_vb - 1).OnlyEnforceIf(x[v, b])
            if not hasattr(m, "_itv"): m._itv = {}
            m._itv.setdefault(b, []).append(itv)
    for b in B:
        if hasattr(m, "_itv") and b in m._itv:
            m.AddNoOverlap(m._itv[b])
    for b in B:
        for t in Ts:                                                               # berth_cap: sum_v x_vb * delta_vt <= 1
            zs = []
            for v in V:
                if t < A[v]:
                    continue
                z = m.NewBoolVar("")
                m.AddBoolAnd([x[v, b], dlt[v, t]]).OnlyEnforceIf(z)
                m.AddBoolOr([x[v, b].Not(), dlt[v, t].Not()]).OnlyEnforceIf(z.Not())
                zs.append(z)
            if zs:
                m.AddAtMostOne(zs)
    for t in Ts:
        m.Add(sum(nq[v, t] for v in V if t >= A[v]) <= nK)                         # crane capacity
    if fix is not None:
        fx, fsig, fq = fix
        cnt = {}
        for (v, k, t), val in fq.items():
            if val: cnt[(v, t)] = cnt.get((v, t), 0) + 1
        for key, var in x.items(): m.Add(var == fx.get(key, 0))
        for key, var in sig.items(): m.Add(var == fsig.get(key, 0))
        for key, val in fsig.items():
            if val and key not in sig:
                return dict(status="INFEASIBLE", obj=float("nan"), bound=float("nan"), time_s=0.0, why=f"sigma{key} inadmissible")
        for key, var in nq.items(): m.Add(var == cnt.get(key, 0))
    if hint is not None:
        hx, hsig, hq = hint
        cnt = {}
        for (v, k, t), val in hq.items():
            if val: cnt[(v, t)] = cnt.get((v, t), 0) + 1
        for key, var in x.items(): m.AddHint(var, hx.get(key, 0))
        for key, var in sig.items(): m.AddHint(var, hsig.get(key, 0))
        for key, var in nq.items(): m.AddHint(var, cnt.get(key, 0))
    m.Minimize(sum(C[v] - A[v] for v in V))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_s)
    solver.parameters.num_workers = int(workers)
    st = solver.Solve(m)
    status = {cp_model.OPTIMAL: "OPTIMAL", cp_model.FEASIBLE: "FEASIBLE", cp_model.INFEASIBLE: "INFEASIBLE"}.get(st, "NONE")
    out = dict(status=status, time_s=solver.WallTime(), obj=float("nan"), bound=float("nan"))
    if status in ("OPTIMAL", "FEASIBLE"):
        out.update(obj=float(solver.ObjectiveValue()), bound=float(solver.BestObjectiveBound()))
        out["x"] = {k: int(solver.Value(var)) for k, var in x.items()}
        out["sigma"] = {k: int(solver.Value(var)) for k, var in sig.items()}
        out["C"] = {v: float(solver.Value(C[v])) for v in V}
        berth_of = {v: next(b for b in B if out["x"][v, b]) for v in V}
        qd = {}
        for t in Ts:
            k0 = 0
            for b in B:
                for v in V:
                    if berth_of[v] == b and (v, t) in nq:
                        cnt = int(solver.Value(nq[v, t]))
                        for k in range(k0, k0 + cnt): qd[(v, k, t)] = 1
                        k0 += cnt
        out["q"] = qd
    return out
