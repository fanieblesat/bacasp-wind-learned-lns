"""Period-indexed simulator: the executable form of Eqs (moor_def)–(completion).

A *solution* x (what the search manipulates) is
    berth[v]     assigned berth                       -> x_vb
    prio[v]      priority key; vessels at a berth are served in increasing prio
                 (default: arrival time)              -> berthing order
    earliest[v]  earliest admissible berthing period (>= A_v); operators that
                 "delay" a vessel raise it             -> sigma_vbt
    weight[v]    weight in the proportional crane-allocation rule (default 1)
The simulator derives the full period-by-period schedule (sigma, delta, m, q)
from x under the wind trajectory, applying the model's rules:
  * berthing at the first period t >= max(earliest, previous vessel released)   (berth_cap, arrival)
  * a berthing in a storm period is NOT prevented but counted in nu            (relaxation of no_berth_storm)
  * mooring phi_bt evaluated at the berthing period and berth; no cranes then  (moor_def, stay_mooring)
  * zero cranes in storm periods at the vessel's berth                         (no_work_storm)
  * per period, cranes shared among vessels in service by the proportional
    rule within [q_min, q_max], crane identities in rail order along the quay  (gang_min/max, crane_excl, noncross)
  * work delivered RHO * mu_bt per crane; released in the first period in
    which cumulative work >= h_v                                               (workload, completion)
Objective Z = sum_v (C_v - A_v); Z_Omega = Z + omega * nu with omega = |V|(|T|-1), |T| = simulated periods.
"""
from __future__ import annotations
import math
from .params import RHO, omega


class Solution:
    __slots__ = ("berth", "prio", "earliest", "weight", "_res")

    def __init__(self, berth, prio, earliest, weight):
        self.berth = list(berth); self.prio = list(prio)
        self.earliest = list(earliest); self.weight = list(weight)
        self._res = None   # cached evaluation (invalidate on any change)

    @classmethod
    def empty(cls, inst):
        return cls([0] * inst.n, [float(a) for a in inst.A], list(inst.A), [1.0] * inst.n)

    def copy(self):
        s = Solution(self.berth, self.prio, self.earliest, self.weight)
        s._res = self._res
        return s

    def touch(self):
        self._res = None


class Result:
    __slots__ = ("Z", "nu", "Z_omega", "C", "tb", "unfinished", "q_hist", "wait_events")

    def __init__(self, Z, nu, C, tb, unfinished, q_hist=None, wait_events=0, omega_=1e9):
        self.Z = Z; self.nu = nu; self.Z_omega = Z + omega_ * nu
        self.C = C; self.tb = tb; self.unfinished = unfinished
        self.q_hist = q_hist; self.wait_events = wait_events


def allocate_cranes(active, K):
    """Proportional crane allocation rule (Section 4.1).
    active: list of (v, qmin, qmax_eff, score, tb). Returns ({v: q}, n_waiting).
    Each vessel first receives q_min (if the cranes are short, the latest-berthed wait);
    the remaining cranes are distributed in proportion to score = weight * remaining work
    by largest remainder, capped at qmax_eff."""
    if len(active) == 1:
        v, qn, qx, sc, tb = active[0]
        return {v: min(qx, K)}, 0
    q = {}; left = K; waited = 0
    tot_min = 0
    for a in active:
        tot_min += a[1]
    if tot_min > K:
        act = sorted(active, key=lambda a: (a[4], a[0]))
        for v, qn, qx, sc, tb in act:
            if left >= qn:
                q[v] = qn; left -= qn
            else:
                q[v] = 0; waited += 1
    else:
        for v, qn, qx, sc, tb in active:
            q[v] = qn
        left = K - tot_min
    if left <= 0:
        return q, waited
    tot = 0.0
    for a in active:
        if q[a[0]] > 0: tot += a[3]
    if tot <= 0:
        return q, waited
    rem = []
    for v, qn, qx, sc, tb in active:
        if q[v] == 0: continue
        want = q[v] + left * sc / tot
        fl = int(want)
        if fl > qx: fl = qx
        if fl > q[v]:
            left -= fl - q[v]; q[v] = fl
        if fl < qx:
            rem.append((want - fl, -v, v))
    # 'left' may have become negative if floors exceeded the pool? No: sum of floors <= sum of wants = K - sum_min + sum q
    if left > 0 and rem:
        rem.sort(reverse=True)
        for _, _, v in rem:
            if left <= 0: break
            q[v] += 1; left -= 1
    return q, waited


def evaluate(inst, wind, sol: Solution, subset=None, export=False) -> Result:
    """Simulate solution `sol` (optionally only the vessels in `subset`) and return a Result."""
    if subset is None and sol._res is not None and not export:
        return sol._res
    n = inst.n; nb = inst.n_berths; K = inst.n_cranes
    ids = list(range(n)) if subset is None else sorted(subset)
    # sequence per berth
    seq = [[] for _ in range(nb)]
    for v in ids:
        seq[sol.berth[v]].append(v)
    for b in range(nb):
        seq[b].sort(key=lambda v: (sol.prio[v], inst.A[v], v))
    ptr = [0] * nb                      # next vessel index in seq[b]
    cur = [-1] * nb                     # vessel currently at berth b
    moor_left = [0] * nb
    rem = {}                            # remaining work (moves)
    tb = {}; C = {}
    nu = 0; wait_events = 0
    L = len(wind)
    S, MU, PHI = wind.S, wind.mu, wind.phi
    q_hist = {} if export else None
    done = 0; target = len(ids)
    A = inst.A; h = inst.h; qmin = inst.qmin; qmax = inst.qmax; wt = sol.weight; e = sol.earliest
    t = 1
    while done < target and t <= L:
        # 1. berthing decisions
        for b in range(nb):
            if cur[b] == -1 and ptr[b] < len(seq[b]):
                v = seq[b][ptr[b]]
                if t >= e[v]:
                    cur[b] = v; ptr[b] += 1; tb[v] = t; rem[v] = h[v]
                    if S[b][t]:
                        nu += 1
                    moor_left[b] = PHI[b][t]
        # 2. who can work this period
        active = []
        for b in range(nb):
            v = cur[b]
            if v == -1:
                continue
            if moor_left[b] > 0:
                moor_left[b] -= 1
                continue
            if S[b][t]:
                continue
            mu = MU[b][t]
            need = int(math.ceil(rem[v] / (RHO * mu) - 1e-9))
            qx = max(qmin[v], min(qmax[v], need))
            active.append((v, qmin[v], qx, wt[v] * rem[v], tb[v]))
        if active:
            q, waited = allocate_cranes(active, K)
            wait_events += waited
            k0 = 0
            for b in range(nb):            # rail order: crane ids increase with berth index
                v = cur[b]
                if v == -1 or v not in q:
                    continue
                qv = q[v]
                if qv == 0:
                    continue
                if export:
                    q_hist[(v, t)] = list(range(k0, k0 + qv))
                k0 += qv
                rem[v] -= RHO * MU[b][t] * qv
                if rem[v] <= 1e-9:
                    C[v] = t; cur[b] = -1; done += 1
        t += 1
    unfinished = 0
    for b in range(nb):
        if cur[b] != -1:
            C[cur[b]] = L; unfinished += 1
        for v in seq[b][ptr[b]:]:
            C[v] = L; tb[v] = L; unfinished += 1
    Z = float(sum(C[v] - A[v] for v in ids))
    res = Result(Z, nu + unfinished, C, tb, unfinished, q_hist, wait_events, omega(n, L))
    if subset is None and not export:
        sol._res = res
    return res


# ------------------------------------------------------------------ checker for MIP variables
def evaluate_fixed(inst, wind, x_vb: dict, sigma_vbt: dict, q_vkt: dict):
    """Follow the solver's decision variables with the model's rules and return
    (Z, nu, problems). `problems` lists any constraint the variables violate under
    the simulator's reading of the model; it must be empty and Z must equal the
    solver objective for the agreement check to pass."""
    n = inst.n; nb = inst.n_berths; K = inst.n_cranes
    L = len(wind); S, MU, PHI = wind.S, wind.mu, wind.phi
    problems = []
    berth = {}; tb = {}
    for v in range(n):
        bs = [b for b in range(nb) if x_vb.get((v, b), 0) == 1]
        ts = [(b, t) for (vv, b, t), val in sigma_vbt.items() if vv == v and val == 1]
        if len(bs) != 1 or len(ts) != 1 or ts[0][0] != bs[0]:
            problems.append(f"vessel {v}: x/sigma inconsistent {bs} {ts}"); continue
        berth[v] = bs[0]; tb[v] = ts[0][1]
        if tb[v] < inst.A[v]:
            problems.append(f"vessel {v} berthed before arrival")
    nu = 0; C = {}
    occ = {}   # (b,t) -> v
    for v in range(n):
        if v not in berth:
            continue
        b, t0 = berth[v], tb[v]
        if S[b][t0]:
            nu += 1
        moor = PHI[b][t0]
        work = 0.0; t = t0; released = None
        while t <= L:
            ks = [k for k in range(K) if q_vkt.get((v, k, t), 0) == 1]
            if (b, t) in occ:
                problems.append(f"berth {b} period {t}: vessels {occ[(b,t)]} and {v}")
            occ[(b, t)] = v
            if t < t0 + moor:
                if ks: problems.append(f"vessel {v} works during mooring at t={t}")
            elif S[b][t]:
                if ks: problems.append(f"vessel {v} works in storm at t={t}")
            else:
                if len(ks) < inst.qmin[v] or len(ks) > inst.qmax[v]:
                    problems.append(f"vessel {v} gang {len(ks)} outside [{inst.qmin[v]},{inst.qmax[v]}] at t={t}")
                work += RHO * MU[b][t] * len(ks)
            if work >= inst.h[v] - 1e-9:
                released = t; break
            t += 1
        if released is None:
            problems.append(f"vessel {v} never completes"); released = L
        C[v] = released
    # crane exclusivity and non-crossing from q
    byt = {}
    for (v, k, t), val in q_vkt.items():
        if val == 1:
            byt.setdefault(t, []).append((k, berth.get(v, -1), v))
    for t, lst in byt.items():
        ks = [k for k, _, _ in lst]
        if len(ks) != len(set(ks)):
            problems.append(f"crane serves two vessels at t={t}")
        for k, b, v in lst:
            for k2, b2, v2 in lst:
                if k < k2 and b2 < b:
                    problems.append(f"non-crossing violated at t={t}: crane {k}@berth {b}, crane {k2}@berth {b2}")
    Z = float(sum(C[v] - inst.A[v] for v in C))
    return Z, nu, problems


def to_mip_variables(inst, wind, sol: Solution):
    """Export a simulator schedule as (x_vb, sigma_vbt, q_vkt) dicts, to be fixed in the MIP."""
    r = evaluate(inst, wind, sol, export=True)
    x = {(v, b): int(sol.berth[v] == b) for v in range(inst.n) for b in range(inst.n_berths)}
    sig = {(v, sol.berth[v], r.tb[v]): 1 for v in range(inst.n)}
    q = {(v, k, t): 1 for (v, t), ks in r.q_hist.items() for k in ks}
    return x, sig, q, r
