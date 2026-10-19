"""Initial solution (Greedy), the six operators of Section 4.2, and the classical baselines.

All search methods share: the simulator, the operator portfolio, the initial solution x0,
the budget I (candidate evaluations; a severity-s action costs s operator applications but
ONE candidate evaluation counts for the budget — the same accounting for every method),
and the objective Z_Omega = Z + OMEGA * nu with nu reported separately.
"""
from __future__ import annotations
import math, random, time
from .simulator import Solution, evaluate

R_FRAC = 0.2       # kappa_0: fraction of vessels removed
THETA_S = 0.3      # kappa_1: storm-share threshold
H_WIN = 12         # kappa_1: forecast window (periods) after the planned berthing

# ------------------------------------------------------------------ initial solution
def greedy_initial(inst, wind) -> Solution:
    """Section 4.1: vessels in arrival order; each to the berth where it starts service earliest
    (berthing + mooring), ties toward the more sheltered berth; cranes by the simulator's rule."""
    sol = Solution.empty(inst)
    placed = []
    order = sorted(range(inst.n), key=lambda v: (inst.A[v], v))
    berths_by_shelter = sorted(range(inst.n_berths), key=lambda b: inst.beta[b])
    for v in order:
        best = None
        for b in berths_by_shelter:                 # iterate sheltered first -> ties keep sheltered
            sol.berth[v] = b
            r = evaluate(inst, wind, sol, subset=placed + [v])
            tb = r.tb[v]
            start = tb + wind.phi[b][tb] if tb < len(wind) else 10 ** 9
            storm = wind.S[b][tb] if tb < len(wind) else 1
            key = (storm, start)                        # a non-storm berthing is always preferred
            if best is None or key < best[0]:
                best = (key, b)
        sol.berth[v] = best[1]
        placed.append(v)
    sol.touch()
    return sol


# ------------------------------------------------------------------ operators
def _reset(sol, inst, v):
    sol.earliest[v] = inst.A[v]; sol.weight[v] = 1.0; sol.prio[v] = float(inst.A[v])


def op0_random_reassign(sol, inst, wind, rng):
    s = sol.copy(); s.touch()
    k = max(1, int(round(R_FRAC * inst.n)))
    for v in rng.sample(range(inst.n), k):
        _reset(s, inst, v)
        s.berth[v] = rng.randrange(inst.n_berths)
    return s


def op1_storm_aware(sol, inst, wind, rng):
    r = evaluate(inst, wind, sol)
    s = sol.copy(); s.touch(); moved = False
    for v in range(inst.n):
        tb = r.tb[v]; b = s.berth[v]
        if wind.storm_share(b, tb, tb + H_WIN - 1) > THETA_S:
            shares = [(wind.storm_share(bb, tb, tb + H_WIN - 1), inst.beta[bb], bb) for bb in range(inst.n_berths)]
            nb_ = min(shares)[2]
            if nb_ != b:
                s.berth[v] = nb_; moved = True
    if not moved:                                   # nothing exposed: behave as a single random move
        v = rng.randrange(inst.n); s.berth[v] = rng.randrange(inst.n_berths)
    return s


def op2_crane_rebalance(sol, inst, wind, rng):
    s = sol.copy(); s.touch()
    heavy = sorted(range(inst.n), key=lambda v: -inst.h[v])[:max(1, inst.n // 4)]
    v = rng.choice(heavy)
    s.weight[v] = min(8.0, s.weight[v] * 2.0)
    return s


def op3_swap(sol, inst, wind, rng):
    s = sol.copy(); s.touch()
    if inst.n < 2:
        return s
    i, j = rng.sample(range(inst.n), 2)
    s.berth[i], s.berth[j] = s.berth[j], s.berth[i]
    return s


def op4_delay_storm(sol, inst, wind, rng):
    r = evaluate(inst, wind, sol)
    s = sol.copy(); s.touch(); changed = False
    for v in range(inst.n):
        b = s.berth[v]; tb = r.tb[v]
        if tb < len(wind) and wind.S[b][tb]:
            t = tb
            while t < len(wind) and wind.S[b][t]:
                t += 1
            s.earliest[v] = t; changed = True
    if not changed:                                 # no violation to remove: relax one delay instead
        v = rng.randrange(inst.n)
        if s.earliest[v] > inst.A[v]:
            s.earliest[v] = inst.A[v]
        else:
            s.prio[v] = s.prio[v] + rng.uniform(-3, 3)   # local reordering within its berth
    return s


def op5_consolidate(sol, inst, wind, rng):
    """Move every vessel off the least-occupied berth; each goes to the least-loaded of the
    remaining berths (load = number of vessels), ties toward the more sheltered berth."""
    s = sol.copy(); s.touch()
    counts = [0] * inst.n_berths
    for v in range(inst.n):
        counts[s.berth[v]] += 1
    least = min(range(inst.n_berths), key=lambda b: (counts[b], b))
    others = [b for b in range(inst.n_berths) if b != least]
    if not others:
        return s
    movers = sorted((v for v in range(inst.n) if s.berth[v] == least), key=lambda v: inst.A[v])
    for v in movers:
        b = min(others, key=lambda bb: (counts[bb], inst.beta[bb], bb))
        s.berth[v] = b; counts[least] -= 1; counts[b] += 1
    return s


OPERATORS = [op0_random_reassign, op1_storm_aware, op2_crane_rebalance,
             op3_swap, op4_delay_storm, op5_consolidate]
OP_NAMES = ["random-reassign", "storm-aware", "crane-rebalance", "swap", "delay-storm", "consolidate"]


def apply(op_i, sol, inst, wind, rng, severity=1):
    cand = sol
    for _ in range(severity):
        cand = OPERATORS[op_i](cand, inst, wind, rng)
    return cand


# ------------------------------------------------------------------ classical baselines
def _sa_T0(z0):
    """Linear-cooling schedule (Santini et al. 2018 style): T_i = T0 (1 - i/I) with T0 such that
    a candidate 1% worse than x0 is accepted with probability 1/2."""
    return 0.01 * z0 / math.log(2)


class SearchRun:
    """Common bookkeeping: best-so-far, trace of (elapsed_s, best Z_omega) per iteration."""
    def __init__(self, inst, wind, budget, trace):
        self.inst, self.wind, self.budget = inst, wind, budget
        self.t0 = time.perf_counter(); self.trace = [] if trace else None
        self.x0 = greedy_initial(inst, wind); r0 = evaluate(inst, wind, self.x0)
        self.cur, self.cur_r = self.x0, r0
        self.best, self.best_r = self.x0, r0
        if self.trace is not None:
            self.trace.append((time.perf_counter() - self.t0, r0.Z_omega))

    def consider(self, cand):
        r = evaluate(self.inst, self.wind, cand)
        if r.Z_omega < self.best_r.Z_omega:
            self.best, self.best_r = cand, r
        if self.trace is not None:
            self.trace.append((time.perf_counter() - self.t0, self.best_r.Z_omega))
        return r

    def result(self):
        return dict(cost=self.best_r.Z_omega, Z=self.best_r.Z, viol=self.best_r.nu,
                    time_s=time.perf_counter() - self.t0, trace=self.trace)


def run_greedy(inst, wind, budget=500, seed=0, trace=False):
    s = SearchRun(inst, wind, budget, trace)
    return s.result()


def run_std_lns(inst, wind, budget=500, seed=0, trace=False):
    rng = random.Random(seed); s = SearchRun(inst, wind, budget, trace)
    T0 = _sa_T0(s.cur_r.Z_omega)
    for i in range(budget):
        T = T0 * (1.0 - i / budget)
        cand = apply(rng.randrange(6), s.cur, inst, wind, rng)
        r = s.consider(cand)
        d = r.Z_omega - s.cur_r.Z_omega
        if d <= 0 or (T > 0 and rng.random() < math.exp(-d / T)):
            s.cur, s.cur_r = cand, r
    return s.result()


def _run_alns(inst, wind, budget, seed, trace, annealing):
    """Ropke–Pisinger segment-based weights: segment length 50, reaction factor 0.5,
    scores 5 (new best) / 3 (improved current) / 1 (accepted worse)."""
    rng = random.Random(seed); s = SearchRun(inst, wind, budget, trace)
    T0 = _sa_T0(s.cur_r.Z_omega) if annealing else 0.0
    w = [1.0] * 6; score = [0.0] * 6; used = [0] * 6
    for i in range(budget):
        T = T0 * (1.0 - i / budget)
        op = rng.choices(range(6), weights=w)[0]
        cand = apply(op, s.cur, inst, wind, rng)
        before_best = s.best_r.Z_omega
        r = s.consider(cand)
        used[op] += 1
        d = r.Z_omega - s.cur_r.Z_omega
        if r.Z_omega < before_best:
            score[op] += 5; s.cur, s.cur_r = cand, r
        elif d < 0:
            score[op] += 3; s.cur, s.cur_r = cand, r
        elif d == 0:
            s.cur, s.cur_r = cand, r
        elif T > 0 and rng.random() < math.exp(-d / T):
            score[op] += 1; s.cur, s.cur_r = cand, r
        if (i + 1) % 50 == 0:
            for k in range(6):
                if used[k]:
                    w[k] = 0.5 * w[k] + 0.5 * score[k] / used[k]
                w[k] = max(w[k], 0.05)
            score = [0.0] * 6; used = [0] * 6
    return s.result()


def run_alns(inst, wind, budget=500, seed=0, trace=False):
    return _run_alns(inst, wind, budget, seed, trace, annealing=False)


def run_alns_sa(inst, wind, budget=500, seed=0, trace=False):
    return _run_alns(inst, wind, budget, seed, trace, annealing=True)
