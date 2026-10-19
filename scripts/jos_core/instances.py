"""Instance and wind generators. Both are deterministic in their seed and in cfg."""
from __future__ import annotations
import numpy as np
from .params import W1, WLIM, WIND_LEN, mu_of, phi_of

# ---------------------------------------------------------------- the Hamburg record
# data/hamburg_bap_vessels_with_weather.csv: 50 port calls at CTA/CTB/CTT, 22–26 April 2025.
# OBSERVED columns used here: arrival_time (-> empirical inter-arrival times), vessel_type.
# GENERATED columns (per the author) used here: H_v0, the nominal handling time in hours.
# Everything else in the file (L_v, D_v, wind, tide, sea state) is not used by this model.
import csv, os
from datetime import datetime
DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data",
                    "hamburg_bap_vessels_with_weather.csv")
# per class: nominal gang used to convert H_v0 (hours) into moves at RHO per crane-period, q_min, q_max
CLASSES = {"feeder": (1, 1, 2), "seagoing vessel": (2, 2, 4)}


def _load_record(path=DATA):
    rows = list(csv.DictReader(open(path)))
    ts = [datetime.strptime(r["arrival_time"], "%Y-%m-%dT%H:%MZ") for r in rows]
    order = sorted(range(len(rows)), key=lambda i: ts[i])
    ts = [ts[i] for i in order]; rows = [rows[i] for i in order]
    gaps = [(ts[i] - ts[i - 1]).total_seconds() / 3600.0 for i in range(1, len(ts))]
    calls = [(r["vessel_type"], float(r["H_v0"])) for r in rows]
    return calls, gaps


_RECORD = None
def record():
    global _RECORD
    if _RECORD is None:
        _RECORD = _load_record()
    return _RECORD


class Instance:
    __slots__ = ("n", "A", "h", "qmin", "qmax", "vtype", "n_berths", "n_cranes",
                 "beta", "horizon", "seed")

    def __init__(self, n, A, h, qmin, qmax, vtype, cfg, seed):
        self.n = n
        self.A, self.h, self.qmin, self.qmax, self.vtype = A, h, qmin, qmax, vtype
        self.n_berths = int(cfg["n_berths"]); self.n_cranes = int(cfg["n_cranes"])
        self.beta = list(cfg["beta"]); self.horizon = int(cfg["horizon"]); self.seed = seed
        assert len(self.beta) == self.n_berths
        # gang lower bounds must be coverable with one vessel per berth (Section 4.1 rule never binds)
        assert max(self.qmax) <= self.n_cranes


def make_instance(seed: int, cfg: dict) -> Instance:
    """Bootstrap an instance of |V| calls from the Hamburg record (deterministic in seed):
    inter-arrival times resampled from the observed gaps (hours -> periods, rounded), class and
    nominal handling time resampled jointly from the observed calls; workload in moves
    h_v = RHO * q_nom(class) * H_v0; gang bounds by class."""
    from .params import RHO
    rng = np.random.RandomState(seed)
    calls, gaps = record()
    n = int(cfg["n_vessels"])
    idx = rng.randint(0, len(calls), size=n)
    g = rng.choice(gaps, size=n - 1, replace=True) if n > 1 else []
    t = 1.0; A = [1]
    for x in g:
        t += x; A.append(int(round(t)))
    h, qmin, qmax, vt = [], [], [], []
    for i in idx:
        cls, H0 = calls[i]; qn, lo, hi = CLASSES[cls]
        h.append(float(round(RHO * qn * H0))); qmin.append(lo); qmax.append(hi); vt.append(cls)
    return Instance(n, A, h, qmin, qmax, vt, cfg, seed)


class Wind:
    """One realization of the OU wind process plus the berth-specific derived series."""
    __slots__ = ("W", "beta", "S", "mu", "phi", "seed", "mu_w", "sigma_w", "theta_w")

    def __init__(self, W, beta, seed, mu_w, sigma_w, theta_w):
        self.W = W; self.beta = list(beta); self.seed = seed
        self.mu_w, self.sigma_w, self.theta_w = mu_w, sigma_w, theta_w
        nb = len(beta); L = len(W)
        # index t = 1..L (position 0 unused) to match the model's T = {1,...}
        self.S = [[0] * (L + 1) for _ in range(nb)]
        self.mu = [[0.0] * (L + 1) for _ in range(nb)]
        self.phi = [[0] * (L + 1) for _ in range(nb)]
        for b in range(nb):
            Sb, mub, phib, bb = self.S[b], self.mu[b], self.phi[b], beta[b]
            for t in range(1, L + 1):
                w = bb * W[t - 1]
                Sb[t] = 1 if w >= WLIM else 0
                mub[t] = mu_of(w); phib[t] = phi_of(w)

    def __len__(self):
        return len(self.W)

    def storm_share(self, b: int, t0: int, t1: int) -> float:
        """Fraction of storm periods at berth b over periods t0..t1 (inclusive, clipped)."""
        t0 = max(1, t0); t1 = min(len(self.W), t1)
        if t1 < t0:
            return 0.0
        return sum(self.S[b][t0:t1 + 1]) / (t1 - t0 + 1)


def make_wind(seed: int, cfg: dict) -> Wind:
    mu_w, sigma_w, theta_w = float(cfg["mu_w"]), float(cfg["sigma_w"]), float(cfg["theta_w"])
    rng = np.random.RandomState(seed)
    W = np.empty(WIND_LEN)
    W[0] = max(0.0, mu_w + rng.normal(0.0, 2.0))
    noise = rng.normal(size=WIND_LEN)
    for t in range(1, WIND_LEN):
        W[t] = max(0.0, W[t - 1] + theta_w * (mu_w - W[t - 1]) + sigma_w * noise[t])
    return Wind(W.tolist(), cfg["beta"], seed, mu_w, sigma_w, theta_w)
