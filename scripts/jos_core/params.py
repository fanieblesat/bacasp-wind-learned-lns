"""Model constants. Every number here appears in Section 3 of the manuscript;
change them here and nowhere else (model_check.py and the validation script read them too)."""

W1    = 10.0   # m/s  productivity starts to degrade (Eq. mu)
WLIM  = 20.0   # m/s  shutdown threshold: storm S_bt = [beta_b W_t >= WLIM]
ALPHA = 0.5    # productivity lost at the shutdown threshold: mu -> 1 - ALPHA as w -> WLIM
PHI1  = 1      # mooring periods for W1 <= w < WLIM
PHI2  = 2      # mooring periods for w >= WLIM (only reachable by a relaxed/violating berthing)
RHO   = 30.0   # container moves per crane per period
def omega(n_vessels: int, n_periods: int) -> float:
    """Penalty per storm berthing in the search relaxation Z_Omega = Z + omega * nu.
    omega = |V| (|T|-1) exceeds max Z - min Z over feasible schedules (Prop. penalty), so the
    relaxation is exact: any feasible schedule is preferred to any infeasible one."""
    return float(n_vessels * (n_periods - 1))

# wind process (Ornstein–Uhlenbeck, one step per period); mu_w, sigma_w, theta_w come from cfg
WIND_LEN = 1000   # periods of wind generated per realization (independent of instance size)


def mu_of(w: float) -> float:
    """Crane productivity factor, continuous piecewise-linear (Eq. mu)."""
    if w < W1:
        return 1.0
    if w < WLIM:
        return round(1.0 - ALPHA * (w - W1) / (WLIM - W1), 3)   # 3 decimals: exact integer arithmetic in the CP model
    return 0.0


def phi_of(w: float) -> int:
    """Mooring delay in periods (Eq. phi)."""
    if w < W1:
        return 0
    if w < WLIM:
        return PHI1
    return PHI2
