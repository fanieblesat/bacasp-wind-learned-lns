"""
interface.py — the five hooks the driver scripts use, PRE-WIRED to jos_core.
Nothing in this file needs editing; it is kept as the single seam between the
experiment drivers (1_..., 2_..., 3_..., 4_...) and the model/algorithm package.

method names: "Greedy","Std-LNS","ALNS","ALNS-SA","DQN-LNS","DQN-R","DQN-op","DQN-noweather","DQN-masked"
policy      : dict returned by train_policy (picklable), or ("random", seed) for DQN-R
"""
from __future__ import annotations
import time
from jos_core.instances import make_instance as _mk_inst, make_wind as _mk_wind
from jos_core.simulator import evaluate_fixed
from jos_core import search, controller

VARIANT_OF = {"DQN-LNS": "full", "DQN-LNS-nodecay": "full", "DQN-op": "op", "DQN-noweather": "noweather", "DQN-masked": "masked"}


def make_instance(seed: int, cfg: dict):
    return _mk_inst(seed, cfg)


def make_wind(seed: int, cfg: dict):
    return _mk_wind(seed, cfg)


def train_policy(seed: int, cfg: dict, variant: str = "full", episodes: int = 8000, **kw):
    return controller.train(seed, cfg, _mk_inst, _mk_wind, variant=variant, episodes=episodes, **kw)


def evaluate(method: str, policy, instance, wind, budget: int = 500, trace: bool = False, seed: int = 0):
    if method == "Greedy":
        return search.run_greedy(instance, wind, budget, seed, trace)
    if method == "Std-LNS":
        return search.run_std_lns(instance, wind, budget, seed, trace)
    if method == "ALNS":
        return search.run_alns(instance, wind, budget, seed, trace)
    if method == "ALNS-SA":
        return search.run_alns_sa(instance, wind, budget, seed, trace)
    if method == "DQN-R":
        return controller.rollout(policy if isinstance(policy, tuple) else ("random", 0), instance, wind, budget, seed, trace)
    if method in VARIANT_OF:
        assert policy["variant"] == VARIANT_OF[method], f"{method} needs a {VARIANT_OF[method]} policy"
        return controller.rollout(policy, instance, wind, budget, seed, trace)
    raise ValueError(method)


def schedule_from_solution(instance, wind, x_vb, sigma_vbt, q_vkt):
    """Agreement check: follow the solver's variables with the simulator's rules -> (Z, nu)."""
    Z, nu, problems = evaluate_fixed(instance, wind, x_vb, sigma_vbt, q_vkt)
    if problems:
        raise RuntimeError("solver schedule violates the simulator's reading of the model: " + "; ".join(problems[:5]))
    return Z, nu


def timed_evaluate(*args, **kwargs):
    t0 = time.perf_counter()
    r = evaluate(*args, **kwargs)
    r.setdefault("time_s", time.perf_counter() - t0)
    return r
