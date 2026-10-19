"""Learned search control (Section 4.3–4.4, Algorithm 1–2).

Environment  : one LNS run; action = (operator, severity, tau) in 6 x 3 x 3 = 54;
               state = 16 features (5 progress, 6 dynamics, 5 wind/structure);
               reward = Eq. (reward).
Variants     : "full"      54 actions, 16 features
               "op"        6 actions (severity 1, tau 0), 16 features
               "noweather" 54 actions, 11 features (wind/structure group removed)
               "masked"    as full, with severity-3 and tau=0.02 actions masked when
                           nu(x) > 0 or the storm share exceeds S_MASK (Section 5.x)
Training     : Double DQN, 2x256, Huber loss, replay 1e5, batch 64, Adam 5e-4 with cosine decay to 5e-5, gamma 0.99,
               target sync every 20 episodes, eps 1 -> 0.05 (x0.999/episode), grad-norm clip 5.
               Every CKPT_EVERY episodes the greedy policy is scored on a fixed validation set
               (instances disjoint from training and test); the checkpoint with the best
               validation mean is returned (reported in the paper as checkpoint selection).
"""
from __future__ import annotations
import math, random, time
import numpy as np
import torch, torch.nn as nn
from .simulator import evaluate
from .search import apply, greedy_initial, SearchRun
from .params import WLIM

SEV = [1, 2, 3]
TAU = [0.0, 0.005, 0.02]
N_OPS = 6
S_MASK = 0.15
CKPT_EVERY = 200
VALID_INST_SEEDS = list(range(4000, 4010))     # validation pool: disjoint from train (5000+) and test (1000+/2000+)


def n_actions(variant):
    return N_OPS if variant == "op" else N_OPS * len(SEV) * len(TAU)


def state_dim(variant):
    return 11 if variant == "noweather" else 16


def decode(a, variant):
    if variant == "op":
        return a, 1, 0.0
    return a % N_OPS, SEV[(a // N_OPS) % 3], TAU[a // (N_OPS * 3)]


class Env:
    def __init__(self, inst, wind, variant="full", budget=500, seed=0):
        self.inst, self.wind, self.variant, self.I = inst, wind, variant, budget
        self.rng = random.Random(seed)
        # wind/structure constants of the window: horizon mean wind and storm share (Section 4.3)
        T = inst.horizon
        self.w_bar = sum(wind.W[:T]) / T / WLIM
        self.s_hat = sum(wind.storm_share(b, 1, T) for b in range(inst.n_berths)) / inst.n_berths

    def reset(self):
        self.x0 = greedy_initial(self.inst, self.wind)
        r0 = evaluate(self.inst, self.wind, self.x0)
        self.z0 = max(1.0, r0.Z_omega)
        self.z0_plain = max(1.0, r0.Z)          # unpenalized greedy cost (validation normalizer)
        self.cur, self.cur_r = self.x0, r0
        self.best, self.best_r = self.x0, r0
        self.i = 0; self.stag = 0; self.last_acc = 0.0; self.last_imp = 0.0; self.last_tau = 0.0
        return self.state()

    def structure(self):
        n = self.inst.n; nb = self.inst.n_berths
        cnt = [0] * nb
        for b in self.cur.berth:
            cnt[b] += 1
        sh = [c / n for c in cnt]
        return [max(sh), float(np.std(sh)), sum(s * be for s, be in zip(sh, self.inst.beta))]

    def state(self):
        cr, br = self.cur_r, self.best_r
        f = [cr.Z_omega / self.z0, cr.nu / self.inst.n, br.Z_omega / self.z0,
             (cr.Z_omega - br.Z_omega) / max(1.0, br.Z_omega), self.i / self.I,
             min(self.stag, 50) / 50.0, self.last_acc, self.last_imp,
             float(cr.Z_omega <= br.Z_omega + 1e-9), self.last_tau / max(TAU), float(cr.nu > 0)]
        if self.variant != "noweather":
            f += [self.w_bar, self.s_hat] + self.structure()
        return np.array(f, dtype=np.float32)

    def mask(self):
        """Boolean mask over actions (True = allowed). Only the 'masked' variant restricts."""
        nA = n_actions(self.variant)
        if self.variant != "masked":
            return None
        if self.cur_r.nu > 0 or self.s_hat > S_MASK:
            m = np.ones(nA, dtype=bool)
            for a in range(nA):
                op, sev, tau = decode(a, self.variant)
                if sev == 3 or tau == 0.02:
                    m[a] = False
            return m
        return None

    def step(self, a):
        op, sev, tau = decode(a, self.variant)
        self.last_tau = tau
        cand = apply(op, self.cur, self.inst, self.wind, self.rng, severity=sev)
        r = evaluate(self.inst, self.wind, cand)
        d = r.Z_omega - self.cur_r.Z_omega
        accept = d <= 0
        if not accept and tau > 0:
            accept = self.rng.random() < math.exp(-d / (tau * max(1.0, self.best_r.Z_omega)))
        reward = 0.0; self.last_imp = 0.0
        if accept:
            reward += 0.1 * (self.cur_r.Z_omega - r.Z_omega) / max(1.0, self.cur_r.Z_omega)
            self.cur, self.cur_r = cand, r
        if r.Z_omega < self.best_r.Z_omega:
            self.best, self.best_r = cand, r
            reward += 5.0; self.stag = 0; self.last_imp = 1.0
        else:
            self.stag += 1
        self.last_acc = float(accept)
        self.i += 1
        done = self.i >= self.I
        if done:
            rho = (self.z0 - self.best_r.Z_omega) / self.z0
            reward += 20.0 * rho - 2.0 * self.best_r.nu
        return self.state(), reward, done


class QNet(nn.Module):
    def __init__(self, d_in, d_out, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, d_out))

    def forward(self, x):
        return self.net(x)


def greedy_action(net, s, mask=None):
    with torch.no_grad():
        q = net(torch.as_tensor(s).unsqueeze(0)).squeeze(0).numpy()
    if mask is not None:
        q = np.where(mask, q, -np.inf)
    return int(np.argmax(q))


# ------------------------------------------------------------------ rollout (deployment, Algorithm 2)
def rollout(policy, inst, wind, budget=500, seed=0, trace=False):
    """policy: dict(variant, state_dict) or ("random", seed) for DQN-R. Returns the result dict."""
    t0 = time.perf_counter()
    if isinstance(policy, tuple) and policy[0] == "random":
        variant, net = "full", None
        prng = random.Random(1_000_003 * (policy[1] + 1) + seed)
    else:
        variant = policy["variant"]; net = QNet(state_dim(variant), n_actions(variant))
        net.load_state_dict(policy["state_dict"]); net.eval(); prng = None
    env = Env(inst, wind, variant, budget, seed=seed)
    s = env.reset(); done = False
    tr = [(time.perf_counter() - t0, env.best_r.Z_omega)] if trace else None
    actions = []
    while not done:
        m = env.mask()
        if net is None:
            a = prng.randrange(n_actions(variant))
            if m is not None and not m[a]:
                a = prng.choice(np.flatnonzero(m).tolist())
        else:
            a = greedy_action(net, s, m)
        actions.append(a)
        s, _, done = env.step(a)
        if trace:
            tr.append((time.perf_counter() - t0, env.best_r.Z_omega))
    return dict(cost=env.best_r.Z_omega, Z=env.best_r.Z, viol=env.best_r.nu,
                time_s=time.perf_counter() - t0, trace=tr, actions=actions)


# ------------------------------------------------------------------ training (Algorithm 1)
def validation_score(net, variant, cfg, make_instance, make_wind, budget):
    tot = 0.0
    for k, isd in enumerate(VALID_INST_SEEDS):
        c = dict(cfg, mu_w=[5.0, 10.0, 15.0, 20.0][k % 4])
        inst = make_instance(isd, c); wind = make_wind(20_000_000 + isd, c)
        env = Env(inst, wind, variant, budget, seed=k)
        s = env.reset(); done = False
        while not done:
            s, _, done = env.step(greedy_action(net, s, env.mask()))
        tot += env.best_r.Z_omega / env.z0_plain      # penalized best relative to the UNPENALIZED greedy cost
    return tot / len(VALID_INST_SEEDS)


def train(seed, cfg, make_instance, make_wind, variant="full", episodes=8000, budget=500,
          lr=5e-4, gamma=0.99, batch=64, buf_cap=100_000, eps_min=0.05, eps_decay=0.999,
          target_every=20, ckpt_every=CKPT_EVERY, train_seed0=5000, n_train=50, broad=False,
          log=print, probe_csv=None, lr_final=5e-5, tag=None):
    """lr_final: the learning rate decays with a cosine schedule from lr to lr_final over the episodes
    (None = constant rate; the constant-rate runs are the 'full-nodecay' curves reported in Sec. 5);
    tag: label used in the probe CSV (defaults to variant)."""
    tag = tag or variant
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    nA, dS = n_actions(variant), state_dim(variant)
    net, tgt = QNet(dS, nA), QNet(dS, nA); tgt.load_state_dict(net.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    S_buf = np.zeros((buf_cap, dS), np.float32); S2_buf = np.zeros((buf_cap, dS), np.float32)
    A_buf = np.zeros(buf_cap, np.int64); R_buf = np.zeros(buf_cap, np.float32); D_buf = np.zeros(buf_cap, np.float32)
    M2_buf = np.ones((buf_cap, nA), bool)
    n_buf = 0; ptr = 0
    eps = 1.0; t0 = time.time(); best_val = float("inf"); best_state = None; hist = []
    mus = [5.0, 10.0, 15.0, 20.0]
    for ep in range(episodes):
        inst_seed = train_seed0 + (ep % n_train)
        c = dict(cfg, mu_w=mus[ep % 4])
        if broad:
            erng = random.Random(1000 * seed + ep)
            c["n_vessels"] = erng.choice([20, 30, 50, 75, 100])
            nb, nc, beta = erng.choice([(2, 4, [1.0, 0.4]), (3, 6, [1.0, 0.7, 0.4]),
                                        (4, 8, [1.0, 0.8, 0.6, 0.4]), (5, 10, [1.0, 0.85, 0.7, 0.55, 0.4])])
            c.update(n_berths=nb, n_cranes=nc, beta=beta)
        inst = make_instance(inst_seed, c); wind = make_wind(30_000_000 + 7919 * ep + 100 * seed, c)
        env = Env(inst, wind, variant, budget, seed=ep + 10 * seed)
        s = env.reset(); done = False
        while not done:
            m = env.mask()
            if random.random() < eps:
                a = random.randrange(nA) if m is None else int(random.choice(np.flatnonzero(m)))
            else:
                a = greedy_action(net, s, m)
            s2, r, done = env.step(a)
            S_buf[ptr] = s; A_buf[ptr] = a; R_buf[ptr] = r; S2_buf[ptr] = s2; D_buf[ptr] = float(done)
            m2 = env.mask(); M2_buf[ptr] = True if m2 is None else m2
            ptr = (ptr + 1) % buf_cap; n_buf = min(n_buf + 1, buf_cap); s = s2
            if n_buf >= batch:
                idx = np.random.randint(0, n_buf, batch)
                bs = torch.from_numpy(S_buf[idx]); ba = torch.from_numpy(A_buf[idx])
                br = torch.from_numpy(R_buf[idx]); bs2 = torch.from_numpy(S2_buf[idx]); bd = torch.from_numpy(D_buf[idx])
                q = net(bs).gather(1, ba.unsqueeze(1)).squeeze(1)
                with torch.no_grad():                           # Double DQN target
                    q_online = net(bs2).masked_fill(~torch.from_numpy(M2_buf[idx]), -1e9)
                    a2 = q_online.argmax(1, keepdim=True)
                    q2 = tgt(bs2).gather(1, a2).squeeze(1)
                    y = br + gamma * (1 - bd) * q2
                loss = nn.functional.smooth_l1_loss(q, y)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()
        eps = max(eps_min, eps * eps_decay)
        if lr_final is not None:
            frac = (ep + 1) / episodes
            for g in opt.param_groups:
                g["lr"] = lr_final + 0.5 * (lr - lr_final) * (1 + math.cos(math.pi * frac))
        if (ep + 1) % target_every == 0:
            tgt.load_state_dict(net.state_dict())
        if (ep + 1) % ckpt_every == 0 or ep + 1 == episodes:
            net.eval(); val = validation_score(net, variant, cfg, make_instance, make_wind, budget); net.train()
            hist.append((ep + 1, val, eps, time.time() - t0))
            if val < best_val:
                best_val = val; best_state = {k: v.clone() for k, v in net.state_dict().items()}; best_ep = ep + 1
            log(f"  [{variant} seed {seed}] ep {ep+1}/{episodes} val={val:.4f} best={best_val:.4f}@{best_ep} eps={eps:.2f} {time.time()-t0:.0f}s")
            if probe_csv:
                with open(probe_csv, "a") as f:
                    f.write(f"{tag},{seed},{ep+1},{val:.5f},{eps:.4f},{time.time()-t0:.0f}\n")
    return dict(variant=variant, seed=seed, state_dict=best_state, best_val=best_val, best_episode=best_ep,
                history=hist, lr_final=lr_final, tag=tag, episodes=episodes, budget=budget, train_time_s=time.time() - t0, broad=broad)
