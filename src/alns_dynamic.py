"""T15 - ALNS hoan chinh: warm-start + trong so toan tu thich nghi (+ lua chon toan tu theo cau hinh).

solve(pb, init_routes, iters, rng, cfg) chay ALNS tren MOT Problem (mot snapshot lap ke hoach).
cfg:
  destroy : danh sach ten toan tu go (xem destroy_ops.DESTROY)
  repair  : danh sach ten toan tu sua (xem repair_ops.REPAIR)
  adaptive: True -> chon toan tu bang roulette theo trong so thich nghi (Ropke & Pisinger 2006)
  warm    : True -> dung init_routes lam nghiem khoi tao; False -> xay lai tu dau bang regret-2
  archive : True -> luu tap nghiem khong troi (quang duong, rui ro, tre+khong phuc vu) de so sanh da muc tieu
Chap nhan theo simulated annealing; q (so khach go moi vong) lay ngau nhien trong [q_lo, q_hi] x so khach da xep.
"""
from __future__ import annotations

import math
import time

import numpy as np

from .destroy_ops import DESTROY
from .model import components, empty_solution, solution_from_routes
from .repair_ops import REPAIR

BASE_CFG = dict(destroy=["random", "worst", "related"], repair=["greedy", "regret2"],
                adaptive=False, warm=False, archive=False, q_lo=0.1, q_hi=0.3, q_max=30,
                sigma=(33.0, 9.0, 3.0), reaction=0.1, segment=50, t_start_frac=0.05, t_end_frac=0.002)
FULL_CFG = dict(BASE_CFG, destroy=["random", "worst", "related", "risk"],
                repair=["greedy", "regret2", "riskprio"], adaptive=True, warm=True)


def dominates(a, b):
    return all(x <= y + 1e-12 for x, y in zip(a, b)) and any(x < y - 1e-12 for x, y in zip(a, b))


def update_archive(arch, vec, w_u_over_w_l):
    if any(dominates(v, vec) or v == vec for v in arch):
        return
    arch[:] = [v for v in arch if not dominates(vec, v)]
    arch.append(vec)


def objective_vector(pb, sol):
    d, r, late, nlate, nun = components(pb, sol)
    return (round(d, 6), round(r, 6), round(late + nun * pb.w_u / pb.w_l, 6))


def _roulette(w, rng):
    tot = sum(w)
    x = rng.random() * tot
    acc = 0.0
    for i, v in enumerate(w):
        acc += v
        if x <= acc:
            return i
    return len(w) - 1


def initial_solution(pb, init_routes, cfg, rng):
    if cfg["warm"] and init_routes is not None:
        sol = solution_from_routes(pb, init_routes)
    else:
        sol = empty_solution(pb)
    REPAIR["regret2"](pb, sol, rng)
    return sol


def solve(pb, init_routes=None, iters=300, rng=None, cfg=None, time_limit=None, record_trace=False):
    cfg = cfg or FULL_CFG
    rng = rng or np.random.default_rng(0)
    t_begin = time.perf_counter()
    cur = initial_solution(pb, init_routes, cfg, rng)
    best = cur.copy()
    info = dict(iters=0, trace=[(0, best.cost)], used={}, archive=[])
    arch = info["archive"]
    if cfg["archive"]:
        update_archive(arch, objective_vector(pb, cur), pb.w_u / pb.w_l)
    if pb.m == 0 or iters <= 0:
        info["time"] = time.perf_counter() - t_begin
        return best, info
    dn, rn = cfg["destroy"], cfg["repair"]
    dfun, rfun = [DESTROY[n] for n in dn], [REPAIR[n] for n in rn]
    wd, wr = [1.0] * len(dn), [1.0] * len(rn)
    sd, sr = [0.0] * len(dn), [0.0] * len(rn)
    ud, ur = [0] * len(dn), [0] * len(rn)
    T = max(1e-9, cfg["t_start_frac"] * max(cur.cost, 1.0) / math.log(2.0))
    Tend = max(1e-12, cfg["t_end_frac"] * T)
    cool = (Tend / T) ** (1.0 / max(1, iters))
    s1, s2, s3 = cfg["sigma"]
    for it in range(1, iters + 1):
        if time_limit is not None and time.perf_counter() - t_begin > time_limit:
            break
        di = _roulette(wd, rng) if cfg["adaptive"] else int(rng.integers(len(dn)))
        ri = _roulette(wr, rng) if cfg["adaptive"] else int(rng.integers(len(rn)))
        cand = cur.copy()
        n_assigned = sum(len(r) for r in cand.routes)
        q = int(round(rng.uniform(cfg["q_lo"], cfg["q_hi"]) * max(n_assigned, 1)))
        q = max(1, min(q, cfg["q_max"]))
        dfun[di](pb, cand, q, rng)
        rfun[ri](pb, cand, rng)
        ud[di] += 1
        ur[ri] += 1
        delta = cand.cost - cur.cost
        sc = 0.0
        if cand.cost < best.cost - 1e-9:
            best = cand.copy()
            cur = cand
            sc = s1
        elif delta < -1e-9:
            cur = cand
            sc = s2
        elif delta > 1e-9 and rng.random() < math.exp(-delta / T):
            cur = cand
            sc = s3
        elif abs(delta) <= 1e-9:
            cur = cand
        if cfg["archive"]:
            update_archive(arch, objective_vector(pb, cand), pb.w_u / pb.w_l)
        sd[di] += sc
        sr[ri] += sc
        if cfg["adaptive"] and it % cfg["segment"] == 0:
            rho = cfg["reaction"]
            for i in range(len(dn)):
                if ud[i]:
                    wd[i] = (1 - rho) * wd[i] + rho * sd[i] / ud[i]
                wd[i] = max(wd[i], 0.05)
                sd[i], ud[i] = 0.0, 0
            for i in range(len(rn)):
                if ur[i]:
                    wr[i] = (1 - rho) * wr[i] + rho * sr[i] / ur[i]
                wr[i] = max(wr[i], 0.05)
                sr[i], ur[i] = 0.0, 0
        T *= cool
        info["iters"] = it
        if record_trace and it % 10 == 0:
            info["trace"].append((it, best.cost))
    if record_trace:
        info["trace"].append((info["iters"], best.cost))
    info["weights"] = dict(destroy=dict(zip(dn, wd)), repair=dict(zip(rn, wr)))
    info["time"] = time.perf_counter() - t_begin
    return best, info
