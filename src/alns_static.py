"""T07 - Baseline ALNS TINH: toi uu ke hoach ban dau MOT LAN (luc t = 0) roi bay theo ke hoach.

Khong biet su kien dong:
  * chi toi uu khach da biet luc t = 0 bang ALNS co ban (go random/worst/related, sua greedy/regret-2,
    khong thich nghi, khong warm-start) voi risk map theo belief tai t = 0;
  * khong tai toi uu khi co hotspot moi/re huong, doi deadline, pin tut (chi co phan xa an toan chung);
  * khach xuat hien sau duoc CHEN THAM LAM vao tuyen dang co (khong toi uu lai) - neu khong, moi khach den sau
    deu bi bo, va moc so sanh se qua yeu.
Khac voi baseline_greedy o cho: ke hoach ban dau do ALNS toi uu thay vi chen tham lam.
"""
from __future__ import annotations

import numpy as np

from .alns_dynamic import BASE_CFG, solve
from .model import solution_from_routes
from .repair_ops import repair_greedy
from .rolling_horizon import apply_solution, build_problem, trigger_times, warm_routes


class StaticPlanner:
    def __init__(self, name, cfg, iters, min_gap=40.0, seed=0, keep_trace=False):
        self.name, self.cfg, self.iters = name, cfg, iters
        self.min_gap, self.seed, self.keep_trace = min_gap, seed, keep_trace
        self.trace = None
        self.archive = None

    def triggers(self, scn, Tend):
        return trigger_times(scn, "targets", Tend, self.min_gap)

    def plan(self, sim, t, idx):
        pb = build_problem(sim, t)
        rng = np.random.default_rng(self.seed * 7919 + idx)
        if idx == 0:
            sol, info = solve(pb, None, self.iters, rng, self.cfg, record_trace=self.keep_trace)
            if self.keep_trace:
                self.trace = info["trace"]
            if self.cfg.get("archive"):
                self.archive = info["archive"]
        else:
            sol = solution_from_routes(pb, warm_routes(sim, pb))
            repair_greedy(pb, sol, rng)
        apply_solution(sim, pb, sol)
        return {}


def make_static(iters, seed=0, keep_trace=False, archive=False):
    return StaticPlanner("alns_static", dict(BASE_CFG, archive=archive), iters, seed=seed, keep_trace=keep_trace)
