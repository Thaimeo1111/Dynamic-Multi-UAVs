"""T06 - Baseline greedy: chen re ke tiep (cheapest insertion) tung khach moi vao cac tuyen dang co.

Khong pha/sua tuyen, khong toi uu lai. Khi co khach moi (hoac khach cu chua gan duoc), chen vao vi tri
co delta chi phi (quang duong + rui ro + tre) nho nhat; cac chang khong con kha thi (do risk/pin doi)
bi cat bot o cuoi tuyen. Day la moc so sanh "khong toi uu".
"""
from __future__ import annotations

import numpy as np

from .model import solution_from_routes
from .repair_ops import repair_greedy
from .rolling_horizon import apply_solution, build_problem, trigger_times, warm_routes


class GreedyPlanner:
    def __init__(self, name="greedy", min_gap=40.0, seed=0):
        self.name, self.min_gap, self.seed = name, min_gap, seed
        self.trace = None
        self.archive = None

    def triggers(self, scn, Tend):
        return trigger_times(scn, "targets", Tend, self.min_gap)

    def plan(self, sim, t, idx):
        pb = build_problem(sim, t)
        sol = solution_from_routes(pb, warm_routes(sim, pb))
        repair_greedy(pb, sol, np.random.default_rng(self.seed))
        apply_solution(sim, pb, sol)
        return {}
