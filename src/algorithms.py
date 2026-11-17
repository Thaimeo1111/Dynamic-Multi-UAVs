"""Dang ky thuat toan: ten -> bo dieu phoi. Dung chung cho run_small.py, run_batch.py va cac thi nghiem.

Ngan sach vong lap ALNS moi lan giai (iters) mac dinh theo co instance; chinh trong config.yaml hoac override.
"""
from __future__ import annotations

from .alns_dynamic import BASE_CFG, FULL_CFG
from .alns_static import StaticPlanner, make_static
from .baseline_greedy import GreedyPlanner
from .baseline_resolve import make_resolve
from .rolling_horizon import ALNSPlanner

MAIN = ["greedy", "alns_static", "resolve", "alns_dynamic"]
ABLATION = ["dyn_no_riskprio", "dyn_no_riskrem", "dyn_no_warm", "dyn_no_rolling", "resolve_periodic"]


def default_iters(n):
    return 150 if n <= 30 else (250 if n <= 60 else (400 if n <= 100 else 600))  # khop config.yaml alns.iters


def make_planner(name, n, seed, o=None, keep_trace=False):
    o = o or {}
    iters = int(o.get("iters", default_iters(n)))
    min_gap = float(o.get("min_gap", 40.0))
    dt = o.get("dt", 100.0)
    static_mult = float(o.get("static_mult", 3.0))  # khop config.yaml alns.static_mult
    if name == "greedy":
        return GreedyPlanner(min_gap=min_gap, seed=seed)
    if name == "alns_static":
        return make_static(int(iters * static_mult), seed, keep_trace, archive=bool(o.get("archive", False)))
    if name == "resolve":
        return make_resolve(iters, seed, min_gap, keep_trace)
    if name == "resolve_periodic":      # doi chung: re-solve co ban nhung DUNG LICH tai lap ke hoach cua ALNS dong
        return ALNSPlanner(name, dict(BASE_CFG), iters, "events", min_gap=min_gap, dt=dt, seed=seed, keep_trace=keep_trace)
    cfg = dict(FULL_CFG)
    if "q_hi" in o:
        cfg["q_hi"] = o["q_hi"]
    mode, include_all = "events", False
    if name == "alns_dynamic":
        pass
    elif name == "dyn_no_riskprio":
        cfg["repair"] = [r for r in cfg["repair"] if r != "riskprio"]
    elif name == "dyn_no_riskrem":
        cfg["destroy"] = [d for d in cfg["destroy"] if d != "risk"]
    elif name == "dyn_no_warm":
        cfg["warm"] = False
    elif name == "dyn_no_rolling":
        return StaticPlanner(name, dict(cfg, warm=False), int(iters * static_mult), min_gap, seed, keep_trace)
    else:
        raise ValueError(f"Thuat toan khong biet: {name}")
    return ALNSPlanner(name, cfg, iters, mode, include_all=include_all, min_gap=min_gap, dt=dt, seed=seed,
                       keep_trace=keep_trace)
