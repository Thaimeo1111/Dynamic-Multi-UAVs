"""T12 - Khung rolling-horizon: mo phong thuc thi + kich hoat tai lap ke hoach + dong bang phan da thuc thi.

Cac thanh phan
--------------
Sim              : "the gioi that". Chay thoi gian lien tuc, UAV bay theo tuyen da duoc giao tren risk map THAT,
                   ap dung su kien (khach moi, doi deadline, pin tut) va ghi lai chi so THUC TE.
                   Phan da bay / dang phuc vu la "da dong bang": khong bao gio bi lap ke hoach lai.
build_problem    : chup trang thai luc t -> Problem (chi gom khach chua phuc vu, UAV o vi tri hien tai,
                   risk map theo BELIEF cua bo dieu phoi tai t).
trigger_times    : thoi diem tai lap ke hoach: theo su kien (co khoang cach toi thieu) va/hoac chu ky dt.
run_episode      : vong lap chinh:  advance_to(t) -> plan(t) -> set_plan -> ... -> ket thuc ngay bay.
Phan phan xa an toan (chung cho MOI thuat toan): truoc khi bay toi mot khach, neu pin khong du de toi do
va ve depot thi bo qua khach do; neu khong con du pin de ve depot thi UAV bi coi la MAT (lost).
"""
from __future__ import annotations

import math
import time

import numpy as np

from .alns_dynamic import solve
from .model import make_problem
from .risk_map import Hotspot, RiskField

CRITICAL = {"risk_change", "battery_drop", "deadline_change"}


# =============================================================== the gioi that
class Sim:
    def __init__(self, inst, scn, P):
        self.inst, self.scn, self.P = inst, scn, P
        self.n = inst.n
        self.depot = inst.xy[0].copy()
        hs = [Hotspot.from_dict(d) for d in scn["hotspots"]]
        static = np.array(scn["static_risk"]) if scn.get("static_risk") is not None else None
        self.truth = RiskField(hs, tuple(scn["bbox"]), P["cell"], static=static)
        self.ready = inst.ready.copy()
        self.due = inst.due.copy()
        self.serv, self.dem = inst.serv, inst.dem
        self.release = np.array(scn["release"], float)
        self.due_hist = {}
        self.known = {c for c in range(1, self.n + 1) if self.release[c] <= 0.0}
        self.served = {}                       # khach -> (thoi diem bat dau, uav)
        K = int(scn["uav"]["K"])
        bat0 = scn["uav"]["battery"] / P["eps"]
        self.u = [dict(x=float(self.depot[0]), y=float(self.depot[1]), t=0.0, bat=bat0, load=0.0, seq=[],
                       commit=None, busy=0.0, lost=False, flown=0.0) for _ in range(K)]
        self.events = scn["events"]
        self.ei = 0
        self.t = 0.0
        self.Tend = float(inst.horizon)
        self.dist = self.risk = self.ban_dist = 0.0
        self.n_lost = 0

    # ---------------------------------------------------------- su kien
    def due_at(self, c, t):
        d = self.inst.due[c]
        for te, nd in self.due_hist.get(c, []):
            if te <= t + 1e-9:
                d = nd
        return d

    def _apply(self, e):
        if e["type"] == "new_target":
            self.known.add(e["cust"])
        elif e["type"] == "deadline_change":
            c = e["cust"]
            self.due[c] = e["new_due"]
            self.due_hist.setdefault(c, []).append((e["t"], e["new_due"]))
        elif e["type"] == "battery_drop":
            u = self.u[e["uav"]]
            if not u["lost"]:
                u["bat"] *= (1.0 - e["frac"])
        # risk_change: that su da nam trong truth; bo dieu phoi biet qua belief(t)

    def advance_to(self, T):
        while self.ei < len(self.events) and self.events[self.ei]["t"] <= T + 1e-9:
            e = self.events[self.ei]
            self._move_all(max(e["t"], self.t))
            self._apply(e)
            self.ei += 1
        self._move_all(T)

    # ---------------------------------------------------------- chuyen dong
    def _move_all(self, T):
        for k, u in enumerate(self.u):
            self._move(k, u, T)
        self.t = max(self.t, T)

    def _expose(self, x0, y0, t0, x1, y1, t1):
        L = math.hypot(x1 - x0, y1 - y0)
        if L < 1e-9:
            return
        n = max(2, int(math.ceil(L / 2.0)))
        f = (np.arange(n) + 0.5) / n
        pts = np.column_stack([x0 + (x1 - x0) * f, y0 + (y1 - y0) * f])
        R = self.truth.risk_at(pts, t0 + (t1 - t0) * f)
        self.risk += float(R.mean()) * L
        away = np.hypot(pts[:, 0] - self.depot[0], pts[:, 1] - self.depot[1]) > self.P.get("depot_safe", 0.0)
        self.ban_dist += L * float(np.mean((R > self.P["theta_ban"]) & away))

    def _fly(self, u, tx, ty, T):
        x0, y0, t0 = u["x"], u["y"], u["t"]
        d = math.hypot(tx - x0, ty - y0)
        if d < 1e-12:
            return True
        tn = d / self.P["speed"]
        full = t0 + tn <= T + 1e-12
        frac = 1.0 if full else max(0.0, (T - t0) / tn)
        dd = d * frac
        exhausted = False
        if dd > u["bat"] + 1e-9:
            frac, dd, exhausted = u["bat"] / d, u["bat"], True
        x1, y1, t1 = x0 + (tx - x0) * frac, y0 + (ty - y0) * frac, t0 + tn * frac
        self._expose(x0, y0, t0, x1, y1, t1)
        u["x"], u["y"], u["t"] = x1, y1, t1
        u["bat"] -= dd
        u["flown"] += dd
        self.dist += dd
        if exhausted:
            u["lost"] = True
            u["seq"] = []
            self.n_lost += 1
            return False
        if not full:
            u["t"] = T
            return False
        return True

    def _home_dist(self, x, y):
        return math.hypot(x - self.depot[0], y - self.depot[1])

    def _move(self, k, u, T):
        xy = self.inst.xy
        while u["t"] < T - 1e-12 or (u["commit"] is not None and u["busy"] <= T + 1e-12):
            if u["lost"]:
                u["t"] = T
                return
            if u["commit"] is not None:
                if u["busy"] <= T + 1e-12:
                    u["t"] = max(u["t"], u["busy"])
                    u["commit"] = None
                    continue
                u["t"] = T
                return
            if u["bat"] + 1e-9 < self._home_dist(u["x"], u["y"]):      # khong con du pin ve depot
                u["lost"] = True
                u["seq"] = []
                self.n_lost += 1
                u["t"] = T
                return
            tgt = None
            while u["seq"]:
                c = u["seq"][0]
                if c in self.served:
                    u["seq"].pop(0)
                    continue
                need = math.hypot(xy[c, 0] - u["x"], xy[c, 1] - u["y"]) + self._home_dist(xy[c, 0], xy[c, 1])
                if need <= u["bat"] + 1e-9:
                    tgt = c
                    break
                u["seq"].pop(0)                                          # phan xa: khong du pin -> bo qua
            if tgt is None:
                if self._home_dist(u["x"], u["y"]) < 1e-9:
                    u["t"] = T
                    return
                if not self._fly(u, self.depot[0], self.depot[1], T):
                    return
                continue
            if not self._fly(u, xy[tgt, 0], xy[tgt, 1], T):
                return
            u["seq"].pop(0)
            if tgt in self.served:
                continue
            st = max(u["t"], self.ready[tgt], self.release[tgt])
            self.served[tgt] = (st, k)
            u["commit"], u["busy"] = tgt, st + self.serv[tgt]
            u["load"] += self.dem[tgt]

    # ---------------------------------------------------------- ke hoach
    def set_plan(self, routes):
        for k, seq in enumerate(routes):
            self.u[k]["seq"] = [c for c in seq if c not in self.served]

    def metrics(self):
        P = self.P
        late = nlate = 0
        for c, (s, _) in self.served.items():
            lt = s - self.due_at(c, s)
            if lt > 1e-9:
                late += lt
                nlate += 1
        unserved = self.n - len(self.served)
        J = P["w_d"] * self.dist + P["w_r"] * self.risk + P["w_l"] * late + P["w_u"] * unserved
        return dict(J=J, dist=self.dist, risk=self.risk, late=late, n_late=nlate, unserved=unserved,
                    served=len(self.served), ban_dist=self.ban_dist, n_lost=self.n_lost)


# =============================================================== lap ke hoach tu trang thai
def build_problem(sim, t, include_all=False, belief=True):
    inst = sim.inst
    pool = [c for c in range(1, sim.n + 1) if c not in sim.served and (include_all or c in sim.known)]
    idx = np.array(pool, int)
    cust = dict(gid=pool, xy=inst.xy[idx] if pool else np.zeros((0, 2)),
                ready=np.maximum(sim.ready[idx], sim.release[idx]) if pool else [],
                due=sim.due[idx] if pool else [], serv=sim.serv[idx] if pool else [],
                dem=sim.dem[idx] if pool else [])
    uavs = []
    for u in sim.u:
        if u["lost"]:
            uavs.append(dict(x=u["x"], y=u["y"], t_free=t, bat=0.0, cap=0.0, alive=False))
        else:
            uavs.append(dict(x=u["x"], y=u["y"], t_free=max(t, u["busy"]) if u["commit"] is not None else t,
                             bat=u["bat"], cap=inst.Q - u["load"], alive=True))
    field = sim.truth.belief(t) if belief else sim.truth
    return make_problem(sim.P, field, t, sim.depot, cust, uavs, sim.Tend)


def warm_routes(sim, pb):
    g2l = {g: i for i, g in enumerate(pb.gid) if i > 0}
    return [[g2l[c] for c in u["seq"] if c in g2l] for u in sim.u]


def apply_solution(sim, pb, sol):
    sim.set_plan([[pb.gid[c] for c in r] for r in sol.routes])


# =============================================================== thoi diem kich hoat
def trigger_times(scn, mode, Tend, min_gap=40.0, dt=None):
    """mode: 'once' | 'targets' (chi khi co khach moi) | 'events' (moi su kien) ; dt: them chu ky co dinh."""
    if mode == "once":
        return [0.0]
    times, last, pending = [0.0], 0.0, False
    for e in sorted(scn["events"], key=lambda e: e["t"]):
        if mode == "targets" and e["type"] != "new_target":
            continue
        if pending and last + min_gap <= e["t"] - 1e-9:
            last += min_gap
            times.append(last)
            pending = False
        if e["t"] <= 1e-9:
            continue
        if (mode == "events" and e["type"] in CRITICAL) or e["t"] - last >= min_gap:
            times.append(e["t"])
            last = e["t"]
            pending = False
        else:
            pending = True
    if pending:
        times.append(last + min_gap)
    if dt:
        times += list(np.arange(dt, Tend - dt / 2, dt))
    times = sorted(t for t in set(round(float(t), 3) for t in times) if t < Tend - 10.0)
    out = []
    for t in times:
        if not out or t - out[-1] > 1.0:
            out.append(t)
    return out


# =============================================================== bo dieu phoi dung ALNS
class ALNSPlanner:
    """Bo dieu phoi ALNS dung chung cho: ALNS tinh, re-solve moi su kien, ALNS dong (rolling-horizon)."""

    def __init__(self, name, cfg, iters, trigger_mode, include_all=False, min_gap=40.0, dt=None, seed=0,
                 keep_trace=False):
        self.name, self.cfg, self.iters = name, cfg, iters
        self.trigger_mode, self.include_all = trigger_mode, include_all
        self.min_gap, self.dt, self.seed, self.keep_trace = min_gap, dt, seed, keep_trace
        self.trace = None
        self.archive = None

    def triggers(self, scn, Tend):
        return trigger_times(scn, self.trigger_mode, Tend, self.min_gap, self.dt)

    def plan(self, sim, t, idx):
        pb = build_problem(sim, t, include_all=self.include_all)
        init = warm_routes(sim, pb) if self.cfg.get("warm") else None
        rng = np.random.default_rng(self.seed * 7919 + idx)
        sol, info = solve(pb, init, self.iters, rng, self.cfg, record_trace=self.keep_trace and idx == 0)
        if self.keep_trace and idx == 0:
            self.trace = info["trace"]
        if self.cfg.get("archive") and idx == 0:
            self.archive = info["archive"]
        apply_solution(sim, pb, sol)
        return info


# =============================================================== vong lap chinh
def run_episode(inst, scn, P, planner):
    sim = Sim(inst, scn, P)
    trig = planner.triggers(scn, sim.Tend)
    plan_time, n_replans, reassign = 0.0, 0, 0
    prev = {}
    for idx, t in enumerate(trig):
        sim.advance_to(t)
        t0 = time.perf_counter()
        planner.plan(sim, t, idx)
        plan_time += time.perf_counter() - t0
        n_replans += 1
        cur = {c: k for k, u in enumerate(sim.u) for c in u["seq"]}
        reassign += sum(1 for c, k in cur.items() if c in prev and prev[c] != k) \
            + sum(1 for c in prev if c not in sim.served and c not in cur)
        prev = cur
    sim.advance_to(sim.Tend + 1.0)
    m = sim.metrics()
    m.update(replans=n_replans, plan_time=plan_time, reassign=reassign)
    return m
