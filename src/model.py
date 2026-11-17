"""Mo hinh bai toan + bang chi phi + danh gia mot tuyen (loi cua ALNS/NSGA-III).

Khai niem
---------
Problem  : bai toan TAI MOT THOI DIEM LAP KE HOACH (snapshot) voi chi so cuc bo:
           nut 0 = depot, 1..m = khach con can phuc vu, m+1..m+K = "diem xuat phat ao" cua tung UAV
           (vi tri hien tai + thoi diem ranh + pin con lai). Nho vay lap ke hoach lai giua chung
           (rolling-horizon) chi la tao mot Problem moi.
Bang     : D (khoang cach), X[k][i][j] (phoi nhiem rui ro cua chang i->j khoi hanh o thung thoi gian k),
           BAN[k][i][j] (chang bi cam: co diem doc duong co R > theta_ban). Tinh tu BELIEF cua bo dieu phoi.
RouteInfo: ket qua danh gia chinh xac mot tuyen (khoang cach, rui ro, tre, kha thi).

Muc tieu (T02):  J = w_d*quang_duong + w_r*phoi_nhiem_rui_ro + w_l*tong_tre + w_u*so_khach_khong_phuc_vu
Rang buoc cung : tai trong, pin (quang duong + ve depot), khong bay qua vung cam (R > theta_ban),
                 ve depot truoc het ngay bay. Tre (qua due) la rang buoc MEM, tinh vao J.
"""
from __future__ import annotations

import math

import numpy as np

DEFAULT_P = dict(
    speed=1.0,        # don vi khoang cach / don vi thoi gian
    eps=1.0,          # nang luong / don vi khoang cach (pin tinh bang quang duong con bay duoc = nang luong / eps)
    battery=300.0,    # nang luong ban dau cua moi UAV (quy ra quang duong khi eps = 1)
    w_d=1.0, w_r=5.0, w_l=1.0, w_u=300.0,
    theta_ban=0.8,    # R vuot nguong nay -> vung cam bay
    depot_safe=8.0,   # ban kinh quanh depot khong tinh la vung cam (depot phai luon ra/vao duoc)
    dB=20.0,          # be rong thung thoi gian cua bang rui ro
    n_samples=6,      # so diem lay mau doc mot chang
    cell=2.0,         # kich thuoc o luoi risk map
)


class RouteInfo:
    __slots__ = ("seq", "cost", "dist", "risk", "late", "nlate", "load", "arr", "st", "dep",
                 "leg_d", "leg_x", "tret")


EMPTY_INFO = RouteInfo()
for _a in ("cost", "dist", "risk", "late", "load"):
    setattr(EMPTY_INFO, _a, 0.0)
EMPTY_INFO.seq, EMPTY_INFO.nlate = [], 0
EMPTY_INFO.arr = EMPTY_INFO.st = EMPTY_INFO.dep = EMPTY_INFO.leg_d = EMPTY_INFO.leg_x = []
EMPTY_INFO.tret = 0.0


class Problem:
    """Bai toan lap ke hoach tai mot thoi diem. Xem make_problem()."""


def make_problem(P, field, t_now, depot_xy, cust, uavs, Tend):
    """Xay Problem tu belief `field`.

    cust: dict(gid=[...], xy=(m,2), ready=[...], due=[...], serv=[...], dem=[...])  (khach can lap ke hoach)
    uavs: list dict(x, y, t_free, bat, cap, alive)
    """
    pb = Problem()
    pb.P = P
    pb.m = m = len(cust["gid"])
    pb.K = K = len(uavs)
    pb.gid = [0] + list(cust["gid"])
    zero = np.zeros(1)
    pb.ready = np.concatenate([zero, np.asarray(cust["ready"], float), np.zeros(K)]).tolist()
    pb.due = np.concatenate([zero, np.asarray(cust["due"], float), np.zeros(K)]).tolist()
    pb.serv = np.concatenate([zero, np.asarray(cust["serv"], float), np.zeros(K)]).tolist()
    pb.dem = np.concatenate([zero, np.asarray(cust["dem"], float), np.zeros(K)]).tolist()
    cxy = np.asarray(cust["xy"], float).reshape(-1, 2)
    uxy = np.array([[u["x"], u["y"]] for u in uavs], float).reshape(-1, 2)
    xy = np.vstack([np.asarray(depot_xy, float)[None], cxy, uxy])
    N0 = len(xy)
    pb.xy = xy
    pb.start_node = [m + 1 + k for k in range(K)]
    pb.t0 = [float(u["t_free"]) for u in uavs]
    pb.bat = [float(u["bat"]) for u in uavs]
    pb.cap = [float(u["cap"]) for u in uavs]
    pb.alive = [bool(u["alive"]) for u in uavs]
    pb.Tend = float(Tend)
    pb.inv_v = 1.0 / P["speed"]
    pb.w_d, pb.w_r, pb.w_l, pb.w_u = P["w_d"], P["w_r"], P["w_l"], P["w_u"]
    pb.t_now = float(t_now)

    d = xy[:, None, :] - xy[None, :, :]
    Dm = np.hypot(d[..., 0], d[..., 1])
    pb.D = Dm.tolist()
    dB = float(P["dB"])
    T0b = math.floor(t_now / dB) * dB
    nb = max(2, int(math.ceil((Tend - T0b) / dB)) + 1)
    pb.dB, pb.idB, pb.T0b, pb.nb = dB, 1.0 / dB, T0b, nb

    df = dB / 2.0
    M = int(math.ceil((Tend - T0b) / df)) + 3
    frames = field.frames(T0b + np.arange(M) * df)            # (M, H, W)
    S = int(P["n_samples"])
    F = (np.arange(S) + 1.0) / S
    x0, y0 = field.bbox[0], field.bbox[1]
    px = xy[:, None, None, 0] + F[None, None, :] * (xy[None, :, None, 0] - xy[:, None, None, 0])
    py = xy[:, None, None, 1] + F[None, None, :] * (xy[None, :, None, 1] - xy[:, None, None, 1])
    ix = np.clip(((px - x0) // field.cell).astype(np.int32), 0, field.W - 1)
    iy = np.clip(((py - y0) // field.cell).astype(np.int32), 0, field.H - 1)
    tau = F[None, None, :] * Dm[:, :, None] * pb.inv_v
    X, BAN = [], []
    th = P["theta_ban"]
    near0 = np.hypot(px - xy[0, 0], py - xy[0, 1]) <= P.get("depot_safe", 0.0)
    for k in range(nb):
        mi = np.clip(np.rint(((k + 0.5) * dB + tau) / df).astype(np.int32), 0, M - 1)
        R = frames[mi, iy, ix]
        X.append((R.mean(axis=2) * Dm).tolist())
        BAN.append((np.where(near0, 0.0, R).max(axis=2) > th).tolist())
    pb.X, pb.BAN = X, BAN
    # rui ro tai vi tri tung nut theo thung (dung cho removal/priority)
    nix = np.clip(((xy[:, 0] - x0) // field.cell).astype(np.int32), 0, field.W - 1)
    niy = np.clip(((xy[:, 1] - y0) // field.cell).astype(np.int32), 0, field.H - 1)
    mid = np.clip(np.rint((np.arange(nb) + 0.5) * dB / df).astype(np.int32), 0, M - 1)
    pb.CR = frames[mid][:, niy, nix].T.tolist()                # CR[node][k]
    # uoc luong thoi diem som nhat co the bat dau phuc vu (dung cho muc khan cap)
    est = []
    for c in range(1, m + 1):
        e = min((pb.t0[k] + Dm[pb.start_node[k]][c] * pb.inv_v) for k in range(K)) if K else 0.0
        est.append(max(e, pb.ready[c]))
    pb.est_start = [0.0] + est
    pb._rel = None
    return pb


def bucket(pb, t):
    kb = int((t - pb.T0b) * pb.idB)
    if kb < 0:
        return 0
    return kb if kb < pb.nb else pb.nb - 1


def eval_route(pb, k, seq):
    """Danh gia chinh xac tuyen `seq` (list chi so cuc bo) cua UAV k. Tra ve RouteInfo hoac None neu vi pham rang buoc cung."""
    if not seq:
        return EMPTY_INFO
    if not pb.alive[k]:
        return None
    D, X, BAN = pb.D, pb.X, pb.BAN
    ready, due, serv, dem = pb.ready, pb.due, pb.serv, pb.dem
    Tb, idB, nb1, iv = pb.T0b, pb.idB, pb.nb - 1, pb.inv_v
    cur = pb.start_node[k]
    t = pb.t0[k]
    dist = risk = late = load = 0.0
    nlate = 0
    arr_l, st_l, dep_l, ld, lx = [], [], [], [], []
    for c in seq:
        d = D[cur][c]
        kb = int((t - Tb) * idB)
        kb = 0 if kb < 0 else (nb1 if kb > nb1 else kb)
        if BAN[kb][cur][c]:
            return None
        x = X[kb][cur][c]
        a = t + d * iv
        s = a if a > ready[c] else ready[c]
        lt = s - due[c]
        if lt > 0.0:
            late += lt
            nlate += 1
        t = s + serv[c]
        arr_l.append(a)
        st_l.append(s)
        dep_l.append(t)
        ld.append(d)
        lx.append(x)
        dist += d
        risk += x
        load += dem[c]
        cur = c
    d = D[cur][0]
    kb = int((t - Tb) * idB)
    kb = 0 if kb < 0 else (nb1 if kb > nb1 else kb)
    if BAN[kb][cur][0]:
        return None
    x = X[kb][cur][0]
    tret = t + d * iv
    dist += d
    risk += x
    ld.append(d)
    lx.append(x)
    if tret > pb.Tend or load > pb.cap[k] + 1e-9 or dist > pb.bat[k] + 1e-9:
        return None
    r = RouteInfo()
    r.seq = list(seq)
    r.dist, r.risk, r.late, r.nlate, r.load = dist, risk, late, nlate, load
    r.cost = pb.w_d * dist + pb.w_r * risk + pb.w_l * late
    r.arr, r.st, r.dep, r.leg_d, r.leg_x, r.tret = arr_l, st_l, dep_l, ld, lx, tret
    return r


# ----------------------------------------------------------------- nghiem
class Sol:
    __slots__ = ("routes", "infos", "un", "cost")

    def copy(self):
        s = Sol()
        s.routes = [r[:] for r in self.routes]
        s.infos = list(self.infos)
        s.un = set(self.un)
        s.cost = self.cost
        return s


def empty_solution(pb):
    s = Sol()
    s.routes = [[] for _ in range(pb.K)]
    s.infos = [EMPTY_INFO] * pb.K
    s.un = set(range(1, pb.m + 1))
    s.cost = pb.w_u * len(s.un)
    return s


def set_route(pb, sol, k, seq, info):
    sol.cost += info.cost - sol.infos[k].cost
    sol.routes[k] = list(seq)
    sol.infos[k] = info


def components(pb, sol):
    """(quang duong, rui ro, tre, so_tre, so_khong_phuc_vu) cua mot nghiem."""
    dist = sum(i.dist for i in sol.infos)
    risk = sum(i.risk for i in sol.infos)
    late = sum(i.late for i in sol.infos)
    nlate = sum(i.nlate for i in sol.infos)
    return dist, risk, late, nlate, len(sol.un)


def recompute_cost(pb, sol):
    sol.cost = sum(i.cost for i in sol.infos) + pb.w_u * len(sol.un)
    return sol.cost


def solution_from_routes(pb, routes):
    """Dung nghiem tu cac tuyen (chi so cuc bo). Tuyen khong kha thi bi cat dan tu cuoi cho den khi kha thi."""
    sol = empty_solution(pb)
    placed = set()
    for k, seq in enumerate(routes):
        seq = [c for c in seq if 1 <= c <= pb.m and c not in placed]
        info = eval_route(pb, k, seq)
        while info is None and seq:
            seq.pop()
            info = eval_route(pb, k, seq)
        if info is None:
            info = EMPTY_INFO
            seq = []
        set_route(pb, sol, k, seq, info)
        placed.update(seq)
    sol.un = set(range(1, pb.m + 1)) - placed
    recompute_cost(pb, sol)
    return sol


def relatedness(pb):
    """Ma tran lien quan kieu Shaw (nho = gan nhau): khoang cach, cua so thoi gian, tai trong."""
    if pb._rel is not None:
        return pb._rel
    m = pb.m
    D = np.array(pb.D)[1:m + 1, 1:m + 1] if m else np.zeros((0, 0))
    rd = np.array(pb.ready[1:m + 1])
    du = np.array(pb.due[1:m + 1])
    dm = np.array(pb.dem[1:m + 1])
    dn = D.max() + 1e-9 if m else 1.0
    tn = max(np.ptp(rd) if m else 1.0, 1.0)
    qn = max(np.ptp(dm) if m else 1.0, 1.0)
    rel = 9 * D / dn + 3 * (np.abs(rd[:, None] - rd[None]) + np.abs(du[:, None] - du[None])) / (2 * tn) \
        + 2 * np.abs(dm[:, None] - dm[None]) / qn
    pb._rel = rel
    return rel
