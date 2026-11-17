"""Phan dung chung cho tai lap Pu, Yang, Li (2026), AST 172:111711.

- Sinh kich ban (ban do 100x100, base [50,50], 5 UAV).
- TSP: nearest neighbour (NN) + 2-Opt, duong MO (khong quay ve base, muc 3.3.2).
- TSP phan tang (hierarchical): di het loai uu tien cao (A) roi moi toi B, C.
- Ham chi phi Eq.14: C = 1*P + 1.5*D + 2*R.

Quy uoc loai task: types = 0 (A, uu tien cao nhat), 1 (B), 2 (C).
"""
from __future__ import annotations

import numpy as np

LAMBDA = (1.0, 1.5, 2.0)          # lambda1 : lambda2 : lambda3 = 1 : 1.5 : 2 (Eq.14)
BASE = np.array([50.0, 50.0])     # base o tam ban do
MAP = 100.0
N_UAV = 5
# Da loai (muc 5.2): ti le 6:3:1 va diem [1,3,5]. Loai hiem nhat co diem cao nhat
# va uu tien cao nhat (giong Bang 1: A:B:C = 1:3:6, A uu tien cao nhat).
MULTI_FRAC = (0.1, 0.3, 0.6)      # A, B, C
MULTI_SCORE = np.array([5.0, 3.0, 1.0])
DDOF = 1  # Hinh 4: std(20,15)=3.533 -> do lech chuan MAU (MATLAB std), khong phai 1/u nhu Eq.16


def make_scenario(n: int, multi: bool, seed: int, frac=MULTI_FRAC) -> dict:
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0.0, MAP, size=(n, 2))
    if multi:
        counts = np.floor(np.array(frac) * n).astype(int)
        counts[-1] = n - counts[:-1].sum()
        types = np.repeat(np.arange(len(frac)), counts)
        rng.shuffle(types)
        scores = MULTI_SCORE[types]
    else:
        types = np.zeros(n, dtype=int)
        scores = np.ones(n)
    return build(xy, types, scores)


def build(xy, types, scores) -> dict:
    xy = np.asarray(xy, float)
    pts = np.vstack([xy, BASE])                      # chi so n la base
    d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
    return {"xy": xy, "types": np.asarray(types, int), "scores": np.asarray(scores, float),
            "D": d, "n": len(xy)}


# ---------------------------------------------------------------- TSP
def _nn_order(D, start, idx):
    """NN tren tap chi so idx, bat dau tu dinh start (chi so trong D). Tra ve thu tu."""
    k = len(idx)
    if k == 0:
        return idx
    sub = D[np.ix_(idx, idx)].copy()
    first = D[start, idx]
    order = np.empty(k, dtype=int)
    cur = int(np.argmin(first))
    order[0] = cur
    sub[:, cur] = np.inf
    for s in range(1, k):
        nxt = int(np.argmin(sub[cur]))
        order[s] = nxt
        sub[:, nxt] = np.inf
        cur = nxt
    return idx[order]


def _two_opt(D, start, route, max_pass=50):
    """2-Opt cho duong mo co diem dau co dinh."""
    r = np.concatenate([[start], route])
    m = len(r)
    if m < 4:
        return route
    for _ in range(max_pass):
        improved = False
        for i in range(1, m - 1):
            a = r[i - 1]
            b = r[i]
            js = np.arange(i + 1, m)
            c = r[js]
            nxt = np.where(js + 1 < m, r[np.minimum(js + 1, m - 1)], -1)
            last = nxt < 0
            d_cn = np.where(last, 0.0, D[c, np.where(last, 0, nxt)])
            d_bn = np.where(last, 0.0, D[b, np.where(last, 0, nxt)])
            delta = D[a, c] + d_bn - D[a, b] - d_cn
            j = int(np.argmin(delta))
            if delta[j] < -1e-9:
                jj = js[j]
                r[i:jj + 1] = r[i:jj + 1][::-1]
                improved = True
        if not improved:
            break
    return r[1:]


def route_len(D, start, route):
    if len(route) == 0:
        return 0.0
    return float(D[start, route[0]] + D[route[:-1], route[1:]].sum())


def hier_route(sc, start, idx, opt=True):
    """TSP phan tang: loai 0 -> 1 -> 2, moi tang bat dau tu diem cuoi cua tang truoc."""
    D, types = sc["D"], sc["types"]
    idx = np.asarray(idx, dtype=int)
    route = []
    cur = start
    for t in np.unique(types[idx]) if len(idx) else []:
        sub = idx[types[idx] == t]
        r = _nn_order(D, cur, sub)
        if opt:
            r = _two_opt(D, cur, r)
        route.append(r)
        cur = r[-1]
    route = np.concatenate(route) if route else np.array([], dtype=int)
    return route, route_len(D, start, route)


# ---------------------------------------------------------------- cost
def cost_from(lengths, scores_per_uav):
    lengths = np.asarray(lengths, float)
    r = np.asarray(scores_per_uav, float)
    P = lengths.sum()
    Dd = lengths.std(ddof=DDOF) if len(lengths) > 1 else 0.0
    R = r.std(ddof=DDOF) if len(r) > 1 else 0.0
    l1, l2, l3 = LAMBDA
    return l1 * P + l2 * Dd + l3 * R, P, Dd, R


def evaluate(sc, assign, u, starts=None, opt=True, return_routes=False):
    """assign[i] = UAV cua task i. starts: chi so dinh xuat phat cua tung UAV (mac dinh base)."""
    n = sc["n"]
    if starts is None:
        starts = [n] * u
    lengths = np.zeros(u)
    sc_per = np.zeros(u)
    routes = []
    for j in range(u):
        idx = np.flatnonzero(assign == j)
        r, L = hier_route(sc, starts[j], idx, opt=opt)
        lengths[j] = L
        sc_per[j] = sc["scores"][idx].sum()
        routes.append(r)
    C, P, Dd, R = cost_from(lengths, sc_per)
    out = {"cost": C, "P": P, "D": Dd, "R": R, "lengths": lengths}
    if return_routes:
        out["routes"] = routes
    return out


def fast_cost(sc, assign, u):
    """Chi phi dung trong vong tim kiem (NN, khong 2-Opt) - cho cac metaheuristic."""
    return evaluate(sc, assign, u, opt=False)["cost"]
