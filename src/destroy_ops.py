"""T13 - Destroy operators: random, worst, related (Shaw) va risk-based removal.

Moi toan tu nhan (pb, sol, q, rng): GO q khach khoi cac tuyen (sua sol tai cho), dua chung vao sol.un,
va tra ve danh sach khach bi go.
"""
from __future__ import annotations

import numpy as np

from .model import EMPTY_INFO, eval_route, relatedness, set_route
from .repair_ops import site_risk


def _assigned(sol):
    out = []
    for k, seq in enumerate(sol.routes):
        for j, c in enumerate(seq):
            out.append((k, j, c))
    return out


def remove_customers(pb, sol, removed):
    removed = set(removed)
    for k, seq in enumerate(sol.routes):
        if not removed.intersection(seq):
            continue
        new = [c for c in seq if c not in removed]
        info = eval_route(pb, k, new) if new else EMPTY_INFO
        if info is None:                       # hiem: go bot lam doi thung thoi gian -> cam. Xoa ca tuyen.
            sol.un.update(new)
            sol.cost += pb.w_u * len(new)
            new, info = [], EMPTY_INFO
        set_route(pb, sol, k, new, info)
    sol.un.update(removed)
    sol.cost += pb.w_u * len(removed)


def _pick_ranked(items, scores, q, rng, p):
    """Chon q phan tu theo thu hang giam dan cua diem, co ngau nhien hoa y = rand^p."""
    order = sorted(range(len(items)), key=lambda i: -scores[i])
    chosen = []
    pool = order[:]
    while pool and len(chosen) < q:
        idx = int(len(pool) * rng.random() ** p)
        chosen.append(items[pool.pop(idx)])
    return chosen


def destroy_random(pb, sol, q, rng):
    cs = [c for _, _, c in _assigned(sol)]
    if not cs:
        return []
    q = min(q, len(cs))
    removed = [cs[i] for i in rng.choice(len(cs), size=q, replace=False)]
    remove_customers(pb, sol, removed)
    return removed


def destroy_worst(pb, sol, q, rng, p=3.0):
    """Go khach co 'loi ich go' lon nhat (tiet kiem nhieu chi phi duong di/rui ro/tre nhat)."""
    items, scores = [], []
    D, X = pb.D, pb.X
    for k, seq in enumerate(sol.routes):
        info = sol.infos[k]
        L = len(seq)
        for j, c in enumerate(seq):
            prev = seq[j - 1] if j else pb.start_node[k]
            nxt = seq[j + 1] if j + 1 < L else 0
            pd = info.dep[j - 1] if j else pb.t0[k]
            kb = min(max(int((pd - pb.T0b) * pb.idB), 0), pb.nb - 1)
            gain = pb.w_d * (info.leg_d[j] + info.leg_d[j + 1] - D[prev][nxt]) \
                + pb.w_r * (info.leg_x[j] + info.leg_x[j + 1] - X[kb][prev][nxt]) \
                + pb.w_l * max(0.0, info.st[j] - pb.due[c])
            items.append(c)
            scores.append(gain)
    if not items:
        return []
    removed = _pick_ranked(items, scores, min(q, len(items)), rng, p)
    remove_customers(pb, sol, removed)
    return removed


def destroy_related(pb, sol, q, rng, p=6.0):
    """Shaw removal: go mot khach ngau nhien roi cac khach 'lien quan' nhat (gan, cua so thoi gian va tai trong tuong tu)."""
    cs = [c for _, _, c in _assigned(sol)]
    if not cs:
        return []
    rel = relatedness(pb)
    q = min(q, len(cs))
    removed = [cs[int(rng.integers(len(cs)))]]
    rest = [c for c in cs if c != removed[0]]
    while len(removed) < q and rest:
        r = removed[int(rng.integers(len(removed)))]
        rest.sort(key=lambda c: rel[r - 1][c - 1])
        removed.append(rest.pop(int(len(rest) * rng.random() ** p)))
    remove_customers(pb, sol, removed)
    return removed


def destroy_risk(pb, sol, q, rng, p=4.0):
    """Risk-based removal: go khach dang nam trong/gan vung RUI RO DANG TANG.

    diem = rui ro tai cho khach luc phuc vu + 2 * xu huong tang (dinh 3 thung toi - hien tai)
           + 0.02 * phoi nhiem rui ro cua hai chang ke khach do.
    Cac khach nay duoc chen lai sau (co the sang UAV/thoi diem khac, hoac de lai neu qua nguy hiem).
    """
    items, scores = [], []
    for k, seq in enumerate(sol.routes):
        info = sol.infos[k]
        for j, c in enumerate(seq):
            now, peak, trend = site_risk(pb, c, info.st[j])
            items.append(c)
            scores.append(now + 2.0 * trend + 0.02 * (info.leg_x[j] + info.leg_x[j + 1]))
    if not items:
        return []
    removed = _pick_ranked(items, scores, min(q, len(items)), rng, p)
    remove_customers(pb, sol, removed)
    return removed


DESTROY = dict(random=destroy_random, worst=destroy_worst, related=destroy_related, risk=destroy_risk)
