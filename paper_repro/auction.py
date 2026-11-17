"""Toi uu dau gia thu cap (muc 3.2.2, Eq.14-18, Algorithm 2).

Voi tung task: thu chuyen sang tung UAV khac, tinh lai gia toan cuc Eq.14;
neu gia giam thi giu thay doi. Lap toi da A_max = 20 vong (Bang 3),
dung som neu mot vong khong cai thien.

Ghi chu: Algorithm 2 trong bai co nhanh "else break" khi c' == c, lam vong for
dung giua chung. Ta lam theo y trong loi van: thu moi c' != c.
Trong vong dau gia duong bay tinh bang TSP phan tang NN (khong 2-Opt) cho nhanh;
chi phi cuoi cung duoc tinh lai bang NN + 2-Opt.
"""
from __future__ import annotations

import numpy as np

from .common import cost_from, hier_route


def auction(sc, assign, u, starts=None, a_max=20, rng=None):
    n = sc["n"]
    if starts is None:
        starts = [n] * u
    assign = assign.copy()
    scores = sc["scores"]
    L = np.zeros(u)
    R = np.zeros(u)
    for j in range(u):
        idx = np.flatnonzero(assign == j)
        L[j] = hier_route(sc, starts[j], idx, opt=False)[1]
        R[j] = scores[idx].sum()
    best = cost_from(L, R)[0]
    history = [best]
    for _ in range(a_max):
        improved = False
        for i in range(n):
            c = assign[i]
            idx_c = np.flatnonzero(assign == c)
            idx_c_wo = idx_c[idx_c != i]
            Lc_wo = hier_route(sc, starts[c], idx_c_wo, opt=False)[1]
            for c2 in range(u):
                if c2 == c:
                    continue
                idx2 = np.append(np.flatnonzero(assign == c2), i)
                L2 = hier_route(sc, starts[c2], idx2, opt=False)[1]
                Lt = L.copy(); Rt = R.copy()
                Lt[c] = Lc_wo; Lt[c2] = L2
                Rt[c] -= scores[i]; Rt[c2] += scores[i]
                val = cost_from(Lt, Rt)[0]
                if val < best - 1e-9:
                    best = val
                    assign[i] = c2
                    L, R = Lt, Rt
                    improved = True
                    break          # task da doi chu; sang task ke tiep
        history.append(best)
        if not improved:
            break
    return assign, history
