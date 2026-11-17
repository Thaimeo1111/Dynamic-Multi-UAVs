"""3D-SOM (muc 3.2.1, Eq.3-13, Algorithm 1).

Moi task la vector 3 chieu (x, y, loai) chuan hoa ve [0,1]. So neuron = so UAV,
xep thanh chuoi 1D (bai khong noi ro topo; Hinh 2 la luoi, voi u=5 ta dung chuoi).
Khoang cach Eq.6: sqrt(dx^2+dy^2) + alpha*|dp|.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from .common import BASE, MAP


def normalize(sc):
    xy = sc["xy"] / MAP
    t = sc["types"].astype(float)
    tmax = max(t.max(), 1.0)
    return np.column_stack([xy, t / tmax])


def train_som(X, u, alpha, it_max=100, ls=0.1, le=0.01, es=2.0, ee=0.1, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    S = rng.random((u, 3))                         # Eq.5: rand(1,3)
    topo = np.arange(u, dtype=float)               # vi tri neuron tren chuoi 1D
    n = len(X)
    for it in range(it_max):
        k = it / it_max                            # Eq.12
        lr = ls * (le / ls) ** k                   # Eq.9
        eps = es * (ee / es) ** k                  # Eq.11
        for i in rng.permutation(n):
            x = X[i]
            d = np.sqrt(((x[:2] - S[:, :2]) ** 2).sum(1)) + alpha * np.abs(x[2] - S[:, 2])
            bmu = int(np.argmin(d))                # Eq.7
            td = np.abs(topo - topo[bmu])
            B = np.exp(-td ** 2 / (2 * eps ** 2))  # Eq.10
            S += lr * B[:, None] * (x - S)         # Eq.8
    return S


def assign_som(X, S, alpha):
    d = np.sqrt(((X[:, None, :2] - S[None, :, :2]) ** 2).sum(-1)) + alpha * np.abs(X[:, None, 2] - S[None, :, 2])
    return d.argmin(1)                             # Eq.13


def match_clusters_to_uavs(sc, labels, u, starts_xy=None):
    """Gan cum -> UAV theo khoang cach tam cum toi vi tri UAV (Hungarian).
    Khi moi UAV cung xuat phat o base, phep gan nay khong anh huong ket qua."""
    if starts_xy is None:
        return labels
    cent = np.array([sc["xy"][labels == j].mean(0) if np.any(labels == j) else BASE for j in range(u)])
    cost = np.sqrt(((cent[:, None, :] - np.asarray(starts_xy)[None]) ** 2).sum(-1))
    r, c = linear_sum_assignment(cost)
    mp = np.empty(u, int)
    mp[r] = c
    return mp[labels]


def som_allocate(sc, u, alpha=3.0, rng=None, it_max=100, starts_xy=None):
    X = normalize(sc)
    S = train_som(X, u, alpha, it_max=it_max, rng=rng)
    lab = assign_som(X, S, alpha)
    return match_clusters_to_uavs(sc, lab, u, starts_xy)
