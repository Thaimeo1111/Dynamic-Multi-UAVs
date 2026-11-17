"""Cac thuat toan so sanh cua muc 5.1 (Bang 3, 4, 5).

Ma hoa chung: vector gan task -> UAV (do dai n, gia tri 0..u-1).
Trong qua trinh tim kiem, chi phi Eq.14 tinh bang TSP phan tang NN (fast_cost);
ket qua cuoi cua MOI thuat toan duoc tinh lai bang NN + 2-Opt (evaluate) de so cong bang.
Tham so mac dinh theo Bang 3; `iters`/`pop` co the giam o che do --quick.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

from .auction import auction
from .common import evaluate, fast_cost, cost_from, hier_route
from .som3d import som_allocate


# ------------------------------------------------------------------ helpers
def _eval_pop(sc, pop, u):
    return np.array([fast_cost(sc, ind, u) for ind in pop])


def _tournament(rng, fit, k=2):
    a = rng.integers(0, len(fit), size=(len(fit), k))
    return a[np.arange(len(fit)), fit[a].argmin(1)]


def _crossover(rng, pop, pc):
    n_ind, n = pop.shape
    child = pop.copy()
    for i in range(0, n_ind - 1, 2):
        if rng.random() < pc:
            a, b = sorted(rng.integers(0, n, 2))   # lai ghep hai diem
            child[i, a:b + 1], child[i + 1, a:b + 1] = pop[i + 1, a:b + 1].copy(), pop[i, a:b + 1].copy()
    return child


def _mutate(rng, pop, pm, u):
    """Voi xac suat pm moi ca the, doi UAV cua ~2% gen (it nhat 1)."""
    n_ind, n = pop.shape
    for i in range(n_ind):
        if rng.random() < pm:
            k = max(1, int(0.02 * n))
            g = rng.integers(0, n, k)
            pop[i, g] = rng.integers(0, u, k)
    return pop


# ------------------------------------------------------------------ GA
def ga(sc, u, rng, pop_size=100, iters=200, pc=0.8, pm=0.1):
    n = sc["n"]
    pop = rng.integers(0, u, (pop_size, n))
    fit = _eval_pop(sc, pop, u)
    for _ in range(iters):
        elite = pop[fit.argmin()].copy(); ef = fit.min()
        sel = pop[_tournament(rng, fit)]
        pop = _mutate(rng, _crossover(rng, sel, pc), pm, u)
        fit = _eval_pop(sc, pop, u)
        w = fit.argmax(); pop[w] = elite; fit[w] = ef      # elitism
    return pop[fit.argmin()]


# ------------------------------------------------------------------ PSO
def _decode(x, u):
    return np.clip(np.floor(x), 0, u - 1).astype(int)


def pso(sc, u, rng, pop_size=100, iters=200, c1=1.5, c2=1.5, w=0.729, hybrid=False, pc=0.8, pm=0.1):
    n = sc["n"]
    x = rng.uniform(0, u, (pop_size, n))
    v = rng.uniform(-1, 1, (pop_size, n))
    fit = _eval_pop(sc, _decode(x, u), u)
    pb, pbf = x.copy(), fit.copy()
    g = pb[pbf.argmin()].copy(); gf = pbf.min()
    vmax = u / 2
    for _ in range(iters):
        r1, r2 = rng.random((2, pop_size, n))
        v = np.clip(w * v + c1 * r1 * (pb - x) + c2 * r2 * (g - x), -vmax, vmax)
        x = np.clip(x + v, 0, u - 1e-9)
        if hybrid:   # GA-PSO (Garg 2016): ap dung lai ghep/dot bien GA len quan the PSO
            order = rng.permutation(pop_size)
            x[order] = _crossover(rng, x[order], pc)
            mask = rng.random(pop_size) < pm
            k = max(1, int(0.02 * n))
            for i in np.flatnonzero(mask):
                gidx = rng.integers(0, n, k)
                x[i, gidx] = rng.uniform(0, u, k)
        fit = _eval_pop(sc, _decode(x, u), u)
        better = fit < pbf
        pb[better] = x[better]; pbf[better] = fit[better]
        if pbf.min() < gf:
            g = pb[pbf.argmin()].copy(); gf = pbf.min()
    return _decode(g, u)


def gapso(sc, u, rng, **kw):
    return pso(sc, u, rng, hybrid=True, **kw)


# ------------------------------------------------------------------ NSGA-II
def _objs(sc, ind, u):
    n = sc["n"]
    L = np.zeros(u); R = np.zeros(u)
    for j in range(u):
        idx = np.flatnonzero(ind == j)
        L[j] = hier_route(sc, n, idx, opt=False)[1]
        R[j] = sc["scores"][idx].sum()
    C, P, D, Rs = cost_from(L, R)
    return np.array([P, D, Rs]), C


def _nondominated_sort(F):
    N = len(F)
    dom = (F[:, None, :] <= F[None, :, :]).all(-1) & (F[:, None, :] < F[None, :, :]).any(-1)
    n_dom = dom.sum(0)          # so ca the troi hon j
    rank = np.full(N, -1)
    cur = np.flatnonzero(n_dom == 0); r = 0
    n_dom = n_dom.copy()
    while len(cur):
        rank[cur] = r
        for i in cur:
            n_dom[dom[i]] -= 1
        n_dom[cur] = -1
        cur = np.flatnonzero(n_dom == 0); r += 1
    return rank


def _crowding(F, rank):
    cd = np.zeros(len(F))
    for r in np.unique(rank):
        idx = np.flatnonzero(rank == r)
        if len(idx) <= 2:
            cd[idx] = np.inf; continue
        for m in range(F.shape[1]):
            o = idx[np.argsort(F[idx, m])]
            span = F[o[-1], m] - F[o[0], m] or 1.0
            cd[o[0]] = cd[o[-1]] = np.inf
            cd[o[1:-1]] += (F[o[2:], m] - F[o[:-2], m]) / span
    return cd


def nsga2(sc, u, rng, pop_size=100, iters=200, pc=0.8, pm=0.1):
    n = sc["n"]
    pop = rng.integers(0, u, (pop_size, n))
    ev = [_objs(sc, p, u) for p in pop]
    F = np.array([e[0] for e in ev]); C = np.array([e[1] for e in ev])
    for _ in range(iters):
        rank = _nondominated_sort(F); cd = _crowding(F, rank)
        a = rng.integers(0, pop_size, (pop_size, 2))
        better = (rank[a[:, 0]] < rank[a[:, 1]]) | ((rank[a[:, 0]] == rank[a[:, 1]]) & (cd[a[:, 0]] > cd[a[:, 1]]))
        par = pop[np.where(better, a[:, 0], a[:, 1])]
        kids = _mutate(rng, _crossover(rng, par, pc), pm, u)
        ev = [_objs(sc, p, u) for p in kids]
        allp = np.vstack([pop, kids])
        allF = np.vstack([F, [e[0] for e in ev]]); allC = np.concatenate([C, [e[1] for e in ev]])
        rank = _nondominated_sort(allF); cd = _crowding(allF, rank)
        keep = np.lexsort((-cd, rank))[:pop_size]
        pop, F, C = allp[keep], allF[keep], allC[keep]
    # chon nghiem tren mat Pareto co gia Eq.14 nho nhat
    return pop[C.argmin()]


# ------------------------------------------------------------------ ACO
def aco(sc, u, rng, ants=100, iters=200, rho=1.0, Q=100.0, a=1.0, b=1.0):
    n = sc["n"]; xy = sc["xy"]; base = sc["D"].shape[0] - 1
    tau = np.ones((n, u))
    best, bestf = None, np.inf
    for _ in range(iters):
        order = rng.permutation(n)
        last = np.repeat(np.array([[50.0, 50.0]]), ants * u, 0).reshape(ants, u, 2)
        load = np.zeros((ants, u))
        sol = np.zeros((ants, n), int)
        for i in order:
            d = np.sqrt(((last - xy[i]) ** 2).sum(-1)) + 1e-6
            # heuristic: gan voi diem cuoi cua UAV, va phat nhe UAV dang nang tai
            eta = 1.0 / (d + 0.05 * load)
            p = (tau[i] ** a)[None] * eta ** b
            p /= p.sum(1, keepdims=True)
            ch = (p.cumsum(1) > rng.random((ants, 1))).argmax(1)
            sol[:, i] = ch
            ar = np.arange(ants)
            load[ar, ch] += d[ar, ch]
            last[ar, ch] = xy[i]
        fit = _eval_pop(sc, sol, u)
        if fit.min() < bestf:
            bestf = fit.min(); best = sol[fit.argmin()].copy()
        tau *= (1 - rho)
        for k in range(ants):
            tau[np.arange(n), sol[k]] += Q / fit[k]
        tau[np.arange(n), best] += Q / bestf
        tau += 1e-6
    return best


# ------------------------------------------------------------------ DQN (numpy)
class _MLP:
    def __init__(self, nin, nout, rng, h=64):
        self.W1 = rng.normal(0, np.sqrt(2 / nin), (nin, h)); self.b1 = np.zeros(h)
        self.W2 = rng.normal(0, np.sqrt(2 / h), (h, h)); self.b2 = np.zeros(h)
        self.W3 = rng.normal(0, np.sqrt(1 / h), (h, nout)); self.b3 = np.zeros(nout)
        self.params = [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]
        self.m = [np.zeros_like(p) for p in self.params]; self.v = [np.zeros_like(p) for p in self.params]
        self.t = 0

    def forward(self, x):
        h1 = np.maximum(0, x @ self.W1 + self.b1)
        h2 = np.maximum(0, h1 @ self.W2 + self.b2)
        return h2 @ self.W3 + self.b3, (x, h1, h2)

    def copy_from(self, o):
        for p, q in zip(self.params, o.params):
            p[...] = q

    def step(self, cache, grad_out, lr):
        x, h1, h2 = cache
        g3 = h2.T @ grad_out; gb3 = grad_out.sum(0)
        d2 = (grad_out @ self.W3.T) * (h2 > 0)
        g2 = h1.T @ d2; gb2 = d2.sum(0)
        d1 = (d2 @ self.W2.T) * (h1 > 0)
        g1 = x.T @ d1; gb1 = d1.sum(0)
        self.t += 1
        for k, g in enumerate([g1, gb1, g2, gb2, g3, gb3]):
            self.m[k] = 0.9 * self.m[k] + 0.1 * g
            self.v[k] = 0.999 * self.v[k] + 0.001 * g * g
            mh = self.m[k] / (1 - 0.9 ** self.t); vh = self.v[k] / (1 - 0.999 ** self.t)
            self.params[k] -= lr * mh / (np.sqrt(vh) + 1e-8)


def dqn(sc, u, rng, episodes=200, gamma=0.99, eps_s=1.0, eps_e=0.01, bs=32, rs=10000, lr=1e-3,
        train_every=4, target_every=200):
    """Gan tuan tu tung task (theo thu tu uu tien, roi goc quanh base) cho mot UAV.
    Trang thai: khoang cach tu diem cuoi moi UAV toi task, tai (quang duong) va diem moi UAV, loai task.
    Thuong: -(tang duong bay) cong phat can bang o cuoi tap = -(1.5*D + 2*R)."""
    n = sc["n"]; xy = sc["xy"]; types = sc["types"]; scores = sc["scores"]
    ang = np.arctan2(xy[:, 1] - 50, xy[:, 0] - 50)
    order = np.lexsort((ang, types))
    nt = int(types.max()) + 1
    nin = 3 * u + nt
    q = _MLP(nin, u, rng); qt = _MLP(nin, u, rng); qt.copy_from(q)
    buf_s = np.zeros((rs, nin)); buf_a = np.zeros(rs, int); buf_r = np.zeros(rs)
    buf_s2 = np.zeros((rs, nin)); buf_d = np.zeros(rs)
    ptr = 0; size = 0; step = 0
    scale = 100.0

    def state(last, L, S, i):
        d = np.sqrt(((last - xy[i]) ** 2).sum(1)) / scale
        oh = np.zeros(nt); oh[types[i]] = 1
        return np.concatenate([d, L / (L.mean() + scale), S / (S.mean() + 1), oh])

    def rollout(eps, learn):
        nonlocal ptr, size, step
        last = np.tile([50.0, 50.0], (u, 1)); L = np.zeros(u); S = np.zeros(u)
        assign = np.zeros(n, int)
        s = state(last, L, S, order[0])
        for k, i in enumerate(order):
            if rng.random() < eps:
                a = int(rng.integers(u))
            else:
                a = int(q.forward(s[None])[0][0].argmax())
            d = float(np.sqrt(((last[a] - xy[i]) ** 2).sum()))
            L[a] += d; S[a] += scores[i]; last[a] = xy[i]; assign[i] = a
            done = k == n - 1
            r = -d / scale
            if done:
                r -= (1.5 * L.std(ddof=1) + 2 * S.std(ddof=1)) / scale
                s2 = s
            else:
                s2 = state(last, L, S, order[k + 1])
            if learn:
                buf_s[ptr] = s; buf_a[ptr] = a; buf_r[ptr] = r; buf_s2[ptr] = s2; buf_d[ptr] = done
                ptr = (ptr + 1) % rs; size = min(size + 1, rs); step += 1
                if size >= bs and step % train_every == 0:
                    b = rng.integers(0, size, bs)
                    qn = qt.forward(buf_s2[b])[0].max(1)
                    y = buf_r[b] + gamma * (1 - buf_d[b]) * qn
                    out, cache = q.forward(buf_s[b])
                    g = np.zeros_like(out)
                    g[np.arange(bs), buf_a[b]] = (out[np.arange(bs), buf_a[b]] - y) / bs
                    q.step(cache, g, lr)
                if step % target_every == 0:
                    qt.copy_from(q)
            s = s2
        return assign

    best, bestf = None, np.inf
    for ep in range(episodes):
        eps = eps_s * (eps_e / eps_s) ** (ep / max(1, episodes - 1))
        a = rollout(eps, True)
        f = fast_cost(sc, a, u)
        if f < bestf:
            best, bestf = a, f
    a = rollout(0.0, False)
    return a if fast_cost(sc, a, u) <= bestf else best


# ------------------------------------------------------------------ NN, 3D-SOM, KCA, OUR
def nn_greedy(sc, u, rng=None):
    """Baseline tat dinh: theo thu tu uu tien, UAV co quang duong ngan nhat
    luon chon task chua tham gan no nhat."""
    n = sc["n"]; xy = sc["xy"]; types = sc["types"]
    assign = np.full(n, -1)
    pos = np.tile([50.0, 50.0], (u, 1)); L = np.zeros(u)
    for t in np.unique(types):
        left = list(np.flatnonzero(types == t))
        while left:
            j = int(L.argmin())
            arr = np.array(left)
            d = np.sqrt(((xy[arr] - pos[j]) ** 2).sum(1))
            k = int(d.argmin()); i = arr[k]
            assign[i] = j; L[j] += d[k]; pos[j] = xy[i]
            left.pop(k)
    return assign


def som_only(sc, u, rng, alpha=3.0):
    return som_allocate(sc, u, alpha=alpha, rng=rng)


def kca(sc, u, rng):
    lab = KMeans(n_clusters=u, n_init=1, random_state=int(rng.integers(1 << 31))).fit_predict(sc["xy"])
    return auction(sc, lab, u)[0]


def ours(sc, u, rng, alpha=3.0):
    a = som_allocate(sc, u, alpha=alpha, rng=rng)
    return auction(sc, a, u)[0]


ALGOS = {
    "GA": ga, "PSO": pso, "GAPSO": gapso, "NSGA2": nsga2, "ACO": aco, "DQN": dqn,
    "NN": nn_greedy, "3D-SOM": som_only, "KCA": kca, "OUR": ours,
}
DETERMINISTIC = {"NN"}
