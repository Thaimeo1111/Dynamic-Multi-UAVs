"""NSGA-III tu viet (numpy, khong dung pymoo) cho bai toan tinh 3 muc tieu + HV/IGD (T21, T32).

Nhiem sac the = hoan vi khach (nhieu o chuoi); giai ma: lan luot chen khach vao cuoi tuyen UAV
kha thi co chi phi tang them nho nhat (eval_route chinh xac), khong kha thi -> khong phuc vu.
Muc tieu f = (quang duong, rui ro, tre + w_u/w_l * so khong phuc vu)  -- cung vector voi ALNS archive.
"""
import itertools
import numpy as np
from .model import eval_route, solution_from_routes, components


def objective_of(pb, sol):
    d, r, late, _, nun = components(pb, sol)
    return np.array([d, r, late + nun * pb.w_u / pb.w_l])


def decode(pb, perm):
    routes = [[] for _ in range(pb.K)]
    infos = [eval_route(pb, k, []) for k in range(pb.K)]
    for c in perm:
        best = None
        for k in range(pb.K):
            seq = routes[k] + [c]
            info = eval_route(pb, k, seq)
            if info is None:
                continue
            delta = info.cost - infos[k].cost
            if best is None or delta < best[0]:
                best = (delta, k, info)
        if best is not None:
            routes[best[1]].append(c)
            infos[best[1]] = best[2]
    return solution_from_routes(pb, routes)


# ---------------------------------------------------------------- khoi niem Pareto
def nondominated_sort(F):
    n = len(F)
    dom = [[] for _ in range(n)]
    cnt = np.zeros(n, int)
    for i in range(n):
        for j in range(i + 1, n):
            if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                dom[i].append(j); cnt[j] += 1
            elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                dom[j].append(i); cnt[i] += 1
    fronts, cur = [], [i for i in range(n) if cnt[i] == 0]
    while cur:
        fronts.append(cur)
        nxt = []
        for i in cur:
            for j in dom[i]:
                cnt[j] -= 1
                if cnt[j] == 0:
                    nxt.append(j)
        cur = nxt
    return fronts


def pareto_filter(F):
    F = np.unique(np.asarray(F, float), axis=0)
    return F[nondominated_sort(F)[0]] if len(F) else F


def das_dennis(m, p):
    out = []
    for c in itertools.combinations(range(p + m - 1), m - 1):
        b = (-1,) + c + (p + m - 1,)
        out.append([(b[i + 1] - b[i] - 1) / p for i in range(m)])
    return np.array(out)


def _normalize(F, ideal):
    Z = F - ideal
    n = F.shape[1]
    try:  # intercept qua cac diem cuc tri (ASF)
        ext = np.array([Z[np.argmin(np.max(Z / np.where(np.eye(n)[i] > 0, 1.0, 1e-6), axis=1))] for i in range(n)])
        a = np.linalg.solve(ext, np.ones(n))
        b = 1.0 / a
        if not np.all(np.isfinite(b)) or np.any(b <= 1e-9):
            raise np.linalg.LinAlgError
    except np.linalg.LinAlgError:
        b = Z.max(axis=0)
    return Z / np.maximum(b, 1e-9)


def _associate(Zn, refs):
    rn = refs / np.linalg.norm(refs, axis=1, keepdims=True)
    proj = Zn @ rn.T
    d = np.linalg.norm(Zn, axis=1, keepdims=True) ** 2 - proj ** 2
    j = np.argmin(d, axis=1)
    return j, np.sqrt(np.maximum(d[np.arange(len(Zn)), j], 0))


def _niching(Zn_last, assoc_last, dist_last, niche_cnt, need, rng):
    chosen, avail = [], set(range(len(Zn_last)))
    cnt = niche_cnt.copy()
    while len(chosen) < need and avail:
        cand_niches = sorted({assoc_last[i] for i in avail}, key=lambda j: (cnt[j], rng.random()))
        j = cand_niches[0]
        mem = [i for i in avail if assoc_last[i] == j]
        i = min(mem, key=lambda i: dist_last[i]) if cnt[j] > 0 else mem[int(rng.integers(len(mem)))]
        chosen.append(i); avail.discard(i); cnt[j] += 1
    return chosen


def ox(a, b, rng):
    n = len(a)
    i, j = sorted(rng.choice(n + 1, 2, replace=False))
    mid = a[i:j]
    rest = [x for x in b if x not in set(mid)]
    return np.array(rest[:i] + list(mid) + rest[i:])


def mutate(p, rng):
    p = p.copy()
    if len(p) < 2:
        return p
    for _ in range(int(rng.integers(1, 4))):
        i, j = rng.choice(len(p), 2, replace=False)
        if rng.random() < 0.5:
            p[i], p[j] = p[j], p[i]
        else:
            x = p[i]; p = np.delete(p, i); p = np.insert(p, j if j < len(p) else len(p), x)
    return p


def nsga3(pb, pop=40, gens=40, divisions=8, rng=None, seeds=None, p_cx=0.9):
    """Tra ve (F_pareto, evals). seeds: danh sach hoan vi khoi tao (vd. tu nghiem tham lam)."""
    rng = rng or np.random.default_rng(0)
    m = pb.m
    if m == 0:
        return np.zeros((0, 3)), 0
    refs = das_dennis(3, divisions)
    P = [rng.permutation(np.arange(1, m + 1)) for _ in range(pop)]
    for i, s in enumerate(seeds or []):
        P[i] = np.array(s)
    F = np.array([objective_of(pb, decode(pb, p)) for p in P])
    evals = pop
    for _ in range(gens):
        Q = []
        for _ in range(pop):
            a, b = rng.choice(len(P), 2, replace=False)
            c = ox(P[a], P[b], rng) if rng.random() < p_cx else P[a].copy()
            Q.append(mutate(c, rng))
        FQ = np.array([objective_of(pb, decode(pb, p)) for p in Q])
        evals += pop
        R, FR = P + Q, np.vstack([F, FQ])
        fronts = nondominated_sort(FR)
        sel, k = [], 0
        while k < len(fronts) and len(sel) + len(fronts[k]) <= pop:
            sel += fronts[k]; k += 1
        if len(sel) < pop and k < len(fronts):
            Z = _normalize(FR[sel + fronts[k]], FR[sel + fronts[k]].min(axis=0))
            assoc, dist = _associate(Z, refs)
            cnt = np.zeros(len(refs), int)
            for i in range(len(sel)):
                cnt[assoc[i]] += 1
            last = fronts[k]
            pick = _niching(Z[len(sel):], assoc[len(sel):], dist[len(sel):], cnt, pop - len(sel), rng)
            sel += [last[i] for i in pick]
        P = [R[i] for i in sel]
        F = FR[sel]
    return pareto_filter(F), evals


# ---------------------------------------------------------------- chi so chat luong
def hypervolume(F, ref):
    """HV 3 chieu (cuc tieu hoa), chinh xac bang lat cat theo muc tieu thu 3. F, ref: mang."""
    F = np.asarray(F, float)
    ref = np.asarray(ref, float)
    F = F[np.all(F < ref, axis=1)]
    if len(F) == 0:
        return 0.0
    F = F[np.argsort(F[:, 2])]
    zs = list(F[:, 2]) + [ref[2]]
    hv = 0.0
    for i in range(len(F)):
        h = zs[i + 1] - zs[i]
        if h <= 0:
            continue
        hv += h * _hv2(F[:i + 1, :2], ref[:2])
    return hv


def _hv2(P, ref):
    P = P[np.argsort(P[:, 0])]
    hv, best_y = 0.0, ref[1]
    for x, y in P:
        if y < best_y:
            hv += (ref[0] - x) * (best_y - y)
            best_y = y
    return hv


def igd(F, ref_front):
    """IGD: trung binh khoang cach tu moi diem cua mat tham chieu den diem gan nhat cua F."""
    F, R = np.asarray(F, float), np.asarray(ref_front, float)
    if len(F) == 0:
        return float("inf")
    return float(np.mean(np.min(np.linalg.norm(R[:, None, :] - F[None, :, :], axis=2), axis=1)))
