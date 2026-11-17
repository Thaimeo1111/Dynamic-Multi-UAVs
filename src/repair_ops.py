"""T14 - Repair operators: greedy, regret-2 va risk-priority repair.

Moi toan tu nhan (pb, sol, rng) va CHEN lai cac khach trong sol.un vao cac tuyen (sua sol tai cho).
Cach chen nhanh: voi moi (khach, tuyen) quet moi vi tri bang cong thuc delta xap xi O(1)
(khoang cach, rui ro, tre cua khach moi + lan truyen tre ve phia sau, dung do tre con thua cua tuyen),
roi XAC NHAN bang eval_route chinh xac truoc khi chap nhan. Nho vay nghiem luon kha thi.
Khach khong chen duoc o dau (hoac chi phi chen >= w_u) duoc de lai trong sol.un.
"""
from __future__ import annotations

import math

from .model import eval_route, set_route

BIG = 1e9


def scan_route(pb, k, info, c, rboost=1.0):
    """Vi tri chen tot nhat (delta_xap_xi, pos) cua khach c vao tuyen k; None neu khong co vi tri nao kha thi."""
    if not pb.alive[k]:
        return None
    if info.load + pb.dem[c] > pb.cap[k] + 1e-9:
        return None
    D, X, BAN = pb.D, pb.X, pb.BAN
    ready, due, serv = pb.ready, pb.due, pb.serv
    Tb, idB, nb1, iv = pb.T0b, pb.idB, pb.nb - 1, pb.inv_v
    w_d, w_r, w_l, Tend = pb.w_d, pb.w_r * rboost, pb.w_l, pb.Tend
    seq, st, arr, dep, leg_d, leg_x = info.seq, info.st, info.arr, info.dep, info.leg_d, info.leg_x
    L = len(seq)
    s_k, t0k = pb.start_node[k], pb.t0[k]
    base = info.dist
    bat = pb.bat[k]
    Dc = D[c]
    rc, dc_, sc = ready[c], due[c], serv[c]
    best_d, best_p = BIG, -1
    for p in range(L + 1):
        if p:
            prev = seq[p - 1]
            pd = dep[p - 1]
        else:
            prev = s_k
            pd = t0k
        nxt = seq[p] if p < L else 0
        d1 = D[prev][c]
        d2 = Dc[nxt]
        dd = d1 + d2 - (leg_d[p] if L else 0.0)
        if base + dd > bat + 1e-9:
            continue
        kb1 = int((pd - Tb) * idB)
        kb1 = 0 if kb1 < 0 else (nb1 if kb1 > nb1 else kb1)
        if BAN[kb1][prev][c]:
            continue
        a = pd + d1 * iv
        s = a if a > rc else rc
        late_c = s - dc_
        if late_c < 0.0:
            late_c = 0.0
        dp = s + sc
        kb2 = int((dp - Tb) * idB)
        kb2 = 0 if kb2 < 0 else (nb1 if kb2 > nb1 else kb2)
        if BAN[kb2][c][nxt]:
            continue
        dx = X[kb1][prev][c] + X[kb2][c][nxt] - (leg_x[p] if L else 0.0)
        arrn = dp + d2 * iv
        extra = 0.0
        if p < L:
            sn = arrn if arrn > ready[nxt] else ready[nxt]
            sh = sn - st[p]
            if sh > 1e-9:
                j = p
                bad = False
                while True:
                    o = st[j] - due[seq[j]]
                    n_ = o + sh
                    extra += (n_ if n_ > 0.0 else 0.0) - (o if o > 0.0 else 0.0)
                    j += 1
                    if j >= L:
                        if info.tret + sh > Tend:
                            bad = True
                        break
                    sh = sh - (st[j] - arr[j])
                    if sh <= 1e-9:
                        break
                if bad:
                    continue
        else:
            if arrn > Tend:
                continue
        delta = w_d * dd + w_r * dx + w_l * (late_c + extra)
        if delta < best_d:
            best_d, best_p = delta, p
    if best_p < 0:
        return None
    return best_d, best_p


def _commit(pb, sol, c, k, pos):
    seq = sol.routes[k][:pos] + [c] + sol.routes[k][pos:]
    info = eval_route(pb, k, seq)
    if info is None:
        return False
    set_route(pb, sol, k, seq, info)
    sol.un.discard(c)
    sol.cost -= pb.w_u
    return True


def site_risk(pb, c, t):
    """(rui ro tai cho khach luc t, dinh trong 3 thung tiep theo, xu huong tang)."""
    kb = int((t - pb.T0b) * pb.idB)
    kb = 0 if kb < 0 else (pb.nb - 1 if kb > pb.nb - 1 else kb)
    row = pb.CR[c]
    now = row[kb]
    peak = max(row[kb:kb + 4])
    return now, peak, max(0.0, peak - now)


def _scan_all(pb, sol, c, rboost=1.0):
    out = {}
    for k in range(pb.K):
        r = scan_route(pb, k, sol.infos[k], c, rboost)
        if r is not None:
            out[k] = r
    return out


def _refresh(pb, sol, cand, U, k, rb=None):
    for c in U:
        r = scan_route(pb, k, sol.infos[k], c, 1.0 if rb is None else rb[c])
        if r is None:
            cand[c].pop(k, None)
        else:
            cand[c][k] = r


def _greedy_like(pb, sol, regret):
    U = list(sol.un)
    cand = {c: _scan_all(pb, sol, c) for c in U}
    while U:
        pick = None
        for c in list(U):
            d = cand[c]
            if not d:
                U.remove(c)                      # khong chen duoc o dau -> de khong phuc vu
                continue
            vals = sorted((v[0], k, v[1]) for k, v in d.items())
            b, k, p = vals[0]
            if b >= pb.w_u:
                U.remove(c)                      # chen con dat hon bo qua
                continue
            if regret:
                score = (vals[1][0] - b) if len(vals) > 1 else BIG - b
                key = (-score, b)
            else:
                key = (b, 0.0)
            if pick is None or key < pick[0]:
                pick = (key, c, k, p)
        if pick is None:
            break
        _, c, k, p = pick
        if _commit(pb, sol, c, k, p):
            U.remove(c)
            _refresh(pb, sol, cand, U, k)
        else:
            cand[c].pop(k, None)


def repair_greedy(pb, sol, rng):
    _greedy_like(pb, sol, regret=False)


def repair_regret2(pb, sol, rng):
    _greedy_like(pb, sol, regret=True)


def repair_risk_priority(pb, sol, rng, beta=2.0):
    """Chen theo thu tu uu tien = (khan cap) x (rui ro tai cho khach, dang tang), chi phi chen tang trong so rui ro.

    khan cap  = 1 / (1 + do tre con thua / 50)  (do tre con thua = due - thoi diem som nhat co the phuc vu)
    rui ro    = 0.05 + rui ro hien tai + 2 * xu huong tang (dinh 3 thung toi - hien tai)
    Khach o vung rui ro dang tang va sap het han duoc chen TRUOC khi vung do nguy hiem hon.
    """
    U = list(sol.un)
    prio, rb = {}, {}
    for c in U:
        t_est = pb.est_start[c]
        now, peak, trend = site_risk(pb, c, t_est)
        slack = max(0.0, pb.due[c] - t_est)
        urg = 1.0 / (1.0 + slack / 50.0)
        prio[c] = urg * (0.05 + now + 2.0 * trend)
        rb[c] = 1.0 + beta * (now + trend)
    U.sort(key=lambda c: (-prio[c], c))
    for c in U:
        cands = _scan_all(pb, sol, c, rb[c])
        while cands:
            k, (b, p) = min(cands.items(), key=lambda kv: kv[1][0])
            if b >= pb.w_u:
                break
            if _commit(pb, sol, c, k, p):
                break
            cands.pop(k)


REPAIR = dict(greedy=repair_greedy, regret2=repair_regret2, riskprio=repair_risk_priority)
