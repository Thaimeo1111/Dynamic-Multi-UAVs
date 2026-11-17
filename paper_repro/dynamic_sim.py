"""Mo phong phan bo dong kich hoat boi su kien (muc 3.4 va 5.4, Hinh 8 va 14).

Thiet lap (bai mo ho ve don vi, xem docs/tai_lap_pu2026.md):
  - 100 task ban dau (3 loai, 6:3:1), 5 UAV o base [50,50], van toc v = 1 don vi/buoc (Eq.1-2).
  - Pin E_max = 350 don vi quang duong, ve sac khi con 20%, sac mat 50 buoc.
    (Bai ghi "khoang 10 km" tren ban do 100x100 km - mau thuan; ta chinh de ~4 UAV ve sac giua chung nhu Hinh 14f.)
  - t = 101: them 20 task loai C.
  - 5 diem cuu tro; RBF-MKELM du bao nhu cau; neu du bao >= 90 (90% cua thang 0-100)
    thi sinh task tiep te uu tien cao nhat tai diem do (moi diem nghi 300 buoc).
  - Moi su kien (task moi, UAV ve sac, UAV sac xong, tiep te) -> chay lai 3D-SOM + dau gia + TSP phan tang.
"""
from __future__ import annotations

import numpy as np

from .auction import auction
from .common import MULTI_SCORE, hier_route
from .forecast import make_series, split, rbf_mkelm, windows
from .som3d import som_allocate

V = 1.0
E_MAX = 350.0
LOW = 0.2
CHARGE_T = 50
BASE = np.array([50.0, 50.0])
SERIES_STRIDE = 7          # 1 buoc chuoi du bao = 7 buoc mo phong
SUPPLY_COOLDOWN = 300
SUPPLY_FROM = 150


def _subscenario(xy, types, scores, starts_xy):
    pts = np.vstack([xy, starts_xy])
    D = np.sqrt(((pts[:, None] - pts[None]) ** 2).sum(-1))
    n = len(xy)
    return {"xy": xy, "types": types, "scores": scores, "D": D, "n": n}, list(range(n, n + len(starts_xy)))


def _allocate(tasks, uav_pos, rng, som_it=60):
    """tasks: list dict(id, xy, type). Tra ve {uav_index_in_list: [task ids theo thu tu]}"""
    k = len(uav_pos)
    if not tasks or k == 0:
        return {j: [] for j in range(k)}
    xy = np.array([t["xy"] for t in tasks]); types = np.array([t["type"] for t in tasks])
    scores = MULTI_SCORE[types]
    sc, starts = _subscenario(xy, types, scores, np.array(uav_pos))
    if k == 1:
        a = np.zeros(len(tasks), int)
    else:
        a = som_allocate(sc, k, alpha=3.0, rng=rng, it_max=som_it, starts_xy=np.array(uav_pos)) if len(tasks) >= k \
            else np.arange(len(tasks)) % k
        a, _ = auction(sc, a, k, starts=starts, a_max=10)
    plan = {}
    for j in range(k):
        r, _ = hier_route(sc, starts[j], np.flatnonzero(a == j), opt=True)
        plan[j] = [tasks[i]["id"] for i in r]
    return plan


def _forecast_trigger(rng, quick):
    """Du bao 1 buoc cho moi diem cuu tro tren toan chuoi bang RBF-MKELM (muc tieu holdout)."""
    D = make_series(seed=3)
    pred = np.zeros_like(D)
    for p in range(5):
        y = D[:, p]
        Xtr, ytr, _, _ = split(y)
        X, _ = windows(y)
        out, _, _ = rbf_mkelm(Xtr, ytr, X, rng, "holdout", "de", iters=20 if quick else 60)
        pred[3:, p] = out
    return D, pred


def simulate(seed=1, fig_path=None, quick=False, t_max=3000):
    rng = np.random.default_rng(seed)
    n0 = 100
    types0 = rng.choice(3, n0, p=[0.1, 0.3, 0.6])
    tasks = {i: {"id": i, "xy": rng.uniform(0, 100, 2), "type": int(types0[i])} for i in range(n0)}
    next_id = n0
    rescue = rng.uniform(10, 90, (5, 2))
    series, pred = _forecast_trigger(rng, quick)
    last_supply = np.full(5, -10 ** 9)

    u = 5
    pos = np.tile(BASE, (u, 1)); E = np.full(u, E_MAX)
    state = ["active"] * u; charge_left = np.zeros(u, int)
    plan = {j: [] for j in range(u)}
    done_count = 0
    log, frames = [], []
    path_len = np.zeros(u)

    def realloc(t, reason):
        nonlocal plan
        act = [j for j in range(u) if state[j] == "active"]
        rem = list(tasks.values())
        p = _allocate(rem, [pos[j] for j in act], rng)
        plan = {j: [] for j in range(u)}
        for k, j in enumerate(act):
            plan[j] = p[k]
        log.append({"t": t, "event": reason, "active_uav": len(act), "tasks_left": len(tasks),
                    "done": done_count, "energy": " ".join(f"{e:.0f}" for e in E)})
        frames.append({"t": t, "event": reason, "pos": pos.copy(), "state": list(state),
                       "tasks": {i: (v["xy"].copy(), v["type"]) for i, v in tasks.items()},
                       "plan": {j: list(v) for j, v in plan.items()}, "rescue": rescue})

    realloc(1, "phan bo ban dau")
    t = 1
    while t < t_max:
        t += 1
        events = []
        if t == 101:
            for _ in range(20):
                tasks[next_id] = {"id": next_id, "xy": rng.uniform(0, 100, 2), "type": 2}; next_id += 1
            events.append("them 20 task loai C")
        s_idx = min(t // SERIES_STRIDE, len(series) - 1)
        if t >= SUPPLY_FROM:
            for p in range(5):
                if pred[s_idx, p] >= 90 and t - last_supply[p] >= SUPPLY_COOLDOWN:
                    tasks[next_id] = {"id": next_id, "xy": rescue[p].copy(), "type": 0}; next_id += 1
                    last_supply[p] = t
                    events.append(f"tiep te diem {p + 1} (du bao {pred[s_idx, p]:.0f})")
        # di chuyen
        for j in range(u):
            if state[j] == "active":
                while plan[j] and plan[j][0] not in tasks:
                    plan[j].pop(0)
                if not plan[j]:
                    continue
                tgt = tasks[plan[j][0]]["xy"]
                d = np.linalg.norm(tgt - pos[j])
                step = min(V, d)
                if d > 0:
                    pos[j] += (tgt - pos[j]) / d * step               # Eq.1
                E[j] -= step; path_len[j] += step                       # Eq.2
                if d <= V:
                    del tasks[plan[j].pop(0)]; done_count += 1
                if E[j] <= LOW * E_MAX:
                    state[j] = "returning"; events.append(f"UAV{j + 1} pin thap, ve sac")
            elif state[j] == "returning":
                d = np.linalg.norm(BASE - pos[j]); step = min(V, d)
                if d > 0:
                    pos[j] += (BASE - pos[j]) / d * step
                E[j] -= step; path_len[j] += step
                if d <= V:
                    state[j] = "charging"; charge_left[j] = CHARGE_T
            elif state[j] == "charging":
                charge_left[j] -= 1
                if charge_left[j] <= 0:
                    E[j] = E_MAX; state[j] = "active"; events.append(f"UAV{j + 1} sac xong")
        if events:
            realloc(t, "; ".join(events))
        if not tasks and t > 101 and all(s != "charging" for s in state):
            # het task: ve base
            log.append({"t": t, "event": "hoan thanh moi task", "active_uav": state.count("active"),
                        "tasks_left": 0, "done": done_count, "energy": " ".join(f"{e:.0f}" for e in E)})
            frames.append({"t": t, "event": "hoan thanh", "pos": pos.copy(), "state": list(state),
                           "tasks": {}, "plan": {j: [] for j in range(u)}, "rescue": rescue})
            break
    for r in log:
        r["total_path"] = round(float(path_len.sum()), 1)
    if fig_path is not None:
        _plot(frames, fig_path)
    return log


def _plot(frames, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pick = frames if len(frames) <= 9 else [frames[i] for i in np.linspace(0, len(frames) - 1, 9).astype(int)]
    fig, axs = plt.subplots(3, 3, figsize=(13, 13))
    col = {0: "#d62728", 1: "#ff7f0e", 2: "#1f77b4"}
    ucol = plt.cm.tab10(np.arange(5) + 4)
    for ax, fr in zip(axs.flat, pick):
        for i, (xy, ty) in fr["tasks"].items():
            ax.scatter(*xy, s=12, c=col[ty], marker="o" if ty else "*")
        ax.scatter(*fr["rescue"].T, marker="s", facecolors="none", edgecolors="k", s=50)
        ax.scatter(50, 50, marker="^", c="k", s=60)
        for j, route in fr["plan"].items():
            pts = [fr["pos"][j]] + [fr["tasks"][i][0] for i in route if i in fr["tasks"]]
            if len(pts) > 1:
                pts = np.array(pts); ax.plot(pts[:, 0], pts[:, 1], "-", lw=0.8, color=ucol[j])
            mk = "o" if fr["state"][j] == "active" else "x"
            ax.scatter(*fr["pos"][j], c=[ucol[j]], marker=mk, s=70, **({"edgecolors": "k"} if mk == "o" else {}))
        ax.set_title(f"t={fr['t']}: {fr['event'][:45]}", fontsize=8)
        ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.set_aspect("equal"); ax.tick_params(labelsize=6)
    for ax in axs.flat[len(pick):]:
        ax.axis("off")
    fig.suptitle("Mo phong dong (tai lap Hinh 14). Sao do = loai A/tiep te, cam = B, xanh = C; o vuong = diem cuu tro; x = UAV dang ve/sac")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
