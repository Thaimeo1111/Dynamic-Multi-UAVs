"""T03 - Bo sinh risk map dong: luoi 2D, hotspot di chuyen theo thoi gian.

Y tuong
-------
* Moi hotspot la mot "o nguy hiem" (vung bao, doi chay, du chan, ...) co tam di chuyen
  theo quy dao tuyen tinh tung doan (waypoints), do rong sigma va cuong do amp(t).
* Rui ro tai (x, y, t) la hop "noisy-OR" cua cac hotspot (va lop tinh, neu co):
      R = 1 - prod_h (1 - r_h(x, y, t)),   r_h = amp_h(t) * exp(-d^2 / (2 sigma_h^2))
  nen R luon nam trong [0, 1].
* Hai goc nhin cung mot the gioi:
    - truth  (t_known = None): hotspot dung nhu that su.
    - belief (t_known = t)   : nhung gi bo dieu phoi BIET tai thoi diem t:
        hotspot chua xuat hien thi chua biet; vi tri tuong lai duoc NGOAI SUY tuyen tinh
        theo van toc hien tai (khong biet truoc cho re); t_end chua biet.
  Chenh lech truth - belief chinh la "bat ngo" ma bo dieu phoi dong phai xu ly.

Chay thu:  python -m src.risk_map --seed 3 --out figures
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


# ----------------------------------------------------------------- hotspot
@dataclass
class Hotspot:
    id: int
    sigma: float
    amp: float
    t_start: float                      # luc hotspot ton tai (= luc bo dieu phoi biet no)
    t_end: float                        # luc bien mat (bo dieu phoi chua biet truoc)
    waypoints: list                     # [(t, x, y), ...] tang dan theo t; doan cuoi keo dai mai
    ramp: float = 0.0                   # thoi gian de rui ro tang tu 40% len 100% (0 = co ngay)
    kind: str = "initial"               # initial | new

    def _wp(self):
        a = np.asarray(self.waypoints, float)
        return a[:, 0], a[:, 1], a[:, 2]

    def pos(self, t):
        """Vi tri that su tai thoi diem t (mang hoac so). Ngoai doan cuoi: tiep tuc van toc cuoi."""
        t = np.asarray(t, float)
        tw, xw, yw = self._wp()
        x = np.interp(t, tw, xw)
        y = np.interp(t, tw, yw)
        if len(tw) >= 2:
            vx = (xw[-1] - xw[-2]) / (tw[-1] - tw[-2])
            vy = (yw[-1] - yw[-2]) / (tw[-1] - tw[-2])
            over = np.maximum(t - tw[-1], 0.0)
            x = x + vx * over
            y = y + vy * over
        return x, y

    def vel_at(self, t):
        """Van toc cua doan chua t (lien tuc ben phai)."""
        tw, xw, yw = self._wp()
        if len(tw) < 2:
            return 0.0, 0.0
        k = int(np.clip(np.searchsorted(tw, t, side="right") - 1, 0, len(tw) - 2))
        dt = tw[k + 1] - tw[k]
        return (xw[k + 1] - xw[k]) / dt, (yw[k + 1] - yw[k]) / dt

    def amp_at(self, t):
        t = np.asarray(t, float)
        if self.ramp > 0:
            g = np.clip(0.4 + 0.6 * (t - self.t_start) / self.ramp, 0.4, 1.0)
        else:
            g = np.ones_like(t)
        return self.amp * g

    def to_dict(self):
        return dict(id=self.id, sigma=self.sigma, amp=self.amp, t_start=self.t_start, t_end=self.t_end,
                    waypoints=[list(map(float, w)) for w in self.waypoints], ramp=self.ramp, kind=self.kind)

    @staticmethod
    def from_dict(d):
        return Hotspot(**{**d, "waypoints": [tuple(w) for w in d["waypoints"]]})


# ----------------------------------------------------------------- field
class RiskField:
    """Truong rui ro R(x, y, t). t_known=None -> truth; so -> belief tai thoi diem do."""

    def __init__(self, hotspots, bbox=(0.0, 0.0, 100.0, 100.0), cell=2.0, static=None, t_known=None):
        self.all_hotspots = list(hotspots)
        self.bbox = bbox
        self.cell = float(cell)
        self.t_known = t_known
        self.static = static            # None hoac mang (H, W) cung luoi voi bbox/cell
        x0, y0, x1, y1 = bbox
        self.W = int(np.ceil((x1 - x0) / cell))
        self.H = int(np.ceil((y1 - y0) / cell))
        self.gx = x0 + (np.arange(self.W) + 0.5) * cell
        self.gy = y0 + (np.arange(self.H) + 0.5) * cell
        if t_known is None:
            self.hotspots = self.all_hotspots
        else:  # chi hotspot da xuat hien va chua bien mat
            self.hotspots = [h for h in self.all_hotspots if h.t_start <= t_known < h.t_end]

    # -- trang thai hotspot tai cac thoi diem
    def _state(self, h: Hotspot, t):
        t = np.asarray(t, float)
        if self.t_known is None:
            x, y = h.pos(t)
            a = h.amp_at(t) * ((t >= h.t_start) & (t < h.t_end))
            return x, y, a
        tk = self.t_known
        xk, yk = h.pos(tk)
        vx, vy = h.vel_at(tk)
        past = t <= tk
        xt, yt = h.pos(np.minimum(t, tk))
        x = np.where(past, xt, xk + vx * (t - tk))
        y = np.where(past, yt, yk + vy * (t - tk))
        a = h.amp_at(t) * (t >= h.t_start)      # t_end chua biet -> gia su ton tai mai
        return x, y, a

    # -- truong tren luoi cho nhieu thoi diem: (T, H, W)
    def frames(self, ts):
        ts = np.atleast_1d(np.asarray(ts, float))
        X, Y = np.meshgrid(self.gx, self.gy)
        keep = np.ones((len(ts), self.H, self.W))
        for h in self.hotspots:
            px, py, a = self._state(h, ts)
            d2 = (X[None] - px[:, None, None]) ** 2 + (Y[None] - py[:, None, None]) ** 2
            keep *= 1.0 - a[:, None, None] * np.exp(-d2 / (2.0 * h.sigma ** 2))
        if self.static is not None:
            keep *= (1.0 - self.static)[None]
        return 1.0 - keep

    def snapshot(self, t):
        return self.frames([t])[0]

    # -- truy van diem chinh xac (khong luong tu hoa): xy (M,2), t (M,)
    def risk_at(self, xy, t):
        xy = np.asarray(xy, float)
        t = np.broadcast_to(np.asarray(t, float), (len(xy),))
        keep = np.ones(len(xy))
        for h in self.hotspots:
            px, py, a = self._state(h, t)
            d2 = (xy[:, 0] - px) ** 2 + (xy[:, 1] - py) ** 2
            keep *= 1.0 - a * np.exp(-d2 / (2.0 * h.sigma ** 2))
        if self.static is not None:
            ix = np.clip(((xy[:, 0] - self.bbox[0]) // self.cell).astype(int), 0, self.W - 1)
            iy = np.clip(((xy[:, 1] - self.bbox[1]) // self.cell).astype(int), 0, self.H - 1)
            keep *= 1.0 - self.static[iy, ix]
        return 1.0 - keep

    def belief(self, t_known):
        return RiskField(self.all_hotspots, self.bbox, self.cell, self.static, t_known=t_known)

    def centers(self, t):
        return [(h.id, *[float(v) for v in self._state(h, t)[:2]], float(self._state(h, t)[2]))
                for h in self.hotspots]


# ----------------------------------------------------------------- sinh hotspot ngau nhien
def _fold(u, lo, hi):
    """Gap (phan xa) mot toa do vao [lo, hi]."""
    w = hi - lo
    u = (u - lo) % (2 * w)
    return lo + (u if u <= w else 2 * w - u)


def _keep_inside(p, v, duration, bbox, margin=5.0):
    """Chinh van toc sao cho diem cuoi (sau `duration`) nam trong mien; quy dao thang nen ca doan nam trong."""
    x0, y0, x1, y1 = bbox
    end = p + np.asarray(v, float) * duration
    end = np.array([_fold(end[0], x0 + margin, x1 - margin), _fold(end[1], y0 + margin, y1 - margin)])
    return (end - p) / duration


def generate_hotspots(rng: np.random.Generator, bbox, horizon, n_initial=3, n_new=1, n_turn=1,
                      anchors=None, sigma_range=(5.0, 8.0), amp_range=(0.85, 1.0)):
    """Sinh danh sach hotspot (co seed qua rng).

    n_initial : hotspot co tu t=0 (bo dieu phoi biet va ngoai suy tuyen tinh)
    n_turn    : so lan mot hotspot doi huong/toc do dot ngot (bat ngo ve quy dao)
    n_new     : so hotspot moi xuat hien giua chung (tam gan mot "anchor", vd. mot khach hang)
    """
    x0, y0, x1, y1 = bbox
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    spans = np.array([x1 - x0, y1 - y0])
    hs = []

    def rand_pos(margin=0.2):
        return np.array([rng.uniform(x0 + margin * spans[0], x1 - margin * spans[0]),
                         rng.uniform(y0 + margin * spans[1], y1 - margin * spans[1])])

    for i in range(n_initial):
        p = rand_pos()
        ang = rng.uniform(0, 2 * np.pi)
        sp = rng.uniform(0.04, 0.12)
        v = sp * np.array([np.cos(ang), np.sin(ang)])
        v = _keep_inside(p, v, horizon, bbox)
        hs.append(Hotspot(i, rng.uniform(*sigma_range), rng.uniform(*amp_range), 0.0, np.inf,
                          [(0.0, *p), (horizon, *(p + v * horizon))], 0.0, "initial"))
    # bat ngo ve quy dao: cat doan cuoi, them waypoint moi voi huong va toc do khac
    for _ in range(n_turn):
        if not hs:
            break
        h = hs[int(rng.integers(0, len(hs)))]
        t_turn = rng.uniform(0.2, 0.7) * horizon
        px, py = h.pos(t_turn)
        ang = rng.uniform(0, 2 * np.pi)
        sp = rng.uniform(0.08, 0.2)
        v = sp * np.array([np.cos(ang), np.sin(ang)])
        # huong ve phia tam neu sap ra ngoai mien
        v = _keep_inside(np.array([px, py]), v, horizon - t_turn, bbox)
        wp = [w for w in h.waypoints if w[0] < t_turn] + [(t_turn, px, py), (horizon, px + v[0] * (horizon - t_turn), py + v[1] * (horizon - t_turn))]
        h.waypoints = wp
    for j in range(n_new):
        t_s = rng.uniform(0.15, 0.65) * horizon
        if anchors is not None and len(anchors):
            a = np.asarray(anchors)[int(rng.integers(0, len(anchors)))]
            p = a + rng.normal(0, 4.0, 2)
        else:
            p = rand_pos(0.1)
        ang = rng.uniform(0, 2 * np.pi)
        sp = rng.uniform(0.03, 0.1)
        v = sp * np.array([np.cos(ang), np.sin(ang)])
        v = _keep_inside(p, v, 500.0, bbox)
        hs.append(Hotspot(n_initial + j, rng.uniform(*sigma_range), rng.uniform(*amp_range), t_s,
                          t_s + rng.uniform(250, 450),
                          [(t_s, *p), (t_s + 100.0, *(p + v * 100.0))], 150.0, "new"))
    return hs


# ----------------------------------------------------------------- ve hinh (3 thoi diem)
def plot_snapshots(field: RiskField, times, path, theta_ban=0.8, points=None, depot=None, title=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(times), figsize=(4.3 * len(times) + 0.8, 4.4), constrained_layout=True)
    axes = np.atleast_1d(axes)
    x0, y0, x1, y1 = field.bbox
    im = None
    for ax, t in zip(axes, times):
        R = field.snapshot(t)
        im = ax.imshow(R, origin="lower", extent=(x0, x1, y0, y1), vmin=0, vmax=1, cmap="YlOrRd")
        ax.contour(field.gx, field.gy, R, levels=[theta_ban], colors="k", linewidths=1.2)
        for _, hx, hy, a in field.centers(t):
            if x0 <= hx <= x1 and y0 <= hy <= y1:
                ax.plot(hx, hy, "k+", ms=9, mew=1.6)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        if points is not None:
            ax.scatter(points[:, 0], points[:, 1], s=9, c="#1f4e79", alpha=0.75, zorder=3)
        if depot is not None:
            ax.scatter([depot[0]], [depot[1]], marker="s", s=48, c="#1f4e79", edgecolors="w", zorder=4)
        ax.set_title(f"t = {t:.0f}", fontsize=11)
        ax.set_xlabel("x")
        ax.set_aspect("equal")
    axes[0].set_ylabel("y")
    cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
    cb.set_label("rui ro R(x, y, t)")
    if title:
        fig.suptitle(title, fontsize=11)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--horizon", type=float, default=1000.0)
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    hs = generate_hotspots(rng, (0, 0, 100, 100), a.horizon, n_initial=3, n_new=1, n_turn=1)
    fld = RiskField(hs)
    ts = [0.0, a.horizon * 0.4, a.horizon * 0.8]
    plot_snapshots(fld, ts, Path(a.out) / f"risk_map_seed{a.seed}.png",
                   title="Risk map dong: hotspot di chuyen, mot hotspot moi xuat hien (duong den: nguong cam bay)")
    print("saved", Path(a.out) / f"risk_map_seed{a.seed}.png")
