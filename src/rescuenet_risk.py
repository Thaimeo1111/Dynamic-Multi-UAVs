"""Mat na phan doan (kieu RescueNet) -> lop rui ro tinh tren luoi risk map (T19/T22).

Chi so lop theo docs/rescuenet_classes.md. Bang trong so RISK_W la DE XUAT cua nguoi lap
tai lieu (khong phai tu nguon); sua trong config neu muon. Khong co du lieu RescueNet that
trong moi truong nay: `synthetic_mask` sinh mat na GIA, chi de chay thu pipeline.

Dung voi du lieu that:  python -m src.rescuenet_risk --mask path/mask.png --out figures/x.png
"""
import argparse
import numpy as np

RISK_W = {0: 0.0, 1: 0.35, 2: 0.0, 3: 0.15, 4: 0.6, 5: 0.9, 6: 0.0, 7: 0.0, 8: 0.7, 9: 0.05, 10: 0.0}


def mask_to_grid(mask, bbox=(0.0, 0.0, 100.0, 100.0), cell=2.0, weights=None, smooth=1):
    """mask (h,w) so nguyen lop -> mang (H,W) trong [0,1], trung binh khoi theo o luoi."""
    w = np.zeros(11)
    for k, v in (weights or RISK_W).items():
        w[k] = v
    r = w[np.clip(mask.astype(int), 0, 10)]
    W = int(np.ceil((bbox[2] - bbox[0]) / cell)); H = int(np.ceil((bbox[3] - bbox[1]) / cell))
    ys = np.linspace(0, r.shape[0], H + 1).astype(int); xs = np.linspace(0, r.shape[1], W + 1).astype(int)
    g = np.array([[r[ys[i]:ys[i + 1], xs[j]:xs[j + 1]].mean() for j in range(W)] for i in range(H)])
    for _ in range(smooth):  # lam tron nhe de tranh vung cam 1 o
        p = np.pad(g, 1, mode="edge")
        g = (p[1:-1, 1:-1] * 4 + p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]) / 8
    return np.clip(g, 0, 1)


def synthetic_mask(seed, size=200, n_blobs=9):
    """Mat na GIA: nen 0, vai vung nuoc (1), toa nha hu hong (4/5) va duong bi chan (8)."""
    rng = np.random.default_rng(seed)
    m = np.zeros((size, size), int)
    yy, xx = np.mgrid[:size, :size]
    for _ in range(n_blobs):
        cx, cy = rng.uniform(0.1, 0.9, 2) * size
        rx, ry = rng.uniform(0.04, 0.12, 2) * size
        c = int(rng.choice([1, 4, 5, 8, 3, 2], p=[.2, .2, .2, .15, .1, .15]))
        m[((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 < 1] = c
    for _ in range(2):  # duong bi chan: doan thang day
        x0, y0, x1, y1 = rng.uniform(0.1, 0.9, 4) * size
        for t in np.linspace(0, 1, 400):
            x, y = int(x0 + t * (x1 - x0)), int(y0 + t * (y1 - y0))
            m[max(0, y - 2):y + 3, max(0, x - 2):x + 3] = 8
    return m


def load_mask(path):
    from PIL import Image
    a = np.array(Image.open(path))
    return a[..., 0] if a.ndim == 3 else a


def make_field(seed=0, mask=None, with_hotspots=True, horizon=1000.0, bbox=(0.0, 0.0, 100.0, 100.0)):
    """RiskField = lop tinh tu mat na (+ hotspot dong tuy chon)."""
    from .risk_map import RiskField, generate_hotspots
    mask = synthetic_mask(seed) if mask is None else mask
    st = mask_to_grid(mask, bbox)
    hs = generate_hotspots(np.random.default_rng(seed), bbox, horizon) if with_hotspots else []
    return RiskField(hs, bbox=bbox, static=st), mask, st


if __name__ == "__main__":
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask"); ap.add_argument("--seeds", type=int, nargs="*", default=[1, 2, 3])
    ap.add_argument("--out", default="figures/rescuenet_scenarios.png")
    a = ap.parse_args()
    items = [(load_mask(a.mask), 0)] if a.mask else [(None, s) for s in a.seeds]
    fig, ax = plt.subplots(2, len(items), figsize=(4.2 * len(items), 8), squeeze=False)
    for j, (m, s) in enumerate(items):
        f, m, st = make_field(s, m)
        ax[0, j].imshow(m, cmap="tab10", vmin=0, vmax=10); ax[0, j].set_title(f"mat na {'GIA ' if not a.mask else ''}(seed {s})")
        im = ax[1, j].imshow(f.snapshot(0.0), origin="lower", extent=(0, 100, 0, 100), vmin=0, vmax=1, cmap="YlOrRd")
        ax[1, j].set_title("risk map t=0 (tinh + dong)")
    fig.colorbar(im, ax=ax[1, :], shrink=.7); fig.savefig(a.out, dpi=130, bbox_inches="tight")
    print("saved", a.out)
