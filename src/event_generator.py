"""T05 - Bo sinh su kien dong (co seed co dinh) va ghi kich ban ra JSON.

Mot KICH BAN = the gioi "that" ma cac thuat toan se gap, gom:
  * khach xuat hien dan (release time), chi so du window-aware: xuat hien du som de con kha thi
  * risk map doi: hotspot moi xuat hien, hotspot re huong dot ngot (xem risk_map.py)
  * deadline doi giua chung (siet lai hoac noi ra)
  * pin giam bat thuong cua mot UAV (mat mot ty le nang luong con lai)
Cung (instance, seed, intensity) -> cung kich ban cho moi thuat toan (so sanh cong bang).

Su kien (events) la nhung gi bo dieu phoi BIET khi chung xay ra. Tat ca deu co thoi diem `t`.
"""
from __future__ import annotations

import json
import math
import zlib
from pathlib import Path

import numpy as np

from .risk_map import generate_hotspots

INTENSITY = {            # (hotspot moi, hotspot re huong, doi deadline, pin tut, hotspot ban dau)
    "low": (1, 1, 2, 1, 2),
    "med": (2, 2, 4, 2, 3),
    "high": (3, 3, 8, 3, 3),
}


def n_uav(inst, factor=1.2):
    return int(math.ceil(factor * inst.dem.sum() / inst.Q)) + 1


def make_scenario(inst, seed, intensity="med", battery=300.0, online_frac=0.3, K=None, bbox=None):
    rng = np.random.default_rng((zlib.crc32(inst.name.encode()) % 100003) * 1000 + int(seed))
    n, H = inst.n, inst.horizon
    if bbox is None:
        top = float(np.ceil(inst.xy.max() / 10.0) * 10.0)
        bbox = (0.0, 0.0, max(100.0, top), max(100.0, top))
    n_new, n_turn, n_dl, n_bat, n0 = INTENSITY[intensity]
    d0 = np.hypot(*(inst.xy - inst.xy[0]).T)

    # --- khach xuat hien dan
    release = np.zeros(n + 1)
    online = rng.choice(np.arange(1, n + 1), size=int(round(online_frac * n)), replace=False)
    latest = np.maximum(0.0, inst.due - d0 - inst.serv - 20.0)
    release[online] = np.round(rng.uniform(0, 1, len(online)) * np.minimum(latest[online], 0.6 * H), 1)

    # --- risk map
    hs = generate_hotspots(rng, bbox, H, n_initial=n0, n_new=n_new, n_turn=n_turn, anchors=inst.xy[1:])
    events = []
    for c in online:
        if release[c] > 0:
            events.append(dict(t=float(release[c]), type="new_target", cust=int(c)))
    for h in hs:
        if h.kind == "new":
            events.append(dict(t=float(h.t_start), type="risk_change", kind="new", hotspot=h.id))
        for w in h.waypoints[1:-1]:
            events.append(dict(t=float(w[0]), type="risk_change", kind="turn", hotspot=h.id))

    # --- deadline doi
    chosen = rng.choice(np.arange(1, n + 1), size=min(n_dl, n), replace=False)
    for c in chosen:
        te = float(np.round(rng.uniform(0.15, 0.65) * H, 1))
        due = float(inst.due[c])
        if due - te < 60:
            continue
        if rng.random() < 0.7:
            new = max(te + 15.0, due - rng.uniform(0.2, 0.5) * (due - te))
        else:
            new = due + rng.uniform(50, 150)
        events.append(dict(t=te, type="deadline_change", cust=int(c), new_due=float(np.round(new, 1))))

    # --- pin giam bat thuong
    K = K or n_uav(inst)
    for _ in range(n_bat):
        events.append(dict(t=float(np.round(rng.uniform(0.2, 0.6) * H, 1)), type="battery_drop",
                           uav=int(rng.integers(0, K)), frac=float(np.round(rng.uniform(0.15, 0.4), 2))))
    events.sort(key=lambda e: (e["t"], e["type"]))
    return dict(instance=inst.name, seed=int(seed), intensity=intensity, bbox=list(bbox),
                uav=dict(K=int(K), battery=float(battery), Q=float(inst.Q)),
                release=release.tolist(), hotspots=[h.to_dict() for h in hs], events=events)


def save_scenario(scn, directory="scenarios"):
    path = Path(directory) / f"{scn['instance']}_s{scn['seed']}_{scn['intensity']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(scn, indent=1))
    return path


def load_scenario(path):
    return json.loads(Path(path).read_text())


if __name__ == "__main__":
    import sys
    from .instance_loader import load_instance
    for p in sys.argv[1:]:
        inst = load_instance(p)
        print(save_scenario(make_scenario(inst, 1), "scenarios"))
