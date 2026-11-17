"""T19/T22: kich ban risk map tu nhan RescueNet THAT + so sanh 4 thuat toan tren cac kich ban do.
Chay: python3 run_rescuenet.py <thu_muc_rescuenet_labels> [--n-masks 5]
Luu y: mat na 3000x4000 duoc co gian ve mien vuong 100x100 (bien dang ti le); trong so lop RISK_W la de xuat, chua co nguon."""
import argparse, glob, json, csv
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.rescuenet_risk import mask_to_grid, RISK_W
from src.instance_loader import load_instance
from src.event_generator import make_scenario, save_scenario
from src.experiments import load_config, params_from, algo_options
from src.algorithms import make_planner
from src.rolling_horizon import run_episode, Sim
from src.risk_map import RiskField, Hotspot

ap = argparse.ArgumentParser(); ap.add_argument("labels"); ap.add_argument("--n-masks", type=int, default=5)
ap.add_argument("--inst", default="data/solomon_small/RC101_050.txt"); ap.add_argument("--seeds", type=int, default=3)
a = ap.parse_args()
cfg = load_config(); P = params_from(cfg, {}); inst = load_instance(a.inst)
files = sorted(glob.glob(f"{a.labels}/test-label-img/*.png"))
cand = []
for f in files:
    m = np.array(Image.open(f).resize((400, 300), Image.NEAREST)); c = np.bincount(m.ravel(), minlength=11) / m.size
    if c[1] < 0.6:
        cand.append((c[[4, 5, 8]].sum(), f))
cand.sort(reverse=True); picks = [f for _, f in cand[:a.n_masks]]
Path("scenarios/rescuenet").mkdir(parents=True, exist_ok=True)
rows = []; fig, ax = plt.subplots(2, len(picks), figsize=(3.6 * len(picks), 7), squeeze=False)
for j, f in enumerate(picks):
    mask = np.array(Image.open(f)); name = Path(f).stem.replace("_lab", "")
    grid = mask_to_grid(mask, (0, 0, 100, 100), P["cell"])
    ax[0, j].imshow(np.array(Image.fromarray(mask).resize((200, 150), Image.NEAREST)), cmap="tab10", vmin=0, vmax=10)
    ax[0, j].set_title(f"RescueNet {name}", fontsize=9); ax[0, j].axis("off")
    ax[1, j].imshow(grid, origin="lower", extent=(0, 100, 0, 100), vmin=0, vmax=1, cmap="YlOrRd")
    ax[1, j].contour(np.linspace(1, 99, grid.shape[1]), np.linspace(1, 99, grid.shape[0]), grid, levels=[P["theta_ban"]], colors="k", linewidths=.8)
    ax[1, j].plot(*inst.xy[0], "b*", ms=10); ax[1, j].set_title(f"lop rui ro tinh (den = cam), cam {100*(grid>P['theta_ban']).mean():.1f}%", fontsize=8)
    for s in range(1, a.seeds + 1):
        scn = make_scenario(inst, s, "med", battery=P["battery"], online_frac=cfg["scenario"]["online_frac"])
        scn["static_risk"] = grid.tolist(); scn["rescuenet_mask"] = name
        save_scenario(scn, f"scenarios/rescuenet/{inst.name}_{name}_s{s}.json")
        for alg in ("greedy", "alns_static", "resolve", "alns_dynamic"):
            m = run_episode(inst, scn, P, make_planner(alg, inst.n, s, algo_options(cfg, inst.n)))
            d = P  # J_ref chuan
            J = d["w_d"] * m["dist"] + d["w_r"] * m["risk"] + d["w_l"] * m["late"] + d["w_u"] * m["unserved"]
            rows.append(dict(mask=name, seed=s, algo=alg, J=round(J, 1), dist=round(m["dist"], 1), risk=round(m["risk"], 1),
                             late=round(m["late"], 1), unserved=m["unserved"], ban_dist=round(m["ban_dist"], 2), n_lost=m["n_lost"]))
fig.tight_layout(); fig.savefig("figures/rescuenet_real_scenarios.png", dpi=140)
with open("results/rescuenet_scenarios.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
import pandas as pd
t = pd.DataFrame(rows).groupby("algo")[["J", "dist", "risk", "late", "unserved", "ban_dist", "n_lost"]].mean().round(1)
t.to_markdown("tables/rescuenet_scenarios.md"); print(t); print(picks)
