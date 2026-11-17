"""T33: duong hoi tu ALNS (chi phi tot nhat theo so vong) tai t=0, cau hinh day du vs ban co so, tren instance that."""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.experiments import load_config, params_from
from src.instance_loader import load_instance
from src.event_generator import make_scenario
from src.rolling_horizon import Sim, build_problem
from src.alns_dynamic import solve, FULL_CFG, BASE_CFG

cfg = load_config(); P = params_from(cfg, {})
insts = ["data/solomon_small/C101_050.txt", "data/solomon_small/R101_050.txt", "data/solomon_100/RC101_100.txt"]
ITERS = 1000
fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
for a, path in zip(ax, insts):
    inst = load_instance(path)
    for name, c, col in (("ALNS day du (adaptive, 4 destroy, 3 repair)", FULL_CFG, "#2a9d5c"), ("ALNS co so (ngau nhien)", BASE_CFG, "#d98b2b")):
        curves = []
        for s in range(1, 6):
            scn = make_scenario(inst, s, "med", battery=P["battery"], online_frac=cfg["scenario"]["online_frac"])
            pb = build_problem(Sim(inst, scn, P), 0.0, include_all=True)
            _, info = solve(pb, None, iters=ITERS, rng=np.random.default_rng(s), cfg=dict(c), record_trace=True)
            x, y = zip(*info["trace"]); curves.append(np.interp(np.arange(0, ITERS + 1, 10), x, y))
        C = np.array(curves); C = C / C[:, :1]          # chuan hoa theo chi phi ban dau
        xs = np.arange(0, ITERS + 1, 10)
        a.plot(xs, C.mean(0), color=col, label=name); a.fill_between(xs, C.min(0), C.max(0), color=col, alpha=.18)
    a.set_title(inst.name if hasattr(inst, "name") else path); a.set_xlabel("so vong ALNS"); a.set_ylabel("chi phi / chi phi ban dau")
ax[0].legend(fontsize=7); fig.tight_layout(); fig.savefig("figures/convergence.png", dpi=150); print("ok")
