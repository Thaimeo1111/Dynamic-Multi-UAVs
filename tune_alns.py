"""T23 - Tinh chinh ALNS cho instance 100-200 khach: do hoi tu theo so vong va thoi gian moi vong, de chon
so vong (va time limit) hop ly. Giai bai toan TINH luc t=0 (toan bo khach, belief luc 0), nhieu seed.

  python tune_alns.py            -> docs/tuning_alns.md, figures/tuning_alns.png, results/tuning_alns.csv
"""
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.alns_dynamic import FULL_CFG, solve  # noqa: E402
from src.event_generator import make_scenario  # noqa: E402
from src.experiments import ROOT, load_config, params_from  # noqa: E402
from src.instance_loader import load_instance, make_synthetic  # noqa: E402
from src.rolling_horizon import Sim, build_problem  # noqa: E402

CASES = [("data/solomon_small/C101_025.txt", None), ("data/solomon_small/RC101_050.txt", None),
         ("data/solomon_100/C101_100.txt", None), ("data/solomon_100/R101_100.txt", None),
         ("data/homberger_200/R1_2_1_200.txt", None), ("data/homberger_200/RC1_2_1_200.txt", None)]
# Du lieu that (Solomon goc, Gehring-Homberger); ban cu dung SYN_* va instance 200 tu sinh.
ITERS, SEEDS = 1000, 4


def main():
    cfg = load_config()
    P = params_from(cfg)
    rows, curves = [], {}
    for path, syn in CASES:
        inst = load_instance(ROOT / path) if path else make_synthetic(syn[0], syn[1], 900 + syn[1] + len(syn[0]))
        scn = make_scenario(inst, 1, "med", battery=P["battery"])
        pb = build_problem(Sim(inst, scn, P), 0.0, include_all=True)
        traces, times = [], []
        for s in range(SEEDS):
            t0 = time.perf_counter()
            sol, info = solve(pb, None, ITERS, np.random.default_rng(100 + s), FULL_CFG, record_trace=True)
            times.append(time.perf_counter() - t0)
            tr = dict(info["trace"])
            traces.append([tr.get(i, np.nan) for i in range(0, ITERS + 1, 10)])
        tr = np.nanmean(np.array(traces), axis=0)
        curves[inst.name] = tr
        sec_per_iter = float(np.mean(times) / ITERS)
        final = tr[-1]
        for frac in (0.1, 0.25, 0.5, 1.0):
            it = int(ITERS * frac)
            rows.append(dict(instance=inst.name, n=inst.n, iters=it, mean_cost=round(float(tr[it // 10]), 1),
                             gap_to_1000_pct=round(100 * (tr[it // 10] - final) / final, 2),
                             est_seconds=round(sec_per_iter * it, 2)))
        print(inst.name, "s/iter=%.4f" % sec_per_iter, "build+solve=%.1fs" % np.mean(times), flush=True)
    Path("results").mkdir(exist_ok=True)
    with open("results/tuning_alns.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.2), constrained_layout=True)
    for name, tr in curves.items():
        ax.plot(np.arange(0, ITERS + 1, 10), tr / tr[-1], label=name)
    ax.set_xlabel("so vong ALNS")
    ax.set_ylabel("chi phi tot nhat / chi phi sau 1000 vong")
    ax.set_ylim(0.95, 1.6)
    ax.axvline(150, c="gray", ls=":")
    ax.axvline(300, c="gray", ls=":")
    ax.legend(fontsize=8)
    Path("figures").mkdir(exist_ok=True)
    fig.savefig("figures/tuning_alns.png", dpi=150)
    md = ["# Tinh chinh ALNS (T23)", "",
          "Giai bai toan tinh luc t = 0, toan bo khach, 4 seed, toan tu day du (FULL_CFG). Chi phi la trung binh cua "
          "chi phi tot nhat tai moi moc vong lap.", "",
          "| Instance | n | Vong | Chi phi TB | Cach chi phi sau 1000 vong (%) | Thoi gian uoc tinh (s) |", "|---|---|---|---|---|---|"]
    md += [f"| {r['instance']} | {r['n']} | {r['iters']} | {r['mean_cost']} | {r['gap_to_1000_pct']} | {r['est_seconds']} |" for r in rows]
    md += ["", "![hoi tu](../figures/tuning_alns.png)"]
    Path("docs/tuning_alns.md").write_text("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
