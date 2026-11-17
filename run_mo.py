"""T21: so sanh tinh da muc tieu NSGA-III vs ALNS (archive) cung ngan sach danh gia, do HV/IGD.
Bai toan: thoi diem t=0, mot kich ban, tat ca khach da biet (include_all), belief tai t=0."""
import argparse, csv, time
import numpy as np
from src.experiments import load_config, params_from
from src.instance_loader import load_instance
from src.event_generator import make_scenario
from src.rolling_horizon import Sim, build_problem
from src.alns_dynamic import solve, FULL_CFG as _F
FULL_CFG = dict(_F, archive=True)
from src.nsga3 import nsga3, pareto_filter, hypervolume, igd, objective_of

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--pop", type=int, default=30)
ap.add_argument("--gens", type=int, default=30); ap.add_argument("--out", default="results/mo_static.csv")
ap.add_argument("--instances", nargs="*", default=["data/solomon_small/C101_025.xml", "data/solomon_small/SYN_RC101_025.txt",
                                                   "data/solomon_small/C101_050.xml"])
a = ap.parse_args()
cfg = load_config(); P = params_from(cfg, {})
rows = []
for path in a.instances:
    inst = load_instance(path)
    for seed in range(a.seeds):
        scn = make_scenario(inst, seed, "med", battery=P["battery"], online_frac=cfg["scenario"]["online_frac"])
        sim = Sim(inst, scn, P); pb = build_problem(sim, 0.0, include_all=True)
        budget = a.pop * (a.gens + 1)
        t0 = time.perf_counter()
        _, info = solve(pb, None, iters=budget, rng=np.random.default_rng(seed), cfg=FULL_CFG)
        t_alns = time.perf_counter() - t0
        FA = pareto_filter(np.array([list(v) for v in info["archive"]]))
        t0 = time.perf_counter()
        FN, ev = nsga3(pb, a.pop, a.gens, rng=np.random.default_rng(seed))
        t_ns = time.perf_counter() - t0
        allF = pareto_filter(np.vstack([FA, FN])); ref = np.vstack([FA, FN]).max(axis=0) * 1.1 + 1e-6
        for name, F, t in (("alns_archive", FA, t_alns), ("nsga3", FN, t_ns)):
            rows.append(dict(instance=inst.name, seed=seed, algo=name, evals=budget, n_front=len(F),
                             hv=hypervolume(F, ref), hv_ref=hypervolume(allF, ref), igd=igd(F, allF), time=round(t, 2)))
        print(inst.name, seed, f"HV a={rows[-2]['hv']:.0f} n={rows[-1]['hv']:.0f} of {rows[-1]['hv_ref']:.0f}", flush=True)
with open(a.out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
