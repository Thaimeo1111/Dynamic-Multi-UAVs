"""Chay lai cac thi nghiem cua Pu et al. (2026).

Vi du:
  py -3.11 -m paper_repro.run_paper_repro --exp all --quick
  py -3.11 -m paper_repro.run_paper_repro --exp single multi --full --workers 7

Ket qua: results/paper_repro/<mode>/*.csv ; bang + hinh tao boi paper_repro.report.
CSV ghi tang dan va co the chay tiep (bo qua job da xong).
"""
from __future__ import annotations

import argparse
import csv
import os
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
U = 5

QUICK = dict(sizes=[20, 100, 400], runs=3, iters=50, aco_iters=50, episodes=100, table2_runs=3, som_it=50)
FULL = dict(sizes=[20, 60, 100, 150, 200, 250, 300, 350, 400], runs=10, iters=200, aco_iters=200,
            episodes=200, table2_runs=10, som_it=100)


def out_dir(mode):
    d = ROOT / "results" / "paper_repro" / mode
    d.mkdir(parents=True, exist_ok=True)
    return d


# ------------------------------------------------------------------ Bang 1 va Bang 2
def exp_table1(cfg, mode):
    from .common import make_scenario
    from .som3d import som_allocate
    rows = []
    for seed in range(cfg["table2_runs"]):
        sc = make_scenario(100, True, 500 + seed)
        for alpha in (0.0, 3.0):
            a = som_allocate(sc, U, alpha=alpha, rng=np.random.default_rng(seed), it_max=cfg["som_it"])
            for j in range(U):
                cnt = [int(np.sum((a == j) & (sc["types"] == t))) for t in range(3)]
                rows.append({"seed": seed, "alpha": alpha, "uav": j + 1, "A": cnt[0], "B": cnt[1], "C": cnt[2]})
    _write(out_dir(mode) / "table1_som_alpha.csv", rows)


def exp_table2(cfg, mode):
    from .auction import auction
    from .common import evaluate, make_scenario
    from .som3d import som_allocate
    rows = []
    for n in cfg["sizes"]:
        for r in range(cfg["table2_runs"]):
            sc = make_scenario(n, False, 7000 + 31 * n + r)
            rng = np.random.default_rng(r)
            a0 = som_allocate(sc, U, alpha=0.0, rng=rng, it_max=cfg["som_it"])
            t = time.perf_counter()
            a1, hist = auction(sc, a0, U)
            dt = time.perf_counter() - t
            e0, e1 = evaluate(sc, a0, U), evaluate(sc, a1, U)
            rows.append({"n": n, "run": r, "P0": e0["P"], "D0": e0["D"], "C0": e0["cost"],
                         "P1": e1["P"], "D1": e1["D"], "C1": e1["cost"], "rounds": len(hist) - 1, "time": dt})
        print(f"[table2] n={n} xong", flush=True)
    _write(out_dir(mode) / "table2_auction.csv", rows)


# ------------------------------------------------------------------ Bang 4, 5
def _alloc_job(args):
    kind, n, run, algo, cfg = args
    from .baselines import ALGOS, ours
    from .common import evaluate, make_scenario
    multi = kind == "multi"
    sc = make_scenario(n, multi, 10_000 * (2 if multi else 1) + 101 * n + run)
    rng = np.random.default_rng(zlib.crc32(f"{kind}|{n}|{run}|{algo}".encode()))
    kw = {}
    if algo in ("GA", "PSO", "GAPSO", "NSGA2"):
        kw = dict(iters=cfg["iters"])
    elif algo == "ACO":
        kw = dict(iters=cfg["aco_iters"])
    elif algo == "DQN":
        kw = dict(episodes=cfg["episodes"])
    elif algo in ("3D-SOM", "OUR"):
        kw = dict(alpha=3.0 if multi else 0.0)
    t = time.perf_counter()
    if algo == "OUR-a0":
        a = ours(sc, U, rng, alpha=0.0)
    else:
        a = ALGOS[algo](sc, U, rng, **kw)
    dt = time.perf_counter() - t
    e = evaluate(sc, a, U)
    return {"kind": kind, "n": n, "run": run, "algo": algo, "cost": e["cost"], "P": e["P"],
            "D": e["D"], "R": e["R"], "time": dt}


def exp_alloc(kind, cfg, mode, workers):
    from .baselines import ALGOS, DETERMINISTIC
    path = out_dir(mode) / f"alloc_{kind}.csv"
    done = set()
    if path.exists():
        with open(path) as f:
            done = {(r["algo"], int(r["n"]), int(r["run"])) for r in csv.DictReader(f)}
    algos = list(ALGOS) + (["OUR-a0"] if kind == "multi" else [])
    jobs = []
    # job nang (n lon, metaheuristic) xep truoc de can tai
    for n in sorted(cfg["sizes"], reverse=True):
        for algo in algos:
            runs = 1 if algo in DETERMINISTIC else cfg["runs"]
            for r in range(runs):
                if (algo, n, r) not in done:
                    jobs.append((kind, n, r, algo, cfg))
    print(f"[{kind}] {len(jobs)} job (da co {len(done)})", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_alloc_job, j) for j in jobs]
        for k, fu in enumerate(as_completed(futs), 1):
            row = fu.result()
            _append(path, row)
            if k % 10 == 0 or k == len(jobs):
                print(f"[{kind}] {k}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)


# ------------------------------------------------------------------ Bang 7
def exp_forecast(cfg, mode, quick):
    from .forecast import run_all
    rows, preds, D = run_all(seed=0, quick=quick)
    _write(out_dir(mode) / "forecast.csv", rows)
    np.savez(out_dir(mode) / "forecast_preds.npz", data=D,
             **{f"p{p}_{k}": v for p, d in preds.items() for k, v in d.items()})


# ------------------------------------------------------------------ Hinh 14
def exp_dynamic(cfg, mode):
    from .dynamic_sim import simulate
    fig = ROOT / "figures" / "paper_repro" / mode
    fig.mkdir(parents=True, exist_ok=True)
    log = simulate(seed=1, fig_path=fig / "dynamic_frames.png", quick=mode == "quick")
    _write(out_dir(mode) / "dynamic_events.csv", log)


# ------------------------------------------------------------------ io
def _write(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def _append(path, row):
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", nargs="+", default=["all"],
                    choices=["all", "table1", "table2", "single", "multi", "forecast", "dynamic"])
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--quick", action="store_true")
    g.add_argument("--full", action="store_true")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    quick = not a.full
    cfg, mode = (QUICK, "quick") if quick else (FULL, "full")
    exps = ["table1", "table2", "forecast", "dynamic", "single", "multi"] if "all" in a.exp else a.exp
    for e in exps:
        t = time.time()
        print(f"=== {e} ({mode}) ===", flush=True)
        if e == "table1":
            exp_table1(cfg, mode)
        elif e == "table2":
            exp_table2(cfg, mode)
        elif e in ("single", "multi"):
            exp_alloc(e, cfg, mode, a.workers)
        elif e == "forecast":
            exp_forecast(cfg, mode, quick)
        elif e == "dynamic":
            exp_dynamic(cfg, mode)
        print(f"=== {e} xong sau {time.time() - t:.0f}s ===", flush=True)
    from .report import make_all
    make_all(mode)


if __name__ == "__main__":
    main()
