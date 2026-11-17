"""Chay lai Bang 7 tren nhieu bo du lieu (seed du lieu khac nhau) de xem ket qua co on dinh khong.

Seed 0 da co trong results/paper_repro/<mode>/forecast.csv (do run_paper_repro --exp forecast sinh ra).
Script nay chay them cac seed khac va ghi vao forecast_seeds.csv (cot data_seed).
Ghi lai sau moi seed, nen neu bi dung giua chung thi chay lai se bo qua seed da xong.

Vi du:
  py -3.11 -m paper_repro.forecast_seeds --seeds 1 2 3 4          # du lieu day du (~8 phut / seed)
  py -3.11 -m paper_repro.forecast_seeds --seeds 1 --quick        # thu nhanh, ghi vao thu muc quick
Sau do: py -3.11 -m paper_repro.report full  -> tables/paper_repro/full/table7_seeds.md
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from .forecast import run_all

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    mode = "quick" if a.quick else "full"
    d = ROOT / "results" / "paper_repro" / mode
    d.mkdir(parents=True, exist_ok=True)
    path = d / "forecast_seeds.csv"

    rows = []
    if path.exists():
        with open(path) as f:
            rows = list(csv.DictReader(f))
    done = {int(r["data_seed"]) for r in rows}
    for ds in a.seeds:
        if ds in done:
            print(f"seed {ds} da co, bo qua", flush=True)
            continue
        t = time.time()
        r, _, _ = run_all(seed=ds, quick=a.quick, data_seed=ds)
        for x in r:
            x["data_seed"] = ds
        rows += r
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"seed {ds} xong sau {time.time() - t:.0f}s", flush=True)
    print(f"Ket qua: {path}")


if __name__ == "__main__":
    main()
