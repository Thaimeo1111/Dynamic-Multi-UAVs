"""T08/T09 - Chay 3 baseline (greedy, ALNS tinh, re-solve) tren benchmark nho, do runtime, ghi CSV + bang.

  python run_small.py                       # mac dinh: 6 instance 25-50 khach x 5 seed
  python run_small.py --seeds 10
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.experiments import COLUMNS, ROOT, load_config, make_job, run_job  # noqa: E402

SMALL = ["data/solomon_small/C101_025.txt", "data/solomon_small/R101_025.txt", "data/solomon_small/RC101_025.txt",
         "data/solomon_small/C101_050.txt", "data/solomon_small/R101_050.txt", "data/solomon_small/RC101_050.txt"]
BASE = ["greedy", "alns_static", "resolve"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default="results/baseline_small.csv")
    ap.add_argument("--table", default="tables/baseline_small.md")
    a = ap.parse_args()
    cfg = load_config()
    rows = []
    for p in SMALL:
        for s in range(1, a.seeds + 1):
            for algo in BASE:
                rows.append(run_job(make_job("baseline_small", ROOT / p, s, algo, cfg["scenario"]["intensity"]), cfg))
        print("xong", p, flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    # bang: trung binh theo (instance, thuat toan): J, runtime
    lines = ["| Instance | Thuat toan | J (TB +- std) | Quang duong | Rui ro | Tre | Khong PV | Runtime ke hoach (s) | Wall (s) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for p in SMALL:
        name = Path(p).stem
        for algo in BASE:
            r = [x for x in rows if x["instance"] == name and x["algo"] == algo]
            g = lambda k: np.mean([x[k] for x in r])
            lines.append(f"| {name} | {algo} | {g('J'):.0f} +- {np.std([x['J'] for x in r], ddof=1):.0f} | {g('dist'):.0f} | "
                         f"{g('risk'):.0f} | {g('late'):.0f} | {g('unserved'):.1f} | {g('plan_time'):.2f} | {g('wall'):.2f} |")
    Path(a.table).parent.mkdir(parents=True, exist_ok=True)
    Path(a.table).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
