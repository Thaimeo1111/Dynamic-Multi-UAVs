"""T24 - Chay hang loat (multiprocessing), co dinh seed, ghi CSV.

Vi du:
  python run_batch.py --exp main --out results/raw_main.csv --workers 2
  python run_batch.py --exp ablation --out results/raw_ablation.csv
  python run_batch.py --exp sensitivity --out results/raw_sensitivity.csv
  python run_batch.py --exp main --seeds 3 --out results/smoke.csv      # chay thu nho
  python run_batch.py --exp main --resume --out results/raw_main.csv    # chay tiep neu bi ngat

Dinh dang CSV: moi dong = mot lan chay (exp, factor, level, instance, family, n, seed, intensity, algo) cung cac chi so
(J, J_ref, dist, risk, late, n_late, unserved, served, ban_dist, n_lost, replans, plan_time, reassign, wall).
Ket qua KHONG phu thuoc thu tu chay hay so worker: moi lan chay co seed rieng (seed kich ban + seed thuat toan).
"""
import argparse
import csv
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.experiments import COLUMNS, build_jobs, job_key, load_config, run_job  # noqa: E402

_CFG = None


def _work(job):
    global _CFG
    _CFG = _CFG or load_config()
    try:
        return job, run_job(job, _CFG), None
    except Exception as exc:                      # mot lan chay hong khong duoc lam sap ca lo
        import traceback
        return job, None, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-600:]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, choices=["main", "ablation", "sensitivity", "large", "sensitivity_large"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--seeds", type=int, default=None, help="ghi de so seed (chay thu)")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    jobs = build_jobs(a.exp, cfg, a.seeds)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if a.resume and out.exists():
        for r in csv.DictReader(open(out)):
            done.add((r["exp"], r["factor"], r["level"], r["instance"], int(r["seed"]), r["algo"], r["intensity"]))
    todo = [j for j in jobs if job_key(j) not in done]
    print(f"{len(jobs)} job, {len(done)} da xong, chay {len(todo)} job tren {a.workers} worker", flush=True)
    new = not (a.resume and out.exists())
    t0 = time.time()
    with open(out, "a" if not new else "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        with mp.Pool(a.workers) as pool:
            for i, (job, row, err) in enumerate(pool.imap_unordered(_work, todo, chunksize=1), 1):
                if err:
                    import json
                    with open(str(out) + ".errors.jsonl", "a") as ef:
                        ef.write(json.dumps(dict(job=job, error=err)) + "\n")
                else:
                    w.writerow(row)
                    f.flush()
                if i % 50 == 0 or i == len(todo):
                    el = time.time() - t0
                    print(f"  {i}/{len(todo)}  {el:6.0f}s  ~con {el / i * (len(todo) - i):5.0f}s", flush=True)
    print("xong ->", out, flush=True)


if __name__ == "__main__":
    main()
