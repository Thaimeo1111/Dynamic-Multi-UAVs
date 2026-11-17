"""T28: kiem tra du lieu: so job ky vong vs co trong CSV, trung lap, thieu, loi, NaN; xuat tables/kiem_tra_du_lieu.md va danh sach chay lai."""
import os
import pandas as pd
from pathlib import Path
from src.experiments import load_config, build_jobs, job_key

cfg = load_config(); out = ["# Kiem tra du lieu thi nghiem\n", "| Thi nghiem | Ky vong | Co trong CSV | Thieu | Trung lap | Dong co NaN | File loi |", "|---|---|---|---|---|---|---|"]
rerun = []
for exp in ("main", "ablation", "large", "sensitivity"):
    f = f"results/raw_{exp}.csv"
    if not os.path.exists(f):
        out.append(f"| {exp} | - | chua chay | - | - | - | - |"); continue
    d = pd.read_csv(f)
    exp_keys = {job_key(j) for j in build_jobs(exp, cfg)}
    d["level"] = d["level"].astype(str); d["factor"] = d["factor"].fillna("").astype(str)
    got = [(r.exp, str(r.factor) if r.factor == r.factor else "", str(r.level) if str(r.level) != "nan" else "", r.instance, int(r.seed), r.algo, r.intensity) for r in d.itertuples()]
    norm = lambda k: (k[0], k[1] if k[1] != "nan" else "", k[2] if k[2] != "nan" else "", k[3], k[4], k[5], k[6])
    ek = {norm(tuple(map(lambda x: x if not isinstance(x, float) else str(x), k))) for k in exp_keys}
    gk = set(); dup = 0
    for k in got:
        k = norm(k)
        # so sanh level dang so: 50.0 vs 50
        gk.add((k[0], k[1], k[2].rstrip("0").rstrip(".") if "." in k[2] else k[2], *k[3:]))
    ek = {(k[0], k[1], k[2].rstrip("0").rstrip(".") if "." in k[2] else k[2], *k[3:]) for k in ek}
    dup = len(got) - len(set(got))
    missing = sorted(ek - gk)
    nan = int(d[["J", "J_ref", "dist", "risk", "late"]].isna().any(axis=1).sum())
    err = Path(f"results/raw_{exp}.csv.errors.jsonl")
    errn = sum(1 for _ in open(err)) if err.exists() else 0
    out.append(f"| {exp} | {len(ek)} | {len(gk)} | {len(missing)} | {dup} | {nan} | {errn} |")
    rerun += [f"{exp}: {m}" for m in missing]
out.append("\nDanh sach can chay lai: " + ("khong co." if not rerun else "\n" + "\n".join(f"- {r}" for r in rerun[:50])))
out.append("\nCach chay lai phan thieu: `./run_all_experiments.sh` (co --resume, chi chay job chua co).")
Path("tables/kiem_tra_du_lieu.md").write_text("\n".join(out) + "\n"); print("\n".join(out))
