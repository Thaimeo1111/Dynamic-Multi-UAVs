"""Chuan bi du lieu that: Solomon goc (cat n khach dau tien cho 25/50) va Gehring-Homberger 200.
Nguon: PyVRP/Instances (xem README trong goi du lieu). Chay: python3 prepare_real_data.py <thu_muc_uav_datasets>"""
import sys, shutil
from pathlib import Path
import numpy as np
from src.instance_loader import load_instance, save_solomon_txt, Instance

src = Path(sys.argv[1]); root = Path(__file__).parent / "data"
small, big, large = root / "solomon_small", root / "solomon_100", root / "homberger_200"
for d in (small, big, large):
    d.mkdir(parents=True, exist_ok=True)


def cut(inst, n, name):
    k = n + 1
    return Instance(name, inst.family, "solomon_txt", inst.xy[:k].copy(), inst.dem[:k].copy(), inst.ready[:k].copy(),
                    inst.due[:k].copy(), inst.serv[:k].copy(), inst.Q, inst.horizon)


for base in ("C101", "C109", "C106", "R101", "RC101"):
    inst = load_instance(src / "solomon_orig" / f"{base}.txt")
    save_solomon_txt(cut(inst, 100, f"{base}_100"), big / f"{base}_100.txt")
    for n in (25, 50):
        save_solomon_txt(cut(inst, n, f"{base}_{n:03d}"), small / f"{base}_{n:03d}.txt")
for base in ("C1_2_1", "R1_2_1", "RC1_2_1"):
    inst = load_instance(src / "homberger_200" / f"{base}.txt")
    save_solomon_txt(cut(inst, 200, f"{base}_200"), large / f"{base}_200.txt")
print("ok")
