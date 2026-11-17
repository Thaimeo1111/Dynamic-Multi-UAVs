"""T04 - Doc benchmark Solomon (VRP-REP .xml hoac Solomon .txt) va sinh instance tong hop.

Instance la du lieu TINH (toa do, nhu cau, cua so thoi gian, thoi gian phuc vu).
Cac yeu to dong (khach xuat hien dan, deadline doi, risk map, pin) nam o scenario
(xem event_generator.py), khong nam o instance.

Luu y trung thuc: R101/RC101 goc cua Solomon KHONG co san trong goi du lieu. Ham
make_synthetic tao instance "kieu R / kieu RC" co ten bat dau bang SYN_ de khong
bi nham voi benchmark that. Khi co file that (R101_050.xml hoac r101.txt) chi can
bo vao data/solomon_small/ va load_instance se doc.
"""
from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Instance:
    name: str
    family: str              # 'C', 'R', 'RC'
    source: str              # 'solomon_xml' | 'solomon_txt' | 'synthetic'
    xy: np.ndarray           # (n+1, 2); chi so 0 la depot
    dem: np.ndarray          # (n+1,)
    ready: np.ndarray
    due: np.ndarray
    serv: np.ndarray
    Q: float                 # tai trong toi da cua mot UAV
    horizon: float           # due cua depot = ket thuc ngay bay

    @property
    def n(self) -> int:
        return len(self.xy) - 1

    def dist(self) -> np.ndarray:
        d = self.xy[:, None, :] - self.xy[None, :, :]
        return np.hypot(d[..., 0], d[..., 1])


def _family(name: str) -> str:
    base = name.upper().replace("SYN_", "")
    if base.startswith("RC"):
        return "RC"
    if base.startswith("R"):
        return "R"
    return "C"


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def _child_text(elem, name):
    for ch in elem.iter():
        if ch is not elem and _local(ch.tag) == name and ch.text and ch.text.strip():
            return ch.text.strip()
    return None


def load_xml(path: Path) -> Instance:
    root = ET.parse(path).getroot()
    nodes = {}
    for e in root.iter():
        if _local(e.tag) == "node" and e.attrib.get("id") is not None:
            cx, cy = _child_text(e, "cx"), _child_text(e, "cy")
            if cx is not None and cy is not None:
                nodes[int(e.attrib["id"])] = (float(cx), float(cy))
    reqs = {}
    for e in root.iter():
        if _local(e.tag) == "request" and e.attrib.get("node") is not None:
            node = int(e.attrib["node"])
            reqs[node] = dict(
                ready=float(_child_text(e, "start")), due=float(_child_text(e, "end")),
                dem=float(_child_text(e, "quantity")), serv=float(_child_text(e, "service_time")))
    cap = None
    horizon = None
    for e in root.iter():
        if _local(e.tag) == "vehicle_profile":
            cap = float(_child_text(e, "capacity"))
            mt = _child_text(e, "max_travel_time")
            horizon = float(mt) if mt else None
            break
    if not nodes or not reqs or cap is None:
        raise ValueError(f"{path.name}: thieu node/request/vehicle_profile")
    ids = sorted(nodes)
    if ids[0] != 0:
        raise ValueError(f"{path.name}: khong tim thay depot (node 0)")
    xy = np.array([nodes[i] for i in ids], float)
    n1 = len(ids)
    dem, ready, due, serv = (np.zeros(n1) for _ in range(4))
    for k, i in enumerate(ids):
        if i == 0:
            continue
        r = reqs[i]
        dem[k], ready[k], due[k], serv[k] = r["dem"], r["ready"], r["due"], r["serv"]
    due[0] = horizon if horizon else float(due[1:].max())
    name = path.stem
    return Instance(name, _family(name), "solomon_xml", xy, dem, ready, due, serv, cap, float(due[0]))


def load_txt(path: Path) -> Instance:
    lines = path.read_text().splitlines()
    cap = None
    rows = []
    for i, ln in enumerate(lines):
        nums = re.findall(r"-?\d+\.?\d*", ln)
        if ln.strip().upper().startswith("NUMBER") and i + 1 < len(lines):
            nxt = re.findall(r"-?\d+\.?\d*", lines[i + 1])
            if len(nxt) >= 2:
                cap = float(nxt[1])
        elif len(nums) == 7 and not re.search(r"[A-Za-z]", ln):
            rows.append([float(v) for v in nums])
    if not rows or cap is None:
        raise ValueError(f"{path.name}: khong doc duoc dinh dang Solomon .txt")
    a = np.array(rows)
    order = np.argsort(a[:, 0])
    a = a[order]
    return Instance(path.stem, _family(path.stem), "solomon_txt", a[:, 1:3].copy(), a[:, 3].copy(),
                    a[:, 4].copy(), a[:, 5].copy(), a[:, 6].copy(), cap, float(a[0, 5]))


def load_instance(path) -> Instance:
    path = Path(path)
    if path.suffix.lower() == ".xml":
        return load_xml(path)
    if path.suffix.lower() == ".txt":
        return load_txt(path)
    raise ValueError(f"Dinh dang khong ho tro: {path.suffix}")


def save_solomon_txt(inst: Instance, path) -> None:
    out = [inst.name, "", "VEHICLE", "NUMBER     CAPACITY", f"  {inst.n}         {inst.Q:g}", "",
           "CUSTOMER", "CUST NO.  XCOORD.   YCOORD.    DEMAND   READY TIME  DUE DATE   SERVICE   TIME"]
    for i in range(inst.n + 1):
        out.append(f"{i:5d} {inst.xy[i,0]:9.2f} {inst.xy[i,1]:9.2f} {inst.dem[i]:9.0f} "
                   f"{inst.ready[i]:9.0f} {inst.due[i]:9.0f} {inst.serv[i]:9.0f}")
    Path(path).write_text("\n".join(out) + "\n")


def make_synthetic(family: str, n: int, seed: int, name: str | None = None,
                   horizon: float = 1000.0, depot=(40.0, 50.0)) -> Instance:
    """Instance tong hop theo phong cach Solomon (KHONG phai du lieu Solomon that).

    R : toa do deu ngau nhien; RC : nua cum nua ngau nhien; cua so thoi gian rong hon C.
    Moi khach co cua so kha thi (due >= thoi gian bay thang tu depot).
    """
    rng = np.random.default_rng(seed)
    if family == "R":
        pts = rng.uniform(0, 100, (n, 2))
    elif family == "RC":
        m = n // 2
        centers = rng.uniform(15, 85, (max(2, n // 12), 2))
        cl = centers[rng.integers(0, len(centers), n - m)] + rng.normal(0, 4.0, (n - m, 2))
        pts = np.vstack([rng.uniform(0, 100, (m, 2)), np.clip(cl, 0, 100)])
        pts = pts[rng.permutation(n)]
    else:
        raise ValueError("family phai la 'R' hoac 'RC'")
    pts = np.round(pts)
    depot = np.array(depot, float)
    d0 = np.hypot(*(pts - depot).T)
    dem = rng.choice([10, 20, 30, 40], n).astype(float)
    serv = np.full(n, 10.0)
    width = rng.uniform(60, 160, n)
    centre = rng.uniform(d0 + 30, np.maximum(d0 + 31, horizon - d0 - 40))
    ready = np.maximum(0.0, np.round(centre - width / 2))
    due = np.round(np.maximum(centre + width / 2, d0 + 5))
    due = np.minimum(due, horizon - d0 - serv)
    ready = np.minimum(ready, due)
    xy = np.vstack([depot, pts])
    z = np.zeros(1)
    nm = name or f"SYN_{family}101_{n:03d}"
    return Instance(nm, family, "synthetic", xy, np.concatenate([z, dem]), np.concatenate([z, ready]),
                    np.concatenate([[horizon], due]), np.concatenate([z, serv]), 200.0, horizon)


def prepare_small_data(src_xml_dir, out_dir, sizes=(25, 50)) -> list[str]:
    """Chep C101 (tu goi du lieu cua ban) va sinh SYN_R101/SYN_RC101 vao data/solomon_small."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for n in sizes:
        src = Path(src_xml_dir) / f"C101_{n:03d}.xml"
        if src.exists():
            (out / src.name).write_bytes(src.read_bytes())
            written.append(src.name)
        for fam, seed0 in (("R", 101), ("RC", 202)):
            inst = make_synthetic(fam, n, seed0 + n)
            save_solomon_txt(inst, out / f"{inst.name}.txt")
            written.append(f"{inst.name}.txt")
    return written


if __name__ == "__main__":
    import sys
    for p in sys.argv[1:]:
        i = load_instance(p)
        print(i.name, i.family, i.source, "n=", i.n, "Q=", i.Q, "H=", i.horizon)
