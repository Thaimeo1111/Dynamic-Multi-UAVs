"""T06 - Evaluator: tinh objective va kiem tra kha thi.

Hai muc danh gia, dung CUNG cong thuc muc tieu (xem model.py):
  * plan_report(pb, sol)   : danh gia MOT ke hoach tren bang rui ro cua belief (dung khi lap ke hoach / test)
  * Sim.metrics()          : danh gia THUC TE sau khi UAV bay tren risk map that (dung de so sanh thuat toan)
Cong thuc:  J = w_d*quang_duong + w_r*phoi_nhiem_rui_ro + w_l*tong_tre + w_u*so_khach_khong_phuc_vu
"""
from __future__ import annotations

from .model import components, eval_route


def objective(P, dist, risk, late, unserved):
    return P["w_d"] * dist + P["w_r"] * risk + P["w_l"] * late + P["w_u"] * unserved


def check_plan(pb, routes):
    """Liet ke vi pham cung cua mot ke hoach (routes: danh sach tuyen, chi so cuc bo). Rong = kha thi."""
    problems, seen = [], set()
    for k, seq in enumerate(routes):
        for c in seq:
            if c in seen:
                problems.append(f"khach {c} xuat hien nhieu lan")
            seen.add(c)
        if seq and eval_route(pb, k, seq) is None:
            problems.append(f"UAV {k}: vi pham tai trong/pin/vung cam/ket thuc ngay bay")
    return problems


def plan_report(pb, sol):
    d, r, late, nlate, nun = components(pb, sol)
    return dict(dist=d, risk=r, late=late, n_late=nlate, unserved=nun,
                J=objective(pb.P, d, r, late, nun), violations=check_plan(pb, sol.routes))
