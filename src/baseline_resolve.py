"""T08 - Baseline RE-SOLVE: moi khi co su kien, giai lai tu dau (ALNS co ban, khoi tao nguoi, khong warm-start).

Phan da bay/dang phuc vu van duoc dong bang (khong the quay lai qua khu), nhung moi quyet dinh con lai
bi bo di va giai lai tu mot nghiem rong. Khong co toan tu nhan biet rui ro, khong chu ky dinh ky.
"""
from __future__ import annotations

from .alns_dynamic import BASE_CFG
from .rolling_horizon import ALNSPlanner


def make_resolve(iters, seed=0, min_gap=40.0, keep_trace=False):
    return ALNSPlanner("resolve", dict(BASE_CFG), iters, "events", min_gap=min_gap, seed=seed, keep_trace=keep_trace)
