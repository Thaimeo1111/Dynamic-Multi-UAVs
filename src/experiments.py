"""Xay danh sach cong viec (job) cho tung thi nghiem va chay mot job. Dung boi run_batch.py / run_small.py."""
from __future__ import annotations

import time
from pathlib import Path

import yaml

from .algorithms import default_iters, make_planner
from .event_generator import make_scenario
from .instance_loader import load_instance
from .model import DEFAULT_P
from .rolling_horizon import run_episode

ROOT = Path(__file__).resolve().parents[1]
_CACHE = {}
COLUMNS = ["exp", "factor", "level", "instance", "family", "n", "seed", "intensity", "algo", "J", "J_ref", "dist",
           "risk", "late", "n_late", "unserved", "served", "ban_dist", "n_lost", "replans", "plan_time",
           "reassign", "wall"]


def load_config(path=None):
    return yaml.safe_load(open(path or ROOT / "config.yaml"))


def params_from(cfg, over=None):
    P = dict(DEFAULT_P)
    P.update(cfg.get("params", {}))
    P.update(over or {})
    return P


def algo_options(cfg, n, over=None):
    a = cfg["alns"]
    sizes = sorted(a["iters"])
    key = next((s for s in sizes if n <= s), sizes[-1])
    o = dict(iters=a["iters"][key], static_mult=a["static_mult"], min_gap=a["min_gap"], dt=a["dt_replan"],
             q_hi=a["q_hi"])
    o.update(over or {})
    return o


def make_job(exp, path, seed, algo, intensity="med", p_over=None, o_over=None, factor="", level=""):
    return dict(exp=exp, path=str(path), seed=int(seed), algo=algo, intensity=intensity, p_over=p_over or {},
                o_over=o_over or {}, factor=factor, level=level)


def build_jobs(exp, cfg, n_seeds=None):
    inst_main = cfg["instances"]["main"]
    inst_sub = cfg["instances"]["ablation_sensitivity"]
    inten = cfg["scenario"]["intensity"]
    jobs = []
    if exp == "main":
        e = cfg["experiments"]["main"]
        for p in inst_main:
            for s in range(1, (n_seeds or e["seeds"]) + 1):
                jobs += [make_job("main", ROOT / p, s, a, inten) for a in e["algos"]]
    elif exp == "ablation":
        e = cfg["experiments"]["ablation"]
        for p in inst_sub:
            for s in range(1, (n_seeds or e["seeds"]) + 1):
                jobs += [make_job("ablation", ROOT / p, s, a, inten) for a in e["algos"]]
    elif exp == "large":
        e = cfg["experiments"]["large"]
        for p in cfg["instances"]["large"]:
            for s in range(1, (n_seeds or e["seeds"]) + 1):
                jobs += [make_job("large", ROOT / p, s, a, inten, p_over=dict(battery=e["battery"])) for a in e["algos"]]
    elif exp == "sensitivity":
        e = cfg["experiments"]["sensitivity"]
        ns = n_seeds or e["seeds"]
        for p in inst_sub:
            n = int(p.split("_")[-1].split(".")[0])
            for s in range(1, ns + 1):
                for v in e["dt"]:
                    jobs.append(make_job("sensitivity", ROOT / p, s, "alns_dynamic", inten, o_over=dict(dt=v), factor="dt", level=v))
                for v in e["intensity"]:
                    jobs.append(make_job("sensitivity", ROOT / p, s, "alns_dynamic", v, factor="intensity", level=v))
                for v in e["w_r"]:
                    jobs.append(make_job("sensitivity", ROOT / p, s, "alns_dynamic", inten, p_over=dict(w_r=v), factor="w_r", level=v))
                for v in e["theta_ban"]:
                    jobs.append(make_job("sensitivity", ROOT / p, s, "alns_dynamic", inten, p_over=dict(theta_ban=v), factor="theta_ban", level=v))
                base_it = cfg["alns"]["iters"][min((k for k in cfg["alns"]["iters"] if n <= k), default=100)]
                for v in e["iters_scale"]:
                    jobs.append(make_job("sensitivity", ROOT / p, s, "alns_dynamic", inten, o_over=dict(iters=max(10, int(base_it * v))), factor="iters", level=v))
                for v in e["q_hi"]:
                    jobs.append(make_job("sensitivity", ROOT / p, s, "alns_dynamic", inten, o_over=dict(q_hi=v), factor="q_hi", level=v))
    elif exp == "sensitivity_large":
        # do nhay o 200 khach: chi 2 tham so nhay nhat (theta_ban, w_r), pin 450 nhu thi nghiem large
        e = cfg["experiments"]["sensitivity_large"]
        for p in cfg["instances"]["large"]:
            for s in range(1, (n_seeds or e["seeds"]) + 1):
                for v in e["theta_ban"]:
                    jobs.append(make_job("sensitivity_large", ROOT / p, s, "alns_dynamic", inten, p_over=dict(battery=e["battery"], theta_ban=v), factor="theta_ban", level=v))
                for v in e["w_r"]:
                    jobs.append(make_job("sensitivity_large", ROOT / p, s, "alns_dynamic", inten, p_over=dict(battery=e["battery"], w_r=v), factor="w_r", level=v))
    else:
        raise ValueError(exp)
    jobs.sort(key=lambda j: -int(j["path"].split("_")[-1].split(".")[0]))      # job nang chay truoc
    return jobs


def job_key(j):
    return (j["exp"], str(j["factor"]), str(j["level"]), Path(j["path"]).stem, j["seed"], j["algo"], j["intensity"])


def run_job(job, cfg=None):
    cfg = cfg or load_config()
    t0 = time.perf_counter()
    inst = _CACHE.get(job["path"]) or _CACHE.setdefault(job["path"], load_instance(job["path"]))
    P = params_from(cfg, job["p_over"])
    scn = make_scenario(inst, job["seed"], job["intensity"], battery=P["battery"],
                        online_frac=cfg["scenario"]["online_frac"])
    planner = make_planner(job["algo"], inst.n, job["seed"], algo_options(cfg, inst.n, job["o_over"]))
    m = run_episode(inst, scn, P, planner)
    ref = DEFAULT_P
    J_ref = ref["w_d"] * m["dist"] + ref["w_r"] * m["risk"] + ref["w_l"] * m["late"] + ref["w_u"] * m["unserved"]
    row = dict(exp=job["exp"], factor=job["factor"], level=job["level"], instance=inst.name, family=inst.family,
               n=inst.n, seed=job["seed"], intensity=job["intensity"], algo=job["algo"], J_ref=J_ref,
               wall=time.perf_counter() - t0, **m)
    INTS = {"n", "seed", "n_late", "unserved", "served", "replans", "reassign", "n_lost"}
    out = {}
    for k in COLUMNS:
        v = row[k]
        if k in INTS:
            out[k] = int(v)
        elif isinstance(v, (int, float)) or hasattr(v, "item"):
            out[k] = round(float(v), 4)
        else:
            out[k] = v
    return out
