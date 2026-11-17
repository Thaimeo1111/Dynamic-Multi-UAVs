"""Sinh bang so sanh (bai bao vs ta chay) va hinh tu results/paper_repro/<mode>/*.csv."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from .paper_values import ALGO_ORDER, MODELS7, TABLE1, TABLE2, TABLE4, TABLE5, TABLE7

ROOT = Path(__file__).resolve().parents[1]


def _read(path):
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def _dirs(mode):
    r = ROOT / "results" / "paper_repro" / mode
    t = ROOT / "tables" / "paper_repro" / mode
    f = ROOT / "figures" / "paper_repro" / mode
    t.mkdir(parents=True, exist_ok=True); f.mkdir(parents=True, exist_ok=True)
    return r, t, f


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


# ------------------------------------------------------------------ Bang 1
def table1(mode):
    r, t, _ = _dirs(mode)
    rows = _read(r / "table1_som_alpha.csv")
    if not rows:
        return
    out = ["# Bang 1 - 3D-SOM voi alpha = 0 va alpha = 3 (100 task, A:B:C = 1:3:6)", "",
           "Do 'thuan loai' = ti le task thuoc loai chiem da so cua moi UAV, trung binh tren cac UAV va cac lan chay.", ""]
    for alpha in ("0.0", "3.0"):
        rs = [x for x in rows if x["alpha"] == alpha]
        pur = []
        for s in set(x["seed"] for x in rs):
            for x in rs:
                if x["seed"] == s:
                    c = np.array([int(x["A"]), int(x["B"]), int(x["C"])])
                    if c.sum():
                        pur.append(c.max() / c.sum())
        p_paper = np.mean([max(c) / sum(c) for c in TABLE1[int(float(alpha))] if sum(c)])
        out += [f"## alpha = {alpha}", "", "| UAV | Bai bao A/B/C | Ta chay (lan 0) A/B/C |", "|---|---|---|"]
        r0 = [x for x in rs if x["seed"] == "0"]
        for j, x in enumerate(r0):
            pa = TABLE1[int(float(alpha))][j]
            out.append(f"| UAV{j + 1} | {pa[0]}/{pa[1]}/{pa[2]} | {x['A']}/{x['B']}/{x['C']} |")
        out += ["", f"Do thuan loai: bai bao {p_paper:.2f}, ta chay {np.mean(pur):.2f} (trung binh {len(set(x['seed'] for x in rs))} lan).", ""]
        # so UAV khong nhan task nao ("neuron chet") va so task toi da tren 1 UAV, theo tung lan chay
        seeds = sorted(set(x["seed"] for x in rs), key=int)
        tot = {s: [int(x["A"]) + int(x["B"]) + int(x["C"]) for x in rs if x["seed"] == s] for s in seeds}
        dead = [sum(1 for v in tot[s] if v == 0) for s in seeds]
        mx = [max(tot[s]) for s in seeds]
        out += [f"So UAV khong nhan task nao (tung lan chay): {dead}",
                f"So task toi da tren 1 UAV (tung lan chay): {mx}", ""]
    (t / "table1_som_alpha.md").write_text("\n".join(out), encoding="utf-8")


# ------------------------------------------------------------------ Bang 2
def table2(mode):
    r, t, _ = _dirs(mode)
    rows = _read(r / "table2_auction.csv")
    if not rows:
        return
    by = defaultdict(list)
    for x in rows:
        by[int(x["n"])].append({k: float(v) for k, v in x.items()})
    out = ["# Bang 2 - Truoc / sau toi uu dau gia (5 UAV, 1 loai task)", "",
           "Sum = Length + 3*Std (cot Sum cua bai khop dung cong thuc nay). Cai thien tinh tren Sum.",
           "Ta chay: trung binh tren cac lan chay; Eq.14 = gia day du 1*P + 1.5*D + 2*R.", "",
           "| n | Bai: Len truoc | Std truoc | Sum truoc | Len sau | Std sau | Sum sau | Cai thien | "
           "Ta: Len truoc | Std truoc | Sum truoc | Len sau | Std sau | Sum sau | Cai thien | Eq.14 cai thien | so vong | thoi gian (s) |",
           "|---" * 19 + "|"]
    imp_ours, imp_paper = [], []
    for n in sorted(by):
        v = by[n]
        m = lambda k: np.mean([x[k] for x in v])
        s0 = np.mean([x["P0"] + 3 * x["D0"] for x in v]); s1 = np.mean([x["P1"] + 3 * x["D1"] for x in v])
        imp = 100 * np.mean([((x["P0"] + 3 * x["D0"]) - (x["P1"] + 3 * x["D1"])) / (x["P0"] + 3 * x["D0"]) for x in v])
        impc = 100 * np.mean([(x["C0"] - x["C1"]) / x["C0"] for x in v])
        imp_ours.append(imp)
        p = TABLE2.get(n)
        pc = (f"{p[0][0]:.1f} | {p[0][1]:.1f} | {p[0][2]:.1f} | {p[1][0]:.1f} | {p[1][1]:.1f} | {p[1][2]:.1f} | {p[2]:.2f}%"
              if p else "- | - | - | - | - | - | -")
        if p:
            imp_paper.append(p[2])
        out.append(f"| {n} | {pc} | {m('P0'):.1f} | {m('D0'):.1f} | {s0:.1f} | {m('P1'):.1f} | {m('D1'):.1f} | {s1:.1f} | "
                   f"{imp:.2f}% | {impc:.2f}% | {m('rounds'):.1f} | {m('time'):.2f} |")
    out += ["", f"Cai thien trung binh: bai bao {np.mean(imp_paper):.2f}%, ta chay {np.mean(imp_ours):.2f}%."]
    allv = [x for v in by.values() for x in v]
    n_up = sum(x["P1"] > x["P0"] for x in allv)
    n_dn = sum(x["D1"] < x["D0"] for x in allv)
    out += [f"Length tang sau dau gia: {n_up}/{len(allv)} lan chay ({100 * n_up / len(allv):.0f}%); "
            f"Std giam: {n_dn}/{len(allv)} ({100 * n_dn / len(allv):.0f}%)."]
    (t / "table2_auction.md").write_text("\n".join(out), encoding="utf-8")


# ------------------------------------------------------------------ Bang 4, 5
def _agg_alloc(rows):
    by = defaultdict(list)
    for x in rows:
        by[(int(x["n"]), x["algo"])].append((float(x["cost"]), float(x["time"])))
    return {k: (np.mean([c for c, _ in v]), np.std([c for c, _ in v], ddof=1) if len(v) > 1 else None,
                np.mean([t for _, t in v]), len(v)) for k, v in by.items()}


def _avg_improvement(agg, sizes, algos, ref="OUR"):
    """Cach bai tinh '34.5%': trung binh (bench - OUR)/bench tren moi cap (quy mo, benchmark)."""
    vals = []
    for n in sizes:
        if (n, ref) not in agg:
            continue
        for a in algos:
            if a != ref and (n, a) in agg:
                vals.append((agg[(n, a)][0] - agg[(n, ref)][0]) / agg[(n, a)][0])
    return 100 * np.mean(vals) if vals else float("nan")


def table_alloc(mode, kind):
    r, t, f = _dirs(mode)
    rows = _read(r / f"alloc_{kind}.csv")
    if not rows:
        return
    paper = TABLE4 if kind == "single" else TABLE5
    agg = _agg_alloc(rows)
    sizes = sorted({n for n, _ in agg})
    algos = [a for a in ALGO_ORDER + ["OUR-a0"] if any((n, a) in agg for n in sizes)]
    title = "Bang 4 - mot loai task" if kind == "single" else "Bang 5 - nhieu loai task (6:3:1, diem 5/3/1)"
    out = [f"# {title}", "", "Gia tri: chi phi Eq.14 (thap hon la tot hon), trung binh +/- do lech chuan, thoi gian (s).",
           "Moi o: **Bai bao** / Ta chay. Cot 'hang' la thu hang cua thuat toan trong quy mo do (1 = tot nhat).", ""]
    for n in sizes:
        out += [f"## {n} task", "", "| Thuat toan | Bai: mean | std | time | hang bai | Ta: mean | std | time | hang ta | so lan | Ta/Bai |",
                "|---|---|---|---|---|---|---|---|---|---|---|"]
        rank_p = {a: i + 1 for i, a in enumerate(sorted(paper.get(n, {}), key=lambda a: paper[n][a][0]))}
        ours_algos = [a for a in algos if (n, a) in agg]
        rank_o = {a: i + 1 for i, a in enumerate(sorted([a for a in ours_algos if a in ALGO_ORDER], key=lambda a: agg[(n, a)][0]))}
        for a in algos:
            p = paper.get(n, {}).get(a)
            o = agg.get((n, a))
            ps = f"{p[0]:.1f} | {'-' if p[1] is None else f'{p[1]:.1f}'} | {p[2]:g} | {rank_p.get(a, '-')}" if p else "- | - | - | -"
            os_ = (f"{o[0]:.1f} | {'-' if o[1] is None else f'{o[1]:.1f}'} | {o[2]:.2f} | {rank_o.get(a, '-')} | {o[3]}"
                   if o else "- | - | - | - | 0")
            ratio = f"{o[0] / p[0]:.2f}" if (o and p) else "-"
            out.append(f"| {a} | {ps} | {os_} | {ratio} |")
        out.append("")
    # tong ket
    bench = [a for a in ALGO_ORDER if a != "OUR"]
    pagg = {(n, a): (paper[n][a][0],) for n in paper for a in paper[n]}
    out += ["## Tong ket", "",
            f"- Muc cai thien trung binh cua OUR so voi cac benchmark: bai bao {_avg_improvement(pagg, sizes, ALGO_ORDER):.1f}% "
            f"(tren cac quy mo da chay), ta chay {_avg_improvement(agg, sizes, ALGO_ORDER):.1f}%."]
    wins = defaultdict(int)
    for n in sizes:
        cand = [a for a in ALGO_ORDER if (n, a) in agg]
        if cand:
            wins[min(cand, key=lambda a: agg[(n, a)][0])] += 1
    out.append("- So quy mo ma moi thuat toan tot nhat (ta chay): " + ", ".join(f"{a}: {k}" for a, k in sorted(wins.items(), key=lambda x: -x[1])))
    for a in ["KCA", "3D-SOM", "OUR-a0"]:
        if any((n, a) in agg for n in sizes):
            d = [100 * (agg[(n, a)][0] - agg[(n, 'OUR')][0]) / agg[(n, 'OUR')][0] for n in sizes if (n, a) in agg and (n, 'OUR') in agg]
            out.append(f"- {a} so voi OUR (ta chay): chenh trung binh {np.mean(d):+.1f}% (duong = {a} te hon).")
    (t / f"alloc_{kind}.md").write_text("\n".join(out), encoding="utf-8")

    # hinh: chi phi theo quy mo, bai bao vs ta
    plt = _plt()
    fig, axs = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    cmap = plt.cm.tab10
    for i, a in enumerate(ALGO_ORDER + (["OUR-a0"] if "OUR-a0" in algos else [])):
        style = dict(color=cmap(i % 10), lw=2.5 if a == "OUR" else 1.2, ls="--" if a == "OUR-a0" else "-", marker="o", ms=3)
        if a in ALGO_ORDER:
            ns = [n for n in sorted(paper) if a in paper[n]]
            axs[0].plot(ns, [paper[n][a][0] for n in ns], label=a, **style)
        ns = [n for n in sizes if (n, a) in agg]
        axs[1].plot(ns, [agg[(n, a)][0] for n in ns], label=a, **style)
    axs[0].set_title("Bai bao"); axs[1].set_title(f"Ta chay ({mode})")
    for ax in axs:
        ax.set_xlabel("So task"); ax.grid(alpha=.3)
    axs[0].set_ylabel("Chi phi Eq.14"); axs[1].legend(fontsize=8, ncol=2)
    fig.suptitle(title)
    fig.tight_layout(); fig.savefig(f / f"alloc_{kind}.png", dpi=110); plt.close(fig)


def fig_paths(mode):
    """Tuong tu Hinh 11/12: duong bay cua OUR va KCA voi 100 task."""
    from .baselines import kca, ours
    from .common import evaluate, make_scenario
    _, _, f = _dirs(mode)
    plt = _plt()
    fig, axs = plt.subplots(2, 2, figsize=(11, 11))
    for row, multi in enumerate([False, True]):
        sc = make_scenario(100, multi, 10_000 * (2 if multi else 1) + 101 * 100)
        for col, (name, fn) in enumerate([("OUR (3D-SOM + dau gia)", ours), ("KCA (K-means + dau gia)", kca)]):
            kw = {"alpha": 3.0 if multi else 0.0} if fn is ours else {}
            a = fn(sc, 5, np.random.default_rng(0), **kw)
            e = evaluate(sc, a, 5, return_routes=True)
            ax = axs[row, col]
            mk = {0: "*", 1: "s", 2: "o"}
            for j, rt in enumerate(e["routes"]):
                pts = np.vstack([[50, 50], sc["xy"][rt]]) if len(rt) else np.array([[50, 50]])
                ax.plot(pts[:, 0], pts[:, 1], "-", color=plt.cm.tab10(j), lw=1)
                for ty in (0, 1, 2):
                    sel = rt[sc["types"][rt] == ty]
                    ax.scatter(*sc["xy"][sel].T, color=plt.cm.tab10(j), marker=mk[ty], s=30 if ty == 0 else 14)
            ax.scatter(50, 50, marker="^", c="k", s=80)
            ax.set_title(f"{'Da loai' if multi else 'Mot loai'} - {name}\nEq.14 = {e['cost']:.1f}, P = {e['P']:.1f}, D = {e['D']:.1f}, R = {e['R']:.1f}", fontsize=9)
            ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.set_aspect("equal")
    fig.suptitle("Duong bay 5 UAV, 100 task (sao = A, vuong = B, tron = C; mau = UAV)")
    fig.tight_layout(); fig.savefig(f / "paths_100.png", dpi=110); plt.close(fig)


# ------------------------------------------------------------------ Bang 7
def table7(mode):
    r, t, f = _dirs(mode)
    rows = _read(r / "forecast.csv")
    if not rows:
        return
    ours = {(int(x["pos"]), x["model"]): (float(x["mse"]), float(x["mae"]), float(x["r2"]), x["note"]) for x in rows}
    models = list(dict.fromkeys(x["model"] for x in rows))
    out = ["# Bang 7 - Du bao cung-cau (MSE / MAE / R2 tren 20% du lieu kiem dinh)", "",
           "OUR = RBF-MKELM + AOO voi muc tieu RMSE tap train (dung nhu bai). "
           "OUR-holdout = AOO toi uu RMSE tren 20% cuoi tap train. OUR-DE = holdout + Differential Evolution thay AOO.",
           "Muc nhieu cua du lieu: 2*N(0,1) -> MSE toi thieu ly thuyet ~ 4.", ""]
    for pos in [1, 2, 3, 4, 5, "mean"]:
        out += [f"## POS {pos}", "", "| Mo hinh | Bai: MSE | MAE | R2 | Ta: MSE | MAE | R2 |", "|---|---|---|---|---|---|---|"]
        for m in models:
            p = TABLE7[pos][MODELS7.index(m)] if m in MODELS7 else None
            if pos == "mean":
                v = [ours[(k, m)] for k in range(1, 6) if (k, m) in ours]
                o = tuple(np.mean([x[i] for x in v]) for i in range(3))
            else:
                o = ours.get((pos, m))
            ps = f"{p[0]:.3f} | {p[1]:.3f} | {p[2]:.3f}" if p else "- | - | -"
            out.append(f"| {m} | {ps} | {o[0]:.3f} | {o[1]:.3f} | {o[2]:.3f} |")
        out.append("")
    notes = sorted({(m, ours[(k, m)][3]) for (k, m) in ours if ours[(k, m)][3]})
    out += ["## Ghi chu tham so", ""] + [f"- {m}: {n}" for m, n in notes]
    (t / "table7_forecast.md").write_text("\n".join(out), encoding="utf-8")

    z = np.load(r / "forecast_preds.npz")
    plt = _plt()
    fig, axs = plt.subplots(5, 1, figsize=(10, 15))
    show = [m for m in ["RF", "ARIMA", "OUR", "OUR-holdout"] if f"p1_{m}" in z]
    for p in range(1, 6):
        ax = axs[p - 1]
        ax.plot(z[f"p{p}_y"], "k-o", ms=3, lw=1.5, label="Thuc te")
        for m in show:
            ax.plot(np.clip(z[f"p{p}_{m}"], -50, 200), lw=1, label=m)
        ax.set_title(f"POS{p}", fontsize=9); ax.grid(alpha=.3)
    axs[0].legend(fontsize=8, ncol=5)
    fig.suptitle("Du bao 1 buoc tren tap kiem dinh (tuong tu Hinh 13; da cat truc y ve [-50, 200])")
    fig.tight_layout(); fig.savefig(f / "forecast_fit.png", dpi=110); plt.close(fig)


def table7_seeds(mode):
    """Gop forecast.csv (seed du lieu 0) voi forecast_seeds.csv (paper_repro.forecast_seeds)."""
    r, t, _ = _dirs(mode)
    extra = _read(r / "forecast_seeds.csv")
    if not extra:
        return
    rows = [dict(x, data_seed="0") for x in _read(r / "forecast.csv")] + extra
    models = list(dict.fromkeys(x["model"] for x in rows))
    seeds = sorted(set(x["data_seed"] for x in rows), key=int)
    mse = {(x["data_seed"], x["model"], x["pos"]): float(x["mse"]) for x in rows}
    mean = {(s, m): np.mean([v for (s2, m2, _), v in mse.items() if s2 == s and m2 == m]) for s in seeds for m in models}
    fmt = lambda v: f"{v:.2f}" if v < 1e3 else f"{v:.1e}"
    out = ["# Bang 7 qua nhieu bo du lieu - MSE trung binh 5 POS", "",
           "Seed 0 = forecast.csv; cac seed khac = forecast_seeds.csv. Gia tri nho nhat moi dong in dam.", "",
           "| Seed | " + " | ".join(models) + " |", "|---" * (len(models) + 1) + "|"]
    win_seed, win_pos = defaultdict(int), defaultdict(int)
    for s in seeds:
        best = min(models, key=lambda m: mean[(s, m)])
        win_seed[best] += 1
        out.append(f"| {s} | " + " | ".join(("**%s**" if m == best else "%s") % fmt(mean[(s, m)]) for m in models) + " |")
        for p in sorted(set(x["pos"] for x in rows)):
            win_pos[min(models, key=lambda m: mse.get((s, m, p), np.inf))] += 1
    med = {m: float(np.median([mean[(s, m)] for s in seeds])) for m in models}
    out.append("| Trung vi | " + " | ".join(fmt(med[m]) for m in models) + " |")
    n_pairs = sum(win_pos.values())
    out += ["", f"So seed moi mo hinh tot nhat: " + ", ".join(f"{m} {win_seed[m]}" for m in models if win_seed[m]),
            f"So cap (seed, POS) moi mo hinh tot nhat (tren {n_pairs}): "
            + ", ".join(f"{m} {c}" for m, c in sorted(win_pos.items(), key=lambda kv: -kv[1]))]
    (t / "table7_seeds.md").write_text("\n".join(out), encoding="utf-8")


def table_dynamic(mode):
    r, t, _ = _dirs(mode)
    rows = _read(r / "dynamic_events.csv")
    if not rows:
        return
    out = ["# Mo phong dong (tuong tu Hinh 14)", "", "| t | Su kien | UAV hoat dong | Task con | Da xong | Pin 5 UAV |", "|---|---|---|---|---|---|"]
    out += [f"| {x['t']} | {x['event']} | {x['active_uav']} | {x['tasks_left']} | {x['done']} | {x['energy']} |" for x in rows]
    out += ["", f"Tong quang duong bay: {rows[-1]['total_path']}"]
    (t / "dynamic_events.md").write_text("\n".join(out), encoding="utf-8")


def make_all(mode):
    table1(mode); table2(mode); table7(mode); table7_seeds(mode); table_dynamic(mode)
    for k in ("single", "multi"):
        table_alloc(mode, k)
    try:
        fig_paths(mode)
    except Exception as e:   # pragma: no cover
        print("fig_paths loi:", e)


if __name__ == "__main__":
    import sys
    make_all(sys.argv[1] if len(sys.argv) > 1 else "quick")
