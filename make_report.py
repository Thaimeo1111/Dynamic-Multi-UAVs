"""T26-T33: tong hop raw_*.csv -> multi_seed.csv, bang markdown, hinh (boxplot, hoi tu, ablation, sensitivity)."""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("tables", exist_ok=True); os.makedirs("figures", exist_ok=True); os.makedirs("results", exist_ok=True)
MET = ["J_ref", "dist", "risk", "late", "unserved", "ban_dist", "n_lost", "replans", "reassign", "plan_time"]
ORDER = ["greedy", "alns_static", "resolve", "alns_dynamic"]
COL = {"greedy": "#8c8c8c", "alns_static": "#d98b2b", "resolve": "#3b78b8", "alns_dynamic": "#2a9d5c",
       "resolve_periodic": "#7a5fb0"}


def md(df, path, fmt=".4g"):
    open(path, "w").write(df.to_markdown(index=False, floatfmt=fmt) + "\n")


def agg(d, by):
    g = d.groupby(by)
    t = g[MET].mean().add_suffix("_mean").join(g[["J_ref"]].std().rename(columns={"J_ref": "J_ref_std"}))
    t["J_ref_best"] = g["J_ref"].min()
    t["n_runs"] = g.size()
    return t.reset_index()


def main():
    if os.path.exists("results/raw_main.csv"):
        d = pd.read_csv("results/raw_main.csv")
        ms = agg(d, ["algo"]).set_index("algo").loc[[a for a in ORDER if a in set(d.algo)]].reset_index()
        ms.to_csv("results/multi_seed.csv", index=False); md(ms, "tables/multi_seed_overall.md")
        by = agg(d, ["n", "intensity", "algo"]); by.to_csv("results/multi_seed_by_size.csv", index=False)
        piv = d.pivot_table(index=["n", "intensity"], columns="algo", values="J_ref").reindex(columns=ORDER).reset_index()
        md(piv, "tables/main_J_by_size_intensity.md")
        # J nho hon la tot hon: lay do cai thien so voi resolve va static
        gains = d.pivot_table(index=["instance", "intensity", "seed"], columns="algo", values="J_ref").dropna()
        if "alns_dynamic" in gains:
            imp = pd.DataFrame({f"gain_vs_{b}_pct": 100 * (gains[b] - gains["alns_dynamic"]) / gains[b]
                                for b in ORDER[:-1] if b in gains}).reset_index()
            imp.to_csv("results/improved_vs_baseline.csv", index=False)
            md(imp.drop(columns=["seed"]).groupby("intensity").mean(numeric_only=True).reset_index(),
               "tables/improved_vs_baseline.md")
        # boxplot J_ref theo kich thuoc
        sizes = sorted(d.n.unique())
        fig, ax = plt.subplots(1, len(sizes), figsize=(4.2 * len(sizes), 4), squeeze=False)
        for a, n in zip(ax[0], sizes):
            sub = d[d.n == n]; algs = [x for x in ORDER if x in set(sub.algo)]
            bp = a.boxplot([sub[sub.algo == x].J_ref.values for x in algs], tick_labels=algs, patch_artist=True, showfliers=True)
            for p, x in zip(bp["boxes"], algs):
                p.set_facecolor(COL[x]); p.set_alpha(.8)
            a.set_title(f"n = {n}"); a.set_ylabel("J (thap hon = tot hon)"); a.tick_params(axis="x", rotation=25)
        fig.tight_layout(); fig.savefig("figures/boxplot_main.png", dpi=150)
    if os.path.exists("results/raw_ablation.csv"):
        d = pd.read_csv("results/raw_ablation.csv")
        if os.path.exists("results/raw_main.csv"):  # dong 'alns_dynamic' day du lay tu lo main tren cung instance/seed
            m = pd.read_csv("results/raw_main.csv")
            m = m[(m.algo == "alns_dynamic") & m.instance.isin(set(d.instance)) & m.seed.isin(set(d.seed))]
            d = pd.concat([d, m], ignore_index=True)
        t = agg(d, ["algo"]).sort_values("J_ref_mean"); t.to_csv("results/ablation.csv", index=False)
        base = t.loc[t.algo == "alns_dynamic", "J_ref_mean"]
        if len(base):
            t["delta_vs_full_pct"] = 100 * (t.J_ref_mean - base.iloc[0]) / base.iloc[0]
        md(t[[c for c in ["algo", "J_ref_mean", "J_ref_std", "dist_mean", "risk_mean", "late_mean", "unserved_mean",
                          "replans_mean", "delta_vs_full_pct", "n_runs"] if c in t]], "tables/ablation.md")
        fig, a = plt.subplots(figsize=(6.5, 3.8)); tt = t.iloc[::-1]
        a.barh(tt.algo, tt.J_ref_mean, xerr=tt.J_ref_std / np.sqrt(tt.n_runs), color=[COL.get(x, "#666") for x in tt.algo])
        a.set_xlabel("J trung binh (sai so chuan)"); fig.tight_layout(); fig.savefig("figures/ablation.png", dpi=150)
    if os.path.exists("results/raw_sensitivity.csv"):
        d = pd.read_csv("results/raw_sensitivity.csv")
        t = d.groupby(["factor", "level", "algo"])[["J_ref", "dist", "risk", "late", "unserved"]].mean().reset_index()
        t.to_csv("results/sensitivity.csv", index=False); md(t, "tables/sensitivity.md")
        fs = sorted(d.factor.unique())
        fig, ax = plt.subplots(1, len(fs), figsize=(4 * len(fs), 3.6), squeeze=False)
        for a, f in zip(ax[0], fs):
            for alg, g in t[t.factor == f].groupby("algo"):
                g = g.assign(lv=pd.to_numeric(g.level, errors="coerce")).sort_values("lv" if g.lv.notna().all() else "level") if False else g
                a.plot(g.level.astype(str), g.J_ref, marker="o", label=alg, color=COL.get(alg))
            a.set_title(f); a.set_ylabel("J")
        ax[0, 0].legend(fontsize=7); fig.tight_layout(); fig.savefig("figures/sensitivity.png", dpi=150)


if __name__ == "__main__":
    main()
