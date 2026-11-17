"""T32: kiem dinh thong ke tren results/raw_main.csv (theo cap, khoi = instance x intensity x seed).
- Friedman tren cac thuat toan chinh (trung binh theo khoi);
- Wilcoxon signed-rank cap doi voi alns_dynamic, hieu chinh Holm (tu cai dat, khong co statsmodels);
- co hieu ung: A12 (Vargha-Delaney), Cliff's delta, rank-biserial; HV/IGD xem run_mo.py."""
import argparse
import numpy as np, pandas as pd
from scipy import stats


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i]); adj[i] = min(1.0, run)
    return adj


def a12(x, y):  # P(x<y) + 0.5 P(x=y): >0.5 nghia la x (chi phi) tot hon y
    x, y = np.asarray(x), np.asarray(y)
    return float(((x[:, None] < y[None]).sum() + 0.5 * (x[:, None] == y[None]).sum()) / (len(x) * len(y)))


def cliffs(x, y):
    return 1 - 2 * a12(x, y)  # <0: x thap hon (tot hon)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="results/raw_main.csv"); ap.add_argument("--metric", default="J_ref")
    ap.add_argument("--ref", default="alns_dynamic"); ap.add_argument("--out", default="tables/stats_tests")
    a = ap.parse_args()
    d = pd.read_csv(a.raw)
    piv = d.pivot_table(index=["instance", "intensity", "seed"], columns="algo", values=a.metric)
    piv = piv.dropna()
    algos = [c for c in piv.columns]
    out = [f"# Kiem dinh thong ke ({a.metric}, {len(piv)} khoi ghep cap day du)\n"]
    if len(algos) >= 3:
        fr = stats.friedmanchisquare(*[piv[c] for c in algos])
        ranks = piv.rank(axis=1).mean().sort_values()
        out.append(f"Friedman: chi2={fr.statistic:.2f}, p={fr.pvalue:.2e}\n\nRang trung binh (thap = tot): "
                   + ", ".join(f"{k}={v:.2f}" for k, v in ranks.items()) + "\n")
    rows = []
    for c in algos:
        if c == a.ref:
            continue
        x, y = piv[a.ref].values, piv[c].values
        try:
            w = stats.wilcoxon(x, y)
            p = w.pvalue
        except ValueError:
            p = 1.0
        diff = y - x
        rb = float((np.sum(diff > 0) - np.sum(diff < 0)) / max(1, np.sum(diff != 0)))
        rows.append(dict(vs=c, mean_ref=x.mean(), mean_other=y.mean(), rel_gain_pct=100 * (y.mean() - x.mean()) / y.mean(),
                         p_raw=p, A12=a12(x, y), cliffs_delta=cliffs(x, y), rank_biserial=rb, n=len(x)))
    t = pd.DataFrame(rows)
    if len(t):
        t["p_holm"] = holm(t.p_raw.values)
        t["signif_0.05"] = t.p_holm < 0.05
    t.to_csv(a.out + ".csv", index=False)
    out.append(f"\nWilcoxon `{a.ref}` vs con lai (hieu chinh Holm); A12>0.5: `{a.ref}` co chi phi thap hon; "
               f"Cliff delta<0: `{a.ref}` tot hon.\n\n" + (t.to_markdown(index=False, floatfmt=".3g") if len(t) else "") + "\n")
    open(a.out + ".md", "w").write("".join(out))
    print("".join(out))


if __name__ == "__main__":
    main()
