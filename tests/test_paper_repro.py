"""Kiem tra cac thanh phan tai lap bai Pu et al. (2026)."""
import numpy as np

from paper_repro.auction import auction
from paper_repro.common import build, cost_from, evaluate, hier_route, make_scenario, _nn_order, _two_opt, route_len
from paper_repro.forecast import make_series, metrics
from paper_repro.som3d import som_allocate


def test_fig4_std_is_sample_std():
    # Hinh 4: duong 20 va 15 -> std 3.533 ; 13 va 17 -> std 2.8284 (do lech chuan mau)
    assert abs(cost_from([20, 15], [0, 0])[2] - 3.5355) < 1e-3
    assert abs(cost_from([13, 17], [0, 0])[2] - 2.8284) < 1e-3


def test_two_opt_never_longer_than_nn():
    sc = make_scenario(60, False, 3)
    idx = np.arange(60)
    r = _nn_order(sc["D"], 60, idx)
    r2 = _two_opt(sc["D"], 60, r.copy())
    assert sorted(r2) == list(range(60))
    assert route_len(sc["D"], 60, r2) <= route_len(sc["D"], 60, r) + 1e-9


def test_hierarchical_tsp_visits_high_priority_first():
    sc = make_scenario(40, True, 5)
    r, _ = hier_route(sc, sc["n"], np.arange(40))
    t = sc["types"][r]
    assert np.all(np.diff(t) >= 0)


def test_auction_does_not_increase_cost():
    sc = make_scenario(80, False, 2)
    a0 = som_allocate(sc, 5, alpha=0.0, rng=np.random.default_rng(0), it_max=20)
    a1, hist = auction(sc, a0, 5)
    assert all(b <= a + 1e-9 for a, b in zip(hist, hist[1:]))
    assert evaluate(sc, a1, 5, opt=False)["cost"] <= evaluate(sc, a0, 5, opt=False)["cost"] + 1e-9


def test_som_alpha_separates_types():
    def purity(alpha):
        vals = []
        for s in range(3):
            sc = make_scenario(100, True, 50 + s)
            a = som_allocate(sc, 5, alpha=alpha, rng=np.random.default_rng(s), it_max=30)
            for j in range(5):
                c = np.bincount(sc["types"][a == j], minlength=3)
                if c.sum():
                    vals.append(c.max() / c.sum())
        return np.mean(vals)
    assert purity(3.0) > purity(0.0) + 0.1


def test_fig4_toy_auction():
    # 2 UAV, 8 task cung loai: dau gia khong lam tang chi phi va giu du task
    xy = np.array([[45, 60], [40, 70], [42, 80], [60, 62], [60, 70], [65, 80], [55, 40], [45, 35]], float)
    sc = build(xy, np.zeros(8, int), np.ones(8))
    a0 = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    a1, hist = auction(sc, a0, 2)
    assert hist[-1] <= hist[0]
    assert np.bincount(a1, minlength=2).sum() == 8


def test_series_and_metrics():
    D = make_series(M=100, seed=0)
    assert D.shape == (100, 5)
    y = D[:, 1]
    mse, mae, r2 = metrics(y, y)
    assert mse == 0 and mae == 0 and r2 == 1


def test_report_extras_on_fake_csv(tmp_path, monkeypatch):
    import csv
    from paper_repro import report

    monkeypatch.setattr(report, "ROOT", tmp_path)
    r = tmp_path / "results" / "paper_repro" / "t"
    r.mkdir(parents=True)

    def w(name, rows):
        with open(r / name, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)

    w("table1_som_alpha.csv", [{"seed": 0, "alpha": a, "uav": j + 1, "A": c[0], "B": c[1], "C": c[2]}
                               for a in (0.0, 3.0) for j, c in enumerate([(10, 0, 0), (0, 30, 0), (0, 0, 60), (0, 0, 0), (0, 0, 0)])])
    w("table2_auction.csv", [{"n": 20, "run": 0, "P0": 100, "D0": 10, "C0": 120, "P1": 105, "D1": 2, "C1": 110, "rounds": 2, "time": 0.1}])
    base = [{"pos": p, "model": m, "mse": v + p, "mae": 1, "r2": 0.9, "note": ""} for p in (1, 2) for m, v in (("RF", 1), ("OUR", 9))]
    w("forecast.csv", base)
    w("forecast_seeds.csv", [dict(x, data_seed=1) for x in base])
    report.table1("t"); report.table2("t"); report.table7_seeds("t")
    t = tmp_path / "tables" / "paper_repro" / "t"
    assert "[2]" in (t / "table1_som_alpha.md").read_text(encoding="utf-8")
    assert "Length tang sau dau gia: 1/1" in (t / "table2_auction.md").read_text(encoding="utf-8")
    s = (t / "table7_seeds.md").read_text(encoding="utf-8")
    assert "RF 2" in s and "(tren 4): RF 4" in s
