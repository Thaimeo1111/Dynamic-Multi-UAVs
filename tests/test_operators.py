"""T16 - Unit test cho toan tu va mo phong: kiem tra tuyen kha thi ve pin, deadline/cua so, vung cam, tai trong.

Chay:  python -m unittest discover -s tests -v
"""
import unittest
from pathlib import Path

import numpy as np

from src.alns_dynamic import BASE_CFG, FULL_CFG, solve
from src.destroy_ops import DESTROY
from src.event_generator import make_scenario
from src.evaluator import check_plan
from src.instance_loader import load_instance, make_synthetic
from src.model import DEFAULT_P, eval_route, make_problem, recompute_cost, solution_from_routes, empty_solution
from src.repair_ops import REPAIR, scan_route
from src.risk_map import Hotspot, RiskField
from src.rolling_horizon import ALNSPlanner, Sim, build_problem, run_episode, trigger_times, warm_routes
from src.algorithms import make_planner, MAIN, ABLATION

ROOT = Path(__file__).resolve().parents[1]
P = dict(DEFAULT_P)


def small_instances():
    out = [make_synthetic("R", 25, 7), make_synthetic("RC", 30, 8)]
    f = ROOT / "data" / "solomon_small" / "C101_025.xml"
    if f.exists():
        out.append(load_instance(f))
    return out


def problem_at_zero(inst, seed=1, K=4, battery=300.0, hotspots=None):
    scn = make_scenario(inst, seed, "med", battery=battery, K=K)
    sim = Sim(inst, scn, P)
    return build_problem(sim, 0.0, include_all=True), sim, scn


def hot(x, y, sigma=6.0, amp=1.0):
    return Hotspot(0, sigma, amp, 0.0, np.inf, [(0.0, x, y), (2000.0, x, y)])


class OperatorFeasibility(unittest.TestCase):
    def check(self, pb, sol, label):
        self.assertEqual(check_plan(pb, sol.routes), [], label)
        placed = [c for r in sol.routes for c in r]
        self.assertEqual(len(placed), len(set(placed)), f"{label}: khach lap")
        self.assertEqual(set(placed) | sol.un, set(range(1, pb.m + 1)), f"{label}: mat khach")
        self.assertTrue(set(placed).isdisjoint(sol.un), f"{label}: vua xep vua chua xep")
        c0 = sol.cost
        self.assertAlmostEqual(recompute_cost(pb, sol), c0, 6, f"{label}: chi phi lech")

    def test_repair_operators_keep_feasibility(self):
        for inst in small_instances():
            pb, _, _ = problem_at_zero(inst)
            for name, fn in REPAIR.items():
                sol = empty_solution(pb)
                fn(pb, sol, np.random.default_rng(1))
                self.check(pb, sol, f"{inst.name}/{name}")
                self.assertGreater(sum(len(r) for r in sol.routes), 0)

    def test_destroy_then_repair_keeps_feasibility(self):
        rng = np.random.default_rng(3)
        for inst in small_instances():
            pb, _, _ = problem_at_zero(inst)
            base = empty_solution(pb)
            REPAIR["regret2"](pb, base, rng)
            for dn, dfun in DESTROY.items():
                for rn, rfun in REPAIR.items():
                    sol = base.copy()
                    removed = dfun(pb, sol, 5, rng)
                    self.assertGreaterEqual(len(removed), 1)
                    self.assertTrue(set(removed) <= sol.un)
                    rfun(pb, sol, rng)
                    self.check(pb, sol, f"{inst.name}/{dn}+{rn}")

    def test_alns_improves_and_stays_feasible(self):
        for inst in small_instances():
            pb, _, _ = problem_at_zero(inst)
            for cfg in (BASE_CFG, FULL_CFG):
                sol, info = solve(pb, None, 80, np.random.default_rng(2), cfg, record_trace=True)
                self.check(pb, sol, inst.name)
                self.assertLessEqual(info["trace"][-1][1], info["trace"][0][1] + 1e-9)

    def test_alns_is_reproducible(self):
        pb, _, _ = problem_at_zero(small_instances()[0])
        a, _ = solve(pb, None, 60, np.random.default_rng(5), FULL_CFG)
        b, _ = solve(pb, None, 60, np.random.default_rng(5), FULL_CFG)
        self.assertEqual(a.routes, b.routes)


class HardConstraints(unittest.TestCase):
    def make(self, hotspots=None, battery=300.0, K=3):
        inst = make_synthetic("R", 12, 5)
        scn = make_scenario(inst, 1, "low", battery=battery, K=K)
        scn["hotspots"] = [h.to_dict() for h in (hotspots or [])]
        scn["events"] = []
        scn["release"] = [0.0] * (inst.n + 1)
        sim = Sim(inst, scn, P)
        return inst, build_problem(sim, 0.0, include_all=True)

    def test_battery_limit(self):
        inst, pb = self.make(battery=40.0)
        sol = empty_solution(pb)
        REPAIR["regret2"](pb, sol, np.random.default_rng(0))
        for k, seq in enumerate(sol.routes):
            self.assertLessEqual(eval_route(pb, k, seq).dist, 40.0 + 1e-9)
        self.assertGreater(len(sol.un), 0, "pin 40 khong the phuc vu het 12 khach")

    def test_capacity_limit(self):
        inst, pb = self.make()
        pb.cap = [30.0] * pb.K
        sol = empty_solution(pb)
        REPAIR["greedy"](pb, sol, np.random.default_rng(0))
        for k, seq in enumerate(sol.routes):
            self.assertLessEqual(sum(pb.dem[c] for c in seq), 30.0 + 1e-9)

    def test_no_fly_zone_blocks_customer_and_legs(self):
        inst = make_synthetic("R", 12, 5)
        c = 3
        x, y = inst.xy[c]
        _, pb = self.make([hot(x, y, sigma=6.0, amp=1.0)])
        # khach nam ngay tam hotspot: khong the toi (BAN o diem cuoi)
        sol = empty_solution(pb)
        REPAIR["regret2"](pb, sol, np.random.default_rng(0))
        self.assertIn(c, sol.un)
        self.assertEqual(check_plan(pb, sol.routes), [])
        for seq in sol.routes:                                  # khong tuyen nao co chang di qua loi
            self.assertNotIn(c, seq)

    def test_time_window_is_soft_with_penalty(self):
        inst, pb = self.make()
        sol = empty_solution(pb)
        REPAIR["greedy"](pb, sol, np.random.default_rng(0))
        late = sum(i.late for i in sol.infos)
        self.assertGreaterEqual(late, 0.0)
        for k, seq in enumerate(sol.routes):                   # tre trong RouteInfo khop tinh tay
            if not seq:
                continue
            info = eval_route(pb, k, seq)
            manual = sum(max(0.0, s - pb.due[c]) for c, s in zip(seq, info.st))
            self.assertAlmostEqual(info.late, manual, 6)


class SimulationInvariants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inst = small_instances()[0]
        cls.scn = make_scenario(cls.inst, 2, "high")

    def test_each_customer_served_at_most_once_and_capacity(self):
        for name in MAIN + ABLATION:
            sim = Sim(self.inst, self.scn, P)
            pl = make_planner(name, self.inst.n, 1)
            for idx, t in enumerate(pl.triggers(self.scn, sim.Tend)):
                sim.advance_to(t)
                pl.plan(sim, t, idx)
            sim.advance_to(sim.Tend + 1)
            load = {}
            for c, (s, k) in sim.served.items():
                load[k] = load.get(k, 0.0) + self.inst.dem[c]
                self.assertGreaterEqual(s + 1e-9, max(self.inst.ready[c], sim.release[c]), f"{name}: phuc vu truoc ready/release")
            for k, v in load.items():
                self.assertLessEqual(v, self.inst.Q + 1e-9, f"{name}: UAV {k} qua tai")
            for u in sim.u:
                self.assertGreaterEqual(u["bat"], -1e-6, f"{name}: pin am")
            self.assertEqual(len(sim.served) + sim.metrics()["unserved"], self.inst.n)

    def test_reproducible_and_scenario_deterministic(self):
        a = make_scenario(self.inst, 2, "high")
        self.assertEqual(a["events"], self.scn["events"])
        m1 = run_episode(self.inst, self.scn, P, make_planner("alns_dynamic", self.inst.n, 1))
        m2 = run_episode(self.inst, self.scn, P, make_planner("alns_dynamic", self.inst.n, 1))
        for key in ("J", "dist", "risk", "late", "unserved"):
            self.assertAlmostEqual(m1[key], m2[key], 6)

    def test_table_risk_matches_continuous_sampling(self):
        """Phoi nhiem rui ro tinh tu bang (luc lap ke hoach) khop mau lien tuc (luc thuc thi) khi belief = truth."""
        inst = self.inst
        scn = make_scenario(inst, 1, "low")
        scn["events"] = []
        scn["release"] = [0.0] * (inst.n + 1)
        scn["hotspots"] = [h for h in scn["hotspots"] if h["kind"] == "initial"]
        for h in scn["hotspots"]:                              # hotspot dung yen -> belief == truth
            h["waypoints"] = [[0.0, h["waypoints"][0][1], h["waypoints"][0][2]], [2000.0, h["waypoints"][0][1], h["waypoints"][0][2]]]
        sim = Sim(inst, scn, P)
        pb = build_problem(sim, 0.0, include_all=True)
        sol = empty_solution(pb)
        REPAIR["regret2"](pb, sol, np.random.default_rng(0))
        planned = sum(i.risk for i in sol.infos)
        sim.set_plan([[pb.gid[c] for c in r] for r in sol.routes])
        sim.advance_to(sim.Tend + 1)
        self.assertGreater(planned, 1.0)
        self.assertLess(abs(sim.risk - planned) / planned, 0.2)

    def test_trigger_times(self):
        scn = self.scn
        for mode in ("once", "targets", "events"):
            t = trigger_times(scn, mode, 1236.0, 40.0, 100.0 if mode == "events" else None)
            self.assertEqual(t[0], 0.0)
            self.assertEqual(t, sorted(t))
            self.assertTrue(all(b - a > 1.0 for a, b in zip(t, t[1:])))
        self.assertEqual(trigger_times(scn, "once", 1236.0), [0.0])


class WarmStart(unittest.TestCase):
    def test_warm_start_keeps_feasible_routes(self):
        inst = small_instances()[0]
        pb, _, _ = problem_at_zero(inst)
        sol, _ = solve(pb, None, 60, np.random.default_rng(1), FULL_CFG)
        again = solution_from_routes(pb, sol.routes)
        self.assertEqual(again.routes, sol.routes)
        self.assertAlmostEqual(again.cost, sol.cost, 6)
        warm, _ = solve(pb, sol.routes, 0, np.random.default_rng(1), FULL_CFG)
        self.assertLessEqual(warm.cost, sol.cost + 1e-6)




# ====================================================================== kiem tra truc tiep eval_route
def line_problem(c2_due=55.0, Tend=1000.0, battery=300.0, cap=200.0, hotspots=None, safe=0.0):
    """Depot (40,50); c1 (50,50) cua so [0,100]; c2 (60,50) cua so [50, c2_due]; phuc vu 10; toc do 1."""
    Pl = dict(P, depot_safe=safe)
    fld = RiskField(hotspots or [], (0, 0, 100, 100), Pl["cell"])
    cust = dict(gid=[1, 2], xy=np.array([[50.0, 50.0], [60.0, 50.0]]), ready=[0.0, 50.0],
                due=[100.0, c2_due], serv=[10.0, 10.0], dem=[20.0, 30.0])
    uav = [dict(x=40.0, y=50.0, t_free=0.0, bat=battery, cap=cap, alive=True)]
    return make_problem(Pl, fld, 0.0, np.array([40.0, 50.0]), cust, uav, Tend)


class EvalRouteByHand(unittest.TestCase):
    def test_hand_computed_route(self):
        pb = line_problem()
        i = eval_route(pb, 0, [1, 2])
        # 40->50: 10 (den 10, bat dau 10, xong 20); ->60: 10 (den 30, cho den 50, xong 60); ve depot: 20 (tret 80)
        self.assertAlmostEqual(i.dist, 40.0)
        self.assertEqual(i.st, [10.0, 50.0])
        self.assertAlmostEqual(i.tret, 80.0)
        self.assertAlmostEqual(i.late, 0.0)
        self.assertAlmostEqual(i.cost, 40.0)
        j = eval_route(pb, 0, [2, 1])
        self.assertEqual(j.st, [50.0, 70.0])           # c2 doi ready=50, roi sang c1
        self.assertAlmostEqual(j.dist, 40.0)

    def test_lateness_value(self):
        pb = line_problem(c2_due=45.0)
        i = eval_route(pb, 0, [1, 2])
        self.assertAlmostEqual(i.late, 5.0)            # bat dau luc 50, due 45
        self.assertEqual(i.nlate, 1)
        self.assertAlmostEqual(i.cost, 40.0 + 5.0)

    def test_battery_violation(self):
        self.assertIsNotNone(eval_route(line_problem(battery=40.0), 0, [1, 2]))
        self.assertIsNone(eval_route(line_problem(battery=39.9), 0, [1, 2]))

    def test_capacity_violation(self):
        self.assertIsNotNone(eval_route(line_problem(cap=50.0), 0, [1, 2]))
        self.assertIsNone(eval_route(line_problem(cap=49.9), 0, [1, 2]))

    def test_end_of_day_violation(self):
        self.assertIsNotNone(eval_route(line_problem(Tend=80.0), 0, [1, 2]))
        self.assertIsNone(eval_route(line_problem(Tend=79.9), 0, [1, 2]))

    def test_no_fly_leg_violation(self):
        pb = line_problem(hotspots=[hot(55.0, 51.0, sigma=5.0)])   # tam o luoi (55,51); sigma >= 2 o luoi
        self.assertIsNotNone(eval_route(pb, 0, [1]))   # toi c1 chua cham hotspot
        self.assertIsNone(eval_route(pb, 0, [1, 2]))   # chang c1->c2 xuyen loi
        self.assertIsNone(eval_route(pb, 0, [2]))

    def test_depot_safe_radius_exempts_depot_area(self):
        inst_h = [hot(43.0, 51.0, sigma=5.0)]          # hotspot ngay sat depot
        self.assertIsNone(eval_route(line_problem(hotspots=inst_h, safe=0.0), 0, [1]))
        self.assertIsNotNone(eval_route(line_problem(hotspots=inst_h, safe=8.0), 0, [1]))


def independent_eval(pb, k, seq):
    """Bo kiem tra viet doc lap (khong goi eval_route): tra ve (kha_thi, dist, late) tu D, BAN, thoi gian."""
    cur, t, dist, late, load = pb.start_node[k], pb.t0[k], 0.0, 0.0, 0.0
    nodes = list(seq) + [0]
    for idx, c in enumerate(nodes):
        d = pb.D[cur][c]
        kb = min(max(int((t - pb.T0b) / pb.dB), 0), pb.nb - 1)
        if pb.BAN[kb][cur][c]:
            return False, None, None
        dist += d
        t += d / pb.P["speed"]
        if c != 0:
            t = max(t, pb.ready[c])
            late += max(0.0, t - pb.due[c])
            t += pb.serv[c]
            load += pb.dem[c]
        cur = c
    ok = dist <= pb.bat[k] + 1e-9 and load <= pb.cap[k] + 1e-9 and t <= pb.Tend
    return ok, dist, late


class EvalRouteDifferential(unittest.TestCase):
    def test_matches_independent_checker_on_random_routes(self):
        rng = np.random.default_rng(11)
        n_ok = n_bad = 0
        for inst in small_instances():
            for bat in (120.0, 300.0):
                pb, _, _ = problem_at_zero(inst, seed=3, K=3, battery=bat)
                for _ in range(120):
                    seq = list(rng.permutation(np.arange(1, pb.m + 1))[: int(rng.integers(1, 9))])
                    info = eval_route(pb, 0, seq)
                    ok, dist, late = independent_eval(pb, 0, seq)
                    self.assertEqual(info is not None, ok, f"{inst.name} {seq}")
                    if ok:
                        n_ok += 1
                        self.assertAlmostEqual(info.dist, dist, 6)
                        self.assertAlmostEqual(info.late, late, 6)
                    else:
                        n_bad += 1
        self.assertGreater(n_ok, 20)
        self.assertGreater(n_bad, 20)               # phai co ca ca kha thi va khong kha thi


if __name__ == "__main__":
    unittest.main()
