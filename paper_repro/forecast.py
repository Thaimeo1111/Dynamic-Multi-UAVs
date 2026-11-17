"""Du bao cung-cau vat tu (muc 4 va 5.3, Eq.19-43, Bang 6-7).

Du lieu: 5 diem cuu tro, M = 100 buoc, nhieu 2*N(0,1).
Dung 3 buoc lich su -> du bao 1 buoc; 80% dau de train, 20% cuoi de kiem dinh.
Moi diem (POS) co mo hinh rieng.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import differential_evolution
from scipy.signal import sawtooth
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor

from .aoo import aoo

LAG = 3
U_PULSE = 3    # chu ky xung F4. Bai ghi U = "so buoc toi da" (chi 1 xung). Chu ky 3 la gia tri
#               khop voi R2 = 0.998 cua POS4 trong Bang 7 (xung du doan duoc tu 3 buoc lich su).


def make_series(M=100, seed=0):
    rng = np.random.default_rng(seed)
    u = np.arange(1, M + 1, dtype=float)
    eps = lambda: rng.normal(0, 1, M)
    F1 = 0.01 * u + np.abs(100 * np.sin(np.pi * u / 6)) + 2 * eps()
    F2 = np.abs(100 * np.sin(np.pi * u / 4)) + 2 * eps()
    F3 = np.abs(100 * sawtooth(np.pi * u / 2, 0.1)) + 2 * eps()
    F4 = 100 * (np.mod(u, U_PULSE) == 0) + 2 * eps()
    F5 = 0.02 * u + np.abs(100 * np.cos(np.pi * u / 4)) + 2 * eps()
    return np.column_stack([F1, F2, F3, F4, F5])      # Eq.25: M x 5


def windows(y, lag=LAG):
    X = np.column_stack([y[k:len(y) - lag + k] for k in range(lag)])
    return X, y[lag:]


def split(y, frac=0.8, lag=LAG):
    X, t = windows(y, lag)
    n_tr = int(round(frac * len(t)))
    return X[:n_tr], t[:n_tr], X[n_tr:], t[n_tr:]


def metrics(y, p):
    mse = float(np.mean((y - p) ** 2)); mae = float(np.mean(np.abs(y - p)))
    r2 = float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    return mse, mae, r2


class _Scaler:
    def fit(self, X, y):
        self.lo, self.hi = min(X.min(), y.min()), max(X.max(), y.max())
        return self

    def x(self, X):
        return (X - self.lo) / (self.hi - self.lo)

    def inv(self, y):
        return y * (self.hi - self.lo) + self.lo


# ------------------------------------------------------------- baselines
def rf_model(Xtr, ytr, Xte, rng):
    m = RandomForestRegressor(n_estimators=100, random_state=int(rng.integers(1 << 31)))
    return m.fit(Xtr, ytr).predict(Xte)


def rbf_net(Xtr, ytr, Xte, spread=1.0, max_neurons=100, goal=0.0):
    """Gan voi newrb cua MATLAB: them dan neuron tai mau co sai so lon nhat."""
    s = _Scaler().fit(Xtr, ytr); A = s.x(Xtr); y = s.x(ytr); B = s.x(Xte)
    b = np.sqrt(-np.log(0.5)) / spread
    phi = lambda P, C: np.exp(-(b * np.sqrt(((P[:, None] - C[None]) ** 2).sum(-1))) ** 2)
    centers = []
    pred = np.zeros_like(y)
    for _ in range(min(max_neurons, len(A))):
        err = np.abs(y - pred)
        cand = [i for i in np.argsort(-err) if i not in centers]
        if not cand or np.mean(err ** 2) <= goal:
            break
        centers.append(cand[0])
        H = np.column_stack([phi(A, A[centers]), np.ones(len(A))])
        w = np.linalg.lstsq(H, y, rcond=1e-6)[0]
        pred = H @ w
    Ht = np.column_stack([phi(B, A[centers]), np.ones(len(B))])
    return s.inv(Ht @ w)


def kelm(Xtr, ytr, Xte, C=1.0, gamma=0.5):
    s = _Scaler().fit(Xtr, ytr); A = s.x(Xtr); y = s.x(ytr); B = s.x(Xte)
    # quy uoc trong ma KELM MATLAB cua Huang: exp(-||x-y||^2 / kernel_para)
    k = lambda P, Q: np.exp(-((P[:, None] - Q[None]) ** 2).sum(-1) / gamma)
    beta = np.linalg.solve(k(A, A) + np.eye(len(A)) / C, y)
    return s.inv(k(B, A) @ beta)


def elm(Xtr, ytr, Xte, rng, hidden=160):
    s = _Scaler().fit(Xtr, ytr); A = s.x(Xtr); y = s.x(ytr); B = s.x(Xte)
    W = rng.uniform(-1, 1, (A.shape[1], hidden)); bb = rng.uniform(-1, 1, hidden)
    H = lambda P: 1 / (1 + np.exp(-(P @ W + bb)))
    beta = np.linalg.pinv(H(A), rcond=1e-6) @ y    # Eq.33 (cat tri rieng nho de on dinh; 160 nut > 77 mau)
    return s.inv(H(B) @ beta)


def arima(series, n_tr_end, n_te, orders=((13, 1, 13), (13, 1, 1), (3, 1, 1))):
    """Fit tren chuoi train, du bao 1 buoc lan luot tren tap kiem dinh (dung gia tri thuc da quan sat)."""
    from statsmodels.tsa.arima.model import ARIMA
    for od in orders:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = ARIMA(series[:n_tr_end], order=od).fit()
                full = res.apply(series[:n_tr_end + n_te])
                pred = full.predict(start=n_tr_end, end=n_tr_end + n_te - 1)
            if np.all(np.isfinite(pred)):
                return np.asarray(pred), od
        except Exception:
            continue
    raise RuntimeError("ARIMA khong hoi tu")


# ------------------------------------------------------------- RBF-MKELM
class RBFMKELM:
    """Eq.26-38: lop RBF (15 tam K-means, chon lan lap co WCSS nho nhat) -> nhan da kernel Poly + wavelet."""

    def __init__(self, n_rbf=15, seed=0):
        self.n_rbf = n_rbf; self.seed = seed

    def fit_centers(self, A):
        km = KMeans(n_clusters=min(self.n_rbf, len(A)), n_init=20, random_state=self.seed).fit(A)
        self.centers = km.cluster_centers_
        return self

    def rbf(self, P, delta):
        return np.exp(-((P[:, None] - self.centers[None]) ** 2).sum(-1) / (2 * delta ** 2))   # Eq.29

    @staticmethod
    def mk(P, Q, a, b, c, d, alpha):
        poly = (P @ Q.T + c) ** d                                                   # Eq.36
        diff2 = (P[:, None] - Q[None]) ** 2
        wav = np.prod((1 - diff2 / a ** 2) * np.exp(-diff2 / (2 * b ** 2)), axis=-1)  # Eq.37
        return alpha * poly + (1 - alpha) * wav                                     # Eq.35

    def predict_with(self, theta, A, y, B):
        delta, C, a, b, c, d, alpha = theta
        alpha = min(max(alpha, 0.0), 1.0)                # trong so kernel trong [0,1]
        Pa, Pb = self.rbf(A, delta), self.rbf(B, delta)
        K = self.mk(Pa, Pa, a, b, c, d, alpha)
        try:
            beta = np.linalg.solve(K + np.eye(len(A)) / C, y)
        except np.linalg.LinAlgError:
            return None
        return self.mk(Pb, Pa, a, b, c, d, alpha) @ beta                            # Eq.38


LB = np.full(7, 0.01)
UB = np.full(7, 10.0)


def rbf_mkelm(Xtr, ytr, Xte, rng, objective="train", optimizer="aoo", pop=50, iters=100):
    """objective='train': RMSE tren tap train nhu bai (de qua khop);
    'holdout': RMSE tren 20% cuoi cua tap train (khong dung tap kiem dinh)."""
    s = _Scaler().fit(Xtr, ytr); A = s.x(Xtr); y = s.x(ytr); B = s.x(Xte)
    model = RBFMKELM(seed=int(rng.integers(1 << 31))).fit_centers(A)
    if objective == "train":
        fa, fy, fb, fby = A, y, A, y
    else:
        k = int(0.8 * len(A)); fa, fy, fb, fby = A[:k], y[:k], A[k:], y[k:]

    def f(theta):
        with np.errstate(all="ignore"):
            p = model.predict_with(theta, fa, fy, fb)
        if p is None or not np.all(np.isfinite(p)):
            return 1e9
        return float(np.sqrt(np.mean((p - fby) ** 2)))                              # Eq.40

    if optimizer == "aoo":
        theta, best, curve = aoo(f, LB, UB, pop=pop, iters=iters, rng=rng)
    else:
        r = differential_evolution(f, list(zip(LB, UB)), popsize=max(1, pop // 7), maxiter=iters,
                                   seed=int(rng.integers(1 << 31)), tol=1e-8, polish=False)
        theta, best, curve = r.x, r.fun, []
    with np.errstate(all="ignore"):
        p = model.predict_with(theta, A, y, B)
    if p is None or not np.all(np.isfinite(p)):
        p = np.full(len(B), y.mean())
    return s.inv(p), theta, curve


def run_all(seed=0, quick=False, data_seed=0):
    """Tra ve list dict: pos, model, mse, mae, r2 (+ du bao de ve hinh)."""
    rng = np.random.default_rng(seed)
    D = make_series(seed=data_seed)
    rows, preds = [], {}
    it = 30 if quick else 100
    for p in range(5):
        y = D[:, p]
        Xtr, ytr, Xte, yte = split(y)
        n_tr_end = LAG + len(ytr)
        out = {
            "RF": rf_model(Xtr, ytr, Xte, rng),
            "RBF": rbf_net(Xtr, ytr, Xte),
            "KELM": kelm(Xtr, ytr, Xte),
            "ELM": elm(Xtr, ytr, Xte, rng),
        }
        out["ARIMA"], od = arima(y, n_tr_end, len(yte))
        out["OUR"], th, _ = rbf_mkelm(Xtr, ytr, Xte, rng, "train", "aoo", iters=it)
        out["OUR-holdout"], th2, _ = rbf_mkelm(Xtr, ytr, Xte, rng, "holdout", "aoo", iters=it)
        out["OUR-DE"], th3, _ = rbf_mkelm(Xtr, ytr, Xte, rng, "holdout", "de", iters=it)
        for k, v in out.items():
            mse, mae, r2 = metrics(yte, v)
            rows.append({"pos": p + 1, "model": k, "mse": mse, "mae": mae, "r2": r2,
                         "note": f"ARIMA{od}" if k == "ARIMA" else (
                             "theta=" + ",".join(f"{x:.3g}" for x in {"OUR": th, "OUR-holdout": th2, "OUR-DE": th3}[k])
                             if k.startswith("OUR") else "")})
        preds[p + 1] = {"y": yte, **out}
    return rows, preds, D
