"""Animated Oat Optimization (AOO) - Wang et al., Knowledge-Based Systems 318 (2025) 113589.

CANH BAO: khong doc duoc bai goc (khong co ban truy cap mo, khong co ma nguon chinh thuc).
Ban nay dung lai tu phan mo ta AOO goc trong Yang et al., Electronic Research Archive
33(9) (2025) 5536-5590 (bai RLDN-AOO), muc "Animated Oat Optimization", Eq.(1)-(12)
va Algorithm 1. Cong thuc trich tu PDF bi vo mot phan nen mot so he so la UOC DOAN
(danh dau "uoc doan"). Dung de minh hoa co che, khong phai ban chuan.

Ba co che:
  - Phat tan xa (p = 1/2): buoc nhay ngau nhien W quanh vi tri cu / trung binh / tot nhat.
  - Lan hut am (p = 1/4): dao dong quanh X_best voi bien do A giam dan + Levy.
  - Bat vang khi gap vat can (p = 1/4): nhu tren nhung offset kieu nem xien J.
Chon loc tham lam: chi giu vi tri moi neu tot hon.
"""
from __future__ import annotations

import math

import numpy as np


def _levy(rng, dim, beta=1.5):
    sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2)
             / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = rng.normal(0, sigma, dim)
    v = rng.normal(0, 1, dim)
    return 0.01 * u / np.abs(v) ** (1 / beta)


def aoo(f, lb, ub, pop=50, iters=100, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    lb = np.asarray(lb, float); ub = np.asarray(ub, float)
    dim = len(lb)
    X = lb + rng.random((pop, dim)) * (ub - lb)
    fit = np.array([f(x) for x in X])
    b = fit.argmin(); xb = X[b].copy(); fb = fit[b]
    curve = []
    g = 9.8
    for t in range(1, iters + 1):
        for i in range(pop):
            r = rng.random()
            m = 0.5 * r / dim                      # khoi luong hat (Eq.1)
            L = pop * r / dim                      # do dai rau chinh (Eq.1)
            e = m                                  # he so quay lech tam (uoc doan)
            c = rng.random() * (1 - t / iters) ** 3  # he so dong (uoc doan dang suy giam)
            if rng.random() > 0.5:                 # phat tan xa, Eq.(2)-(3)
                W = c / math.pi * (2 * rng.random(dim) - 1) * ub
                if i % max(1, pop // 10) == 0:
                    xn = X.mean(0) + W
                elif i % max(1, pop // 10) == 1:
                    xn = xb + W
                else:
                    xn = X[i] + W
            elif rng.random() > 0.5:               # lan hut am, Eq.(4)-(8)
                A = ub - np.abs(ub * t / iters * math.sin(2 * math.pi * rng.random()))
                R = (m * e + L ** 2) / dim * rng.uniform(-A, A, dim)
                xn = xb + R + c * _levy(rng, dim) * xb
            else:                                  # bat vang, Eq.(9)-(12)
                B = ub - np.abs(ub * t / iters * math.cos(2 * math.pi * rng.random()))
                k = 0.5 + 0.5 * rng.random()
                x_ = 3 * rng.random()
                theta = math.pi * rng.random()
                alpha = 1 / (math.pi * math.e)
                J = 2 * k * x_ ** 2 / (m * g + 1e-12) * math.sin(2 * theta) * (1 - alpha) / dim * rng.uniform(-B, B, dim)
                J = np.clip(J, -(ub - lb), ub - lb)
                xn = xb + J + c * _levy(rng, dim) * xb
            xn = np.clip(xn, lb, ub)
            fn = f(xn)
            if fn < fit[i]:
                X[i] = xn; fit[i] = fn
                if fn < fb:
                    xb = xn.copy(); fb = fn
        curve.append(fb)
    return xb, fb, curve
