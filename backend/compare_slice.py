# backend/compare_slice.py
import numpy as np
from app.boundary import compute_boundary
from swing_option import config

r = compute_boundary(
    T=config.T, N=config.N, sigma=config.sigma, kappa=config.kappa,
    r=config.r, K=config.K, S_0=config.S_0, alpha_func=config.alpha,
    C_max=config.C_max, C_min=config.C_min, q_max=config.q_max, K_sub=24,
)
b, t = r["boundary"], r["time_grid"]
for C in (1000, 1040, 1050):
    col = b[:, C]
    fin = np.isfinite(col)
    print(C, "first t:", round(float(t[fin][0]), 4), "last 5:", np.round(col[fin][-5:], 4))