# this file checks for negative clip in the trinomial lattice, used in chapter 4

import json

import numpy as np

from swing_option.config import T, N, sigma, r, kappa, alpha, S_0, K, C_max, C_min, q_max
from swing_option.core.lattice import build_price_grid, transition_probabilities
from swing_option.core.pricing import compute_admissible_actions, compute_option_value

def scan_columns(K_sub, n_std=4):
    X_0 = S_0 - alpha(0)
    _, delta_t, delta_X, m, _ = build_price_grid(
        T=T, sigma=sigma, X_0=X_0, N=N, kappa=kappa, n_std=n_std, K_sub=K_sub,
    )
    n_nodes = 2 * m + 1

    raw = np.zeros((n_nodes, 3))
    for j in range(n_nodes):
        X_j = (j - m) * delta_X
        raw[j] = transition_probabilities(
            S=X_j, delta_t=delta_t, delta_S=delta_X,
            kappa=kappa, Theta_t=0.0, sigma=sigma,
            is_top=(j == n_nodes - 1), is_bottom=(j == 0),
            clip=False,
        )

    triggered = np.any(raw < 0.0, axis=1)
    worst_col = int(raw.min(axis=1).argmin())

    return dict(
        K_sub=K_sub,
        N_fine=N * K_sub,
        delta_t=float(delta_t),
        delta_X=float(delta_X),
        kappa_delta_t=float(kappa * delta_t),
        n_nodes=n_nodes,
        min_raw_prob=float(raw.min()),
        min_raw_col=worst_col,
        min_raw_X=float((worst_col - m) * delta_X),
        n_triggered=int(triggered.sum()),
        frac_triggered=float(triggered.sum() / n_nodes),
        triggered_columns=np.where(triggered)[0].tolist(),
    )

def price_clip_impact(K_sub, n_std=4):  # price clip on vs off
    X_0 = S_0 - alpha(0)
    grid, delta_t, delta_X, m, col_X0 = build_price_grid(
        T=T, sigma=sigma, X_0=X_0, N=N, kappa=kappa, n_std=n_std, K_sub=K_sub,
    )
    admissible = compute_admissible_actions(
        T=T, N=N, C_min=C_min, C_max=C_max, q_max=q_max, K_sub=K_sub,
    )

    prices = {}
    for clip in (True, False):
        _, _, price = compute_option_value(
            delta_t=delta_t, delta_X=delta_X, m=m, col_X0=col_X0, N=N,
            admissible=admissible, K=K, r=r, kappa=kappa, alpha_func=alpha,
            sigma=sigma, C_max=C_max, q_max=q_max, K_sub=K_sub, clip=clip,
        )
        prices["clipped" if clip else "unclipped"] = float(price)

    diff = prices["unclipped"] - prices["clipped"]
    return dict(
        price_clipped=prices["clipped"],
        price_unclipped=prices["unclipped"],
        abs_diff=diff,
        rel_diff=(diff / prices["clipped"]) if prices["clipped"] else float("nan"),
    )

def run_clip_diagnostics(k_sub_list, output_path=None):
    results = []

    for K_sub in k_sub_list:
        row = dict(scan_columns(K_sub))
        row.update(price_clip_impact(K_sub))
        results.append(row)

    if output_path is not None:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

    return results
