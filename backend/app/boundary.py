"""Parameterised entry point into the thesis solver.

Wraps the existing pipeline -- scripts/run_analysis.py:compute_data() for the
grid/admissible/value-function build, and swing_option/analysis/optimal_strategy.py
for boundary extraction -- with every parameter passed explicitly instead of
read from swing_option/config.py module globals. This is the layer boundary:
no FastAPI, no pydantic, no HTTP. Callable from the API, a script, or a
notebook.

Do not import swing_option.config here. A fallback to baseline values would
be a second, silently-diverging source of defaults alongside the API's own
SolveRequest -- callers (including validate_boundary.py) pass everything.
"""

import time

import numpy as np

from swing_option.analysis.optimal_strategy import extract_boundary, interpolate_boundary
from swing_option.core.lattice import build_price_grid, transition_probabilities
from swing_option.core.pricing import compute_admissible_actions, compute_option_value


def _count_clip_fired(m, delta_X, delta_t, kappa, sigma):
    """Number of lattice nodes whose pre-clip transition probabilities would
    have gone negative. transition_probabilities() silently clips and
    renormalises inside compute_option_value; this re-derives the same
    per-node probabilities with clip=False purely to audit whether the guard
    was actually needed at this resolution. It does not touch the pricing
    loop -- just an O(n_nodes) pass over the same function it calls.
    """
    n_nodes = 2 * m + 1
    fired = 0
    for j in range(n_nodes):
        X_j = (j - m) * delta_X
        probs = transition_probabilities(
            S=X_j, delta_t=delta_t, delta_S=delta_X,
            kappa=kappa, Theta_t=0.0, sigma=sigma,
            is_top=(j == n_nodes - 1), is_bottom=(j == 0),
            clip=False,
        )
        if np.any(probs < 0):
            fired += 1
    return fired


def compute_boundary(
    *,
    T: float,
    N: int,
    sigma: float,
    kappa: float,
    r: float,
    K: float,
    S_0: float,
    alpha_func,
    C_max: int,
    C_min: int,
    q_max: int,
    K_sub: int,
) -> dict:
    """Solve one fully-specified swing option config and return its exercise
    boundary surface.

    Date convention: exercise dates are i in range(0, N_fine + 1, K_sub),
    i.e. N + 1 dates spanning [0, T] inclusive of both endpoints. `boundary`,
    `time_grid` and `alpha` all share this length. The terminal date's
    boundary row is NaN throughout -- there is no forward-looking
    continuation value at T -- matching optimal_strategy.analyse().

    Volume convention: column C of `boundary` is the threshold when C units
    have already been exercised (cumulative volume, not a remaining-capacity
    count), so it must be non-decreasing in C. This falls out of reusing
    extract_boundary/interpolate_boundary as-is; validate_boundary.py checks
    it empirically rather than asserting it here, since a strict per-row
    assertion would be sensitive to isolated floating-point noise near flat
    regions and this module should not make a config fail for that.
    """
    started = time.time()

    X_0 = S_0 - alpha_func(0)
    grid, delta_t, delta_X, m, col_X0 = build_price_grid(
        T=T, sigma=sigma, X_0=X_0, N=N, kappa=kappa, K_sub=K_sub,
    )

    admissible = compute_admissible_actions(
        T=T, N=N, C_min=C_min, C_max=C_max, q_max=q_max, K_sub=K_sub,
    )

    V, policy, option_price = compute_option_value(
        delta_t=delta_t, delta_X=delta_X, m=m, col_X0=col_X0, N=N,
        admissible=admissible, K=K, r=r, kappa=kappa, alpha_func=alpha_func,
        sigma=sigma, C_max=C_max, q_max=q_max, K_sub=K_sub,
    )

    # -- boundary extraction: reuse the thesis code, do not reimplement ----
    data = {
        "grid": grid, "policy": policy, "V": V,
        "m": m, "col_X0": col_X0,
        "delta_t": delta_t, "delta_X": delta_X,
        "admissible": admissible,
    }
    boundary_full, forced_mask_full = extract_boundary(
        grid, policy, admissible, alpha_func, delta_t,
    )
    boundary_interp_full, _diag = interpolate_boundary(
        data, boundary_full, kappa=kappa, sigma=sigma, K=K, r=r, alpha_func=alpha_func,
    )

    # Terminal date has no V[i+1] to interpolate against (interpolate_boundary
    # skips it), so it never gets a genuine threshold.
    boundary_interp_full[-1, :] = np.nan

    N_fine = N * K_sub
    ex_idx = np.arange(0, N_fine + 1, K_sub)  # N + 1 exercise dates
    boundary = boundary_interp_full[ex_idx]
    forced_mask = forced_mask_full[ex_idx]

    time_grid = ex_idx * delta_t
    alpha_vals = alpha_func(time_grid)

    # -- stats: computed per config, nothing hardcoded ---------------------
    stationary_sd = sigma / np.sqrt(2.0 * kappa)

    # Mean date among (date, C) states where q=0 is inadmissible -- i.e.
    # dates that have at least one forced-exercise capacity level.
    forced_rows = forced_mask.any(axis=1)
    forced_onset_mean_t = (
        float(time_grid[forced_rows].mean()) if forced_rows.any() else None
    )

    # No path simulation happens here (that's policy_analysis.py's job), so
    # "share of volume that is forced" is read off the state space directly:
    # the fraction of (date, C) cells where the minimum-volume constraint
    # forbids q=0. Equal-weighted per cell, i.e. per unit of cumulative
    # volume already exercised, across every exercise date.
    pct_forced_volume = float(100.0 * forced_mask.mean())

    alpha_broadcast = np.broadcast_to(alpha_vals[:, None], boundary.shape)
    finite = np.isfinite(boundary)
    if finite.sum() > 1 and np.std(boundary[finite]) > 0:
        boundary_alpha_corr = float(
            np.corrcoef(boundary[finite], alpha_broadcast[finite])[0, 1]
        )
    else:
        boundary_alpha_corr = None

    # Bang-bang share, restricted to actual decision points: exercise dates,
    # and lattice nodes the process can actually reach (policy is meaningless
    # padding at unreachable nodes -- it's just its zero-initialised default).
    reachable = ~np.isnan(grid[ex_idx])  # (N+1, n_nodes)
    policy_ex = policy[ex_idx]  # (N+1, n_nodes, C_max+1)
    considered_mask = np.repeat(reachable[:, :, None], C_max + 1, axis=2)
    considered = policy_ex[considered_mask]
    pct_bang_bang = (
        float(100.0 * np.mean((considered == 0) | (considered == q_max)))
        if considered.size else None
    )

    stats = {
        "price": float(option_price),
        "pct_forced_volume": round(pct_forced_volume, 3),
        "forced_onset_mean_t": (
            None if forced_onset_mean_t is None else round(forced_onset_mean_t, 4)
        ),
        "pct_bang_bang": None if pct_bang_bang is None else round(pct_bang_bang, 3),
        "boundary_alpha_corr": (
            None if boundary_alpha_corr is None else round(boundary_alpha_corr, 4)
        ),
        "stationary_sd": round(float(stationary_sd), 6),
    }

    meta = {
        "N": N,
        "N_fine": N_fine,
        "K_sub": K_sub,
        "n_nodes": 2 * m + 1,
        "delta_t": float(delta_t),
        "delta_X": float(delta_X),
        "m": int(m),
        "col_X0": int(col_X0),
        "solve_seconds": round(time.time() - started, 3),
        "clip_fired": _count_clip_fired(m, delta_X, delta_t, kappa, sigma),
    }

    return {
        "boundary": boundary,
        "alpha": alpha_vals,
        "time_grid": time_grid,
        "stats": stats,
        "meta": meta,
    }
