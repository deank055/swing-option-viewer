"""Service layer.

Deliberately imports nothing from FastAPI. This is what lets the same code be
called from a notebook, a CLI, or a batch precompute script later.

The real solver plugs in at `_solve_boundary`. Everything else -- work budget,
decimation, JSON sanitising, caching -- stays as is.
"""

import time
from collections import OrderedDict
from typing import Any

import numpy as np

from .schema import SolveRequest

# Tune this against the slowest permitted config (T=5, n_max=150) until that
# lands under ~5 seconds on the deployed instance, not on your laptop.
WORK_BUDGET = 5.0e7

# Plotly gets sluggish past roughly 50k surface points, so cap the time axis.
MAX_SURFACE_TIME_POINTS = 252

_CACHE: "OrderedDict[str, dict]" = OrderedDict()
_CACHE_MAX = 128


def choose_k_sub(cfg: SolveRequest, price_nodes: int) -> int:
    """Adaptive sub-stepping so every config costs about the same wall clock."""
    per_step = price_nodes * (cfg.n_max + 1)
    k_sub = WORK_BUDGET / max(cfg.N * per_step, 1.0)
    return int(max(1, min(100, round(k_sub))))


def estimate_price_nodes(cfg: SolveRequest) -> int:
    """Trinomial node count saturates under mean reversion.

    The lattice is truncated at a fixed number of stationary standard
    deviations, so node count depends on kappa but not on horizon. This is
    exactly why kappa is floored away from zero: at kappa = 0 there is no
    stationary distribution and the tree grows without bound.
    """
    half_width = 6.0 * cfg.stationary_sd
    dx = cfg.sigma * (3.0 / (cfg.N * 20)) ** 0.5
    return int(min(2000, max(21, 2 * round(half_width / max(dx, 1e-9)) + 1)))


def alpha_curve(cfg: SolveRequest, t: np.ndarray) -> np.ndarray:
    p = cfg.preset
    return p["level"] * (1.0 + p["amp"] * np.cos(2.0 * np.pi * (t - p["phase"])))


def _solve_boundary(cfg: SolveRequest, k_sub: int) -> tuple[np.ndarray, dict]:
    """STUB. Replace with the real backward DP.

    Returns
    -------
    boundary : (N, n_max + 1) float array
        S*(i, n) in price units. np.nan where no finite threshold exists,
        which happens in the forced region and where exercise is never
        optimal. Those become JSON null downstream.
    stats : dict
    """
    N = cfg.N
    t = np.arange(N) / N * cfg.T
    alpha = alpha_curve(cfg, t)

    n_remaining = np.arange(cfg.n_max + 1)[None, :]
    dates_left = (N - np.arange(N))[:, None]

    # Premium above the seasonal level: falls as exercises accumulate,
    # widened by the stationary spread.
    scarcity = np.clip(1.0 - n_remaining / max(cfg.n_max, 1), 0.0, 1.0)
    premium = 0.9 * cfg.stationary_sd * (0.25 + scarcity)

    boundary = alpha[:, None] + premium + 0.15 * (cfg.K - cfg.preset["level"])

    # Forced region: not enough dates remain to reach n_min, so exercise is
    # mandatory and no threshold exists.
    min_still_needed = np.clip(cfg.n_min - (cfg.n_max - n_remaining), 0, None)
    boundary = np.where(min_still_needed > dates_left, np.nan, boundary)

    finite = np.isfinite(boundary)
    forced_share = float(1.0 - finite.mean())
    rows_forced = (~finite).any(axis=1)
    onset = float(t[rows_forced].mean()) if rows_forced.any() else float("nan")

    stats = {
        "price": float(np.nanmean(boundary) * cfg.q_max * cfg.n_max * 0.01),
        "pct_forced_states": round(100.0 * forced_share, 3),
        "forced_onset_mean_t": None if np.isnan(onset) else round(onset, 4),
        "boundary_alpha_corr": 0.93,
        "pct_bang_bang": 99.0,
        "stationary_sd": round(cfg.stationary_sd, 6),
        "moneyness": round(cfg.moneyness, 6),
        "STUB": True,
    }
    return boundary, stats


def _sanitise(arr: np.ndarray, decimals: int = 4) -> list:
    """NaN and inf are not valid JSON. Convert to null."""
    rounded = np.round(arr.astype(float), decimals)
    return [
        [None if not np.isfinite(v) else float(v) for v in row]
        for row in np.atleast_2d(rounded)
    ]


def solve(cfg: SolveRequest) -> dict[str, Any]:
    key = cfg.cache_key()
    if key in _CACHE:
        _CACHE.move_to_end(key)
        cached = dict(_CACHE[key])
        cached["meta"] = {**cached["meta"], "cached": True}
        return cached

    started = time.perf_counter()

    price_nodes = estimate_price_nodes(cfg)
    k_sub = choose_k_sub(cfg, price_nodes)
    boundary, stats = _solve_boundary(cfg, k_sub)

    N = cfg.N
    stride = max(1, N // MAX_SURFACE_TIME_POINTS)
    idx = np.arange(0, N, stride)

    t_full = np.arange(N) / N * cfg.T
    result = {
        "boundary": _sanitise(boundary[idx, :]),
        "time_grid": [round(float(v), 6) for v in t_full[idx]],
        "volume_grid": [float(n * cfg.q_max) for n in range(cfg.n_max + 1)],
        "alpha": [round(float(v), 6) for v in alpha_curve(cfg, t_full[idx])],
        "stats": stats,
        "meta": {
            "N": N,
            "k_sub": k_sub,
            "price_nodes": price_nodes,
            "time_stride": int(stride),
            "solve_seconds": round(time.perf_counter() - started, 3),
            "solver_version": "stub-0.1",
            "cached": False,
            "resolution_note": (
                f"Solved at K_sub={k_sub}, derived from a fixed work budget. "
                "Production thesis figures use K_sub=100."
            ),
        },
        "config": cfg.model_dump(),
    }

    _CACHE[key] = result
    while len(_CACHE) > _CACHE_MAX:
        _CACHE.popitem(last=False)

    return result
