"""Service layer.

Deliberately imports nothing from FastAPI. This is what lets the same code be
called from a notebook, a CLI, or a batch precompute script later.

Wires the API layer to boundary.compute_boundary(), the thesis solver. Work
budget, decimation, JSON sanitising, and caching are app-layer concerns and
stay here.
"""

import os
import time
from collections import OrderedDict
from typing import Any

import numpy as np

from .boundary import compute_boundary
from .schema import S_0, SolveRequest

# Tune this against the slowest permitted config (T=5, max_exercises=1260,
# kappa at its floor) until that lands under ~5 seconds on the deployed
# instance, not on your laptop.
WORK_BUDGET = 5.0e7

# V is float32 and policy is int8 over the same (N*K_sub+1, n_nodes,
# max_exercises+1) shape, so together they cost about 5 bytes per element.
# choose_k_sub() caps K_sub by this in addition to WORK_BUDGET so a config
# that's cheap in wall-clock terms can't still blow the container's memory.
# Read from an env var (Cloud Run memory limits vary by deploy) with a
# conservative default; kept here rather than in swing_option.core.memory
# since that module's MEMORY_LIMIT_GB is about the thesis's own experiment
# budget, not about what fits in this container.
MEMORY_LIMIT_BYTES = int(os.environ.get("SOLVE_MEMORY_LIMIT_MB", "512")) * 1_000_000
BYTES_PER_ELEMENT = 5

# Plotly gets sluggish past roughly 50k surface points, so cap the time axis.
MAX_SURFACE_TIME_POINTS = 252

_CACHE: "OrderedDict[str, dict]" = OrderedDict()
_CACHE_MAX = 128


def choose_k_sub(cfg: SolveRequest, price_nodes: int) -> int:
    """Adaptive sub-stepping so every config costs about the same wall clock,
    capped so V + policy also fit within the memory ceiling."""
    per_step = price_nodes * (cfg.max_exercises + 1)
    k_sub_budget = WORK_BUDGET / max(cfg.N * per_step, 1.0)
    k_sub_memory = MEMORY_LIMIT_BYTES / max(BYTES_PER_ELEMENT * cfg.N * per_step, 1.0)
    k_sub = min(k_sub_budget, k_sub_memory)
    return int(max(1, min(100, round(k_sub))))


def check_budget(cfg: SolveRequest) -> None:
    """Reject configs that exceed the work budget even at K_sub = 1.

    choose_k_sub() adapts K_sub down to keep wall clock roughly constant, but
    it floors at 1 -- past that point there is no more slack to trade away,
    and a config that's still over budget at the floor has to be refused
    rather than silently solved at a coarser (wrong) resolution.
    """
    price_nodes = estimate_price_nodes(cfg)
    cost_at_floor = cfg.N * 1 * price_nodes * (cfg.max_exercises + 1)
    if cost_at_floor > WORK_BUDGET:
        overshoot = cost_at_floor / WORK_BUDGET
        raise ValueError(
            f"This combination exceeds the solver budget: horizon {cfg.T:g}y "
            f"with {cfg.max_exercises} exercises at kappa={cfg.kappa:g} costs "
            f"roughly {overshoot:.0f}x the limit. Reduce the horizon, the "
            "exercise count, or increase mean reversion."
        )


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


def _alpha_func(cfg: SolveRequest):
    """Seasonal level function for this config's preset.

    Wrapped in np.asarray so it works both as a scalar call --
    compute_option_value calls alpha_func(i * delta_t) -- and as an array
    call -- compute_boundary calls alpha_func(time_grid).
    """
    p = cfg.preset

    def alpha(t):
        return p["level"] * (1.0 + p["amp"] * np.cos(2.0 * np.pi * (np.asarray(t) - p["phase"])))

    return alpha


def _sanitise(arr: np.ndarray, decimals: int = 4) -> list:
    """NaN and inf are not valid JSON. Convert to null."""
    rounded = np.round(arr.astype(float), decimals)
    return [
        [None if not np.isfinite(v) else float(v) for v in row]
        for row in np.atleast_2d(rounded)
    ]


def solve(cfg: SolveRequest) -> dict[str, Any]:
    check_budget(cfg)

    key = cfg.cache_key()
    if key in _CACHE:
        _CACHE.move_to_end(key)
        cached = dict(_CACHE[key])
        cached["meta"] = {**cached["meta"], "cached": True}
        return cached

    started = time.perf_counter()

    price_nodes = estimate_price_nodes(cfg)
    k_sub = choose_k_sub(cfg, price_nodes)

    # Capacity in exercise counts, not volume: the arithmetic payoff is
    # affine, so S*(i, C) is invariant to a common rescaling of q_max and the
    # capacity bounds. Solve at q_max = 1 and rescale price below; q_max is a
    # display label only from here on.
    solved = compute_boundary(
        T=cfg.T, N=cfg.N, sigma=cfg.sigma, kappa=cfg.kappa, r=cfg.r, K=cfg.K,
        S_0=S_0, alpha_func=_alpha_func(cfg),
        C_max=cfg.max_exercises,
        C_min=cfg.min_exercises,
        q_max=1,
        K_sub=k_sub,
    )

    boundary = solved["boundary"]
    time_grid = solved["time_grid"]
    alpha_vals = solved["alpha"]
    solver_stats = solved["stats"]

    # compute_boundary returns N + 1 dates spanning [0, T] inclusive (the
    # stub returned N, stopping short of T); stride all three of boundary,
    # time_grid and alpha together off the solver's own arrays so they stay
    # aligned, rather than rebuilding a separate time axis here.
    total_dates = cfg.N + 1
    stride = max(1, total_dates // MAX_SURFACE_TIME_POINTS)
    idx = np.arange(0, total_dates, stride)

    stats = {
        "price": round(float(solver_stats["price"]) * cfg.q_max, 4),
        "pct_forced_states": solver_stats["pct_forced_volume"],
        "forced_onset_mean_t": solver_stats["forced_onset_mean_t"],
        "boundary_alpha_corr": solver_stats["boundary_alpha_corr"],
        "pct_bang_bang": solver_stats["pct_bang_bang"],
        "stationary_sd": solver_stats["stationary_sd"],
        "moneyness": round(cfg.moneyness, 6),
    }

    result = {
        "boundary": _sanitise(boundary[idx, :]),
        "time_grid": [round(float(v), 6) for v in time_grid[idx]],
        "volume_grid": [float(n * cfg.q_max) for n in range(cfg.max_exercises + 1)],
        "alpha": [round(float(v), 6) for v in alpha_vals[idx]],
        "stats": stats,
        "meta": {
            "N": cfg.N,
            "k_sub": k_sub,
            "price_nodes": price_nodes,
            "time_stride": int(stride),
            "solve_seconds": round(time.perf_counter() - started, 3),
            "solver_version": "thesis-1.0",
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
