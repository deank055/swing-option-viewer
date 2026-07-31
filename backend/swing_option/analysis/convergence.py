# this file checks for swing option price convergence in K_sub
# 1. self contained

import gc
import json
import os
import platform
import time

import numpy as np

from swing_option import config
from swing_option.core.lattice import build_price_grid
from swing_option.core.pricing import compute_admissible_actions, compute_option_value
from swing_option.core.memory import v_memory_gb, MEMORY_LIMIT_GB
from swing_option.analysis.optimal_strategy import extract_boundary, interpolate_boundary
from swing_option.analysis.clip_diagnostics import scan_columns

LADDER = [1, 4, 8, 16, 32, 64, 100, 128, 256]
PRODUCTION_K_SUB = 100
MEM_BUDGET_GB = MEMORY_LIMIT_GB

BASELINE_PARAMS = {
    "T": config.T,
    "N": config.N,
    "sigma": config.sigma,
    "r": config.r,
    "kappa": config.kappa,
    "S_0": config.S_0,
    "K": config.K,
    "C_max": config.C_max,
    "C_min": config.C_min,
    "q_max": config.q_max,
}

def _peak_rss_gb(): # peak memory used (need it for my windows and WSL machine)
    system = platform.system()
    if system == "Windows":
        import ctypes
        from ctypes import wintypes

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        handle = kernel32.GetCurrentProcess()
        if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return counters.PeakWorkingSetSize / 1e9

    import resource
    ru_maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return ru_maxrss * 1024 / 1e9

def _solve_resolution(K_sub, params):
    t0 = time.time()
    X_0 = params["S_0"] - config.alpha(0)
    grid, delta_t, delta_X, m, col_X0 = build_price_grid(
        T=params["T"],
        sigma=params["sigma"],
        X_0=X_0,
        N=params["N"],
        kappa=params["kappa"],
        K_sub=K_sub,
    )
    admissible = compute_admissible_actions(
        T=params["T"],
        N=params["N"],
        C_min=params["C_min"],
        C_max=params["C_max"],
        q_max=params["q_max"],
        K_sub=K_sub,
    )
    V, policy, option_price = compute_option_value(
        delta_t=delta_t,
        delta_X=delta_X,
        m=m,
        col_X0=col_X0,
        N=params["N"],
        admissible=admissible,
        K=params["K"],
        r=params["r"],
        kappa=params["kappa"],
        alpha_func=config.alpha,
        sigma=params["sigma"],
        C_max=params["C_max"],
        q_max=params["q_max"],K_sub=K_sub,
    )
    runtime_s = time.time() - t0

    boundary, _ = extract_boundary(grid, policy, admissible, config.alpha, delta_t)
    interp_data = {
        "grid": grid,
        "policy": policy,
        "V": V,
        "m": m,
        "delta_X": delta_X,
        "delta_t": delta_t
    }
    boundary_interp, boundary_diag = interpolate_boundary(
        interp_data,
        boundary,
        kappa=params["kappa"],
        sigma=params["sigma"],
        K=params["K"],
        r=params["r"],
        alpha_func=config.alpha,
    )
    boundary_interp[-1, :] = np.nan  # terminal row has no continuatiojn
    mean_boundary_C0 = float(np.nanmean(boundary_interp[:, 0]))

    peak_rss_gb = _peak_rss_gb()  # read while V is still alive

    del V, policy, admissible, grid, interp_data, boundary, boundary_interp
    gc.collect()

    return {
        "price": float(option_price),
        "mean_boundary_C0": mean_boundary_C0,
        "delta_t": float(delta_t),
        "delta_X": float(delta_X),
        "runtime_s": runtime_s,
        "fallback_fraction": boundary_diag["fallback_fraction"],
        "n_cells_considered": boundary_diag["n_cells_considered"],
        "n_fallback": boundary_diag["n_fallback"],
        "peak_rss_gb": peak_rss_gb,
    }

def _check_self_consistency(resolution_rows, reference):
    if reference is None:
        return None
    ref_K_sub, ref_price = reference
    match = next((r for r in resolution_rows if r["K_sub"] == ref_K_sub and r["status"] == "ok"), None)
    if match is None:
        return None

    diff = abs(match["price"] - ref_price)
    if diff > 1e-3:
        raise AssertionError(
            f"Self-consistency check FAILED: ladder price at K_sub={ref_K_sub} "
            f"({match['price']:.6f}) != reference price ({ref_price:.6f})"
        )
    return {
        "ref_K_sub": ref_K_sub,
        "ref_price": ref_price,
        "ladder_price": match["price"],
        "abs_diff": diff
    }

def _pow2_resolutions(resolution_rows):
    return [r for r in resolution_rows if r["status"] == "ok" and r["K_sub"] > 1 and (r["K_sub"] & (r["K_sub"] - 1)) == 0]

def _order_and_richardson(pow2_rows, value_key):
    if len(pow2_rows) < 3:
        return {"order_p": None, "richardson_estimate": None, "fit_K_subs": None, "note": f"fewer than 3 feasible power-of-2 resolutions available ({len(pow2_rows)})"}

    top3 = pow2_rows[-3:]
    p1, p2, p3 = (r[value_key] for r in top3)
    d1, d2 = p1 - p2, p2 - p3

    if d1 == 0 or d2 == 0 or (d1 / d2) <= 0:
        return {"order_p": None, "richardson_estimate": None, "fit_K_subs": [r["K_sub"] for r in top3], "note": "non-monotonic successive differences so order fit undefined"}

    order_p = float(np.log(d1 / d2) / np.log(2.0))
    richardson_estimate = float(p3 + (p3 - p2) / (2.0 ** order_p - 1.0))
    return {"order_p": order_p, "richardson_estimate": richardson_estimate, "fit_K_subs": [r["K_sub"] for r in top3]}

def _discretisation_errors(resolution_rows, value_key, richardson_estimate):
    if richardson_estimate is None:
        return None
    ok_rows = [r for r in resolution_rows if r["status"] == "ok"]
    if not ok_rows:
        return None
    row_1 = next((r for r in ok_rows if r["K_sub"] == 1), None)
    top_row = ok_rows[-1]

    def _err(row):
        if row is None:
            return None
        abs_err = row[value_key] - richardson_estimate
        pct_err = (abs_err / richardson_estimate * 100.0) if richardson_estimate != 0 else None
        return {"K_sub": row["K_sub"], "abs": float(abs_err), "pct": pct_err}

    return {"at_K_sub_1": _err(row_1), "at_top_feasible": _err(top_row)}

def _convergence_summary(resolution_rows, reference):
    self_consistency = _check_self_consistency(resolution_rows, reference)

    pow2_rows = _pow2_resolutions(resolution_rows)
    price_fit = _order_and_richardson(pow2_rows, "price")
    mb_fit = _order_and_richardson(pow2_rows, "mean_boundary_C0")

    price_errors = _discretisation_errors(resolution_rows, "price", price_fit["richardson_estimate"])
    mb_errors = _discretisation_errors(resolution_rows, "mean_boundary_C0", mb_fit["richardson_estimate"])

    return {
        "price_convergence": price_fit,
        "price_discretisation_error": price_errors,
        "mean_boundary_convergence": mb_fit,
        "mean_boundary_discretisation_error": mb_errors,
        "production_k_sub": PRODUCTION_K_SUB,
        "self_consistency": self_consistency,
    }

def _to_py(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, float):
        return None if (obj != obj) else obj
    if isinstance(obj, np.ndarray):
        return [_to_py(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {k: _to_py(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_py(x) for x in obj]
    return obj

def run_ladder(output_dir, mem_budget_gb, reference=None): # check K_sub ladder for convergence.
    params = BASELINE_PARAMS

    resolution_rows = []
    for K_sub in LADDER:
        predicted_gb = v_memory_gb(params, K_sub)

        if predicted_gb > mem_budget_gb:
            resolution_rows.append({
                "K_sub": K_sub, "N_fine": params["N"] * K_sub, "delta_X": None, "price": None,
                "clip_triggered": None, "clip_count": None, "max_clip_magnitude": None,
                "mean_boundary_C0": None, "runtime_s": None, "peak_rss_gb": None,
                "fallback_fraction": None, "n_cells_considered": None, "n_fallback": None,
                "predicted_gb": predicted_gb, "status": "skipped_oom",
            })
            continue

        solved = _solve_resolution(K_sub, params)
        clip = scan_columns(K_sub)
        clip_triggered = clip["n_triggered"] > 0
        max_clip_magnitude = abs(clip["min_raw_prob"]) if clip["min_raw_prob"] < 0 else 0.0

        resolution_rows.append({
            "K_sub": K_sub, "N_fine": params["N"] * K_sub, "delta_X": solved["delta_X"],
            "price": solved["price"], "clip_triggered": clip_triggered,
            "clip_count": clip["n_triggered"], "max_clip_magnitude": max_clip_magnitude,
            "mean_boundary_C0": solved["mean_boundary_C0"], "runtime_s": solved["runtime_s"],
            "peak_rss_gb": solved["peak_rss_gb"], "fallback_fraction": solved["fallback_fraction"],
            "n_cells_considered": solved["n_cells_considered"], "n_fallback": solved["n_fallback"],
            "predicted_gb": predicted_gb, "status": "ok",
        })

    summary = _convergence_summary(resolution_rows, reference)
    result = {"ladder": resolution_rows, "summary": summary}

    json_path = os.path.join(output_dir, "convergence_ladder.json")
    with open(json_path, "w") as f:
        json.dump(_to_py(result), f, indent=2)

    return result

def analyse(data, output_dir):
    reference = (data["K_sub"], data["option_price"])
    return run_ladder(output_dir, MEM_BUDGET_GB, reference=reference)
