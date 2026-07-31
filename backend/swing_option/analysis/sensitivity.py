# this file computes the parameter sensitivity sweep

import gc
import json
import os

import numpy as np
import matplotlib.pyplot as plt

from swing_option import config
from swing_option.core.lattice import build_price_grid
from swing_option.core.pricing import compute_admissible_actions, compute_option_value
from swing_option.core.memory import MEMORY_LIMIT_GB, find_feasible_ksub
from swing_option.analysis.optimal_strategy import (
    extract_boundary, interpolate_boundary, simulate_capacity_path, report_boundary_fallback,
)

K_SUB = 100

def _solve_at_ksub(params, K_sub):
    X_0 = params["S_0"] - config.alpha(0)
    kappa_v, sigma_v, K_v, r_v = params["kappa"], params["sigma"], params["K"], params["r"]

    grid, delta_t, delta_X, m, col_X0 = build_price_grid(
        T=params["T"], sigma=sigma_v, X_0=X_0, N=params["N"],
        kappa=kappa_v, K_sub=K_sub,
    )
    admissible = compute_admissible_actions(
        T=params["T"], N=params["N"], C_min=params["C_min"],
        C_max=params["C_max"], q_max=params["q_max"], K_sub=K_sub,
    )
    V, policy, option_price = compute_option_value(
        delta_t=delta_t, delta_X=delta_X, m=m, col_X0=col_X0, N=params["N"],
        admissible=admissible, K=K_v, r=r_v,
        kappa=kappa_v, alpha_func=config.alpha, sigma=sigma_v,
        C_max=params["C_max"], q_max=params["q_max"], K_sub=K_sub,
    )

    interp_data = {
        "grid": grid,
        "delta_t": delta_t,
        "delta_X": delta_X,
        "m": m,
        "col_X0": col_X0,
        "admissible": admissible,
        "V": V,
        "policy": policy,
        "K_sub": K_sub,
    }
    boundary_full, forced_mask_full = extract_boundary(
        grid, policy, admissible, config.alpha, delta_t,
    )

    boundary_interp_full, boundary_diag = interpolate_boundary(
        interp_data, boundary_full,
        kappa=kappa_v, sigma=sigma_v, K=K_v, r=r_v, alpha_func=config.alpha,
    )
    report_boundary_fallback(boundary_diag)
    boundary_full[-1, :] = np.nan
    boundary_interp_full[-1, :] = np.nan

    del interp_data["V"], V

    ex_idx = np.arange(0, grid.shape[0], K_sub)
    boundary_ex = boundary_interp_full[ex_idx]  # copy via fancy index
    forced_ex = forced_mask_full[ex_idx]
    t_ex = ex_idx * delta_t
    del boundary_interp_full, forced_mask_full, boundary_full

    sim_data = {
        "grid": grid, "delta_t": delta_t, "delta_X": delta_X,
        "m": m, "col_X0": col_X0, "policy": policy, "K_sub": K_sub,
    }

    return {
        "data": sim_data,
        "boundary": boundary_ex,
        "forced_mask": forced_ex,
        "t": t_ex,
        "delta_t": delta_t * K_sub,
        "ex_idx": ex_idx,
        "option_price": option_price,
        "params": params,
        "K_sub": K_sub,
        "fallback_fraction": boundary_diag["fallback_fraction"],
    }

def find_uniform_ksub(worst_overrides=None, target_K_sub=K_SUB):
    params = {
        "T": config.T, "N": config.N, "sigma": config.sigma, "r": config.r,
        "kappa": config.kappa, "S_0": config.S_0, "K": config.K,
        "C_max": config.C_max, "C_min": config.C_min, "q_max": config.q_max,
    }
    if worst_overrides:
        params.update(worst_overrides)
    k = find_feasible_ksub(params, target_K_sub, memory_limit_gb=MEMORY_LIMIT_GB)
    return k

def solve_for_exact_ksub(overrides=None, K_sub=K_SUB):
    overrides = overrides or {}
    params = {
        "T": config.T, "N": config.N, "sigma": config.sigma, "r": config.r,
        "kappa": config.kappa, "S_0": config.S_0, "K": config.K,
        "C_max": config.C_max, "C_min": config.C_min, "q_max": config.q_max,
    }
    params.update(overrides)
    return _solve_at_ksub(params, K_sub)

def mean_boundary(result, C):
    b = result["boundary"][:, C]
    return float(np.nanmean(b))

def anticipation_onset(result, K):
    boundary = result["boundary"]
    t = result["t"]
    onset = np.full(boundary.shape[1], np.nan)
    for C in range(boundary.shape[1]):
        rows = np.where(boundary[:, C] < K)[0]
        if rows.size:
            onset[C] = float(t[int(rows[0])])
    return onset

def mean_boundary_alpha_sub(result, C):
    b = result["boundary"][:, C]
    t = result["t"]
    a = config.alpha(t)
    ok = np.isfinite(b)
    if ok.sum() == 0:
        return float("nan")
    return float(np.mean(b[ok])) - float(np.mean(a[ok]))

def forced_onset(result, C=0):
    rows = np.where(result["forced_mask"][:, C])[0]
    if rows.size == 0:
        return None, None
    i = int(rows[0])
    return i, float(result["t"][i])

def simulate_cbar(result, num_paths=250_000, seed=1337):
    params = result["params"]
    Cbar_full = simulate_capacity_path(result["data"], kappa=params["kappa"], sigma=params["sigma"], num_paths=num_paths, seed=seed)
    return Cbar_full[result["ex_idx"]]

def boundary_along_cbar(result, Cbar_ex):
    C_max = result["boundary"].shape[1] - 1
    idx = np.clip(np.round(Cbar_ex).astype(int), 0, C_max)
    N = len(result["t"])
    return np.array([result["boundary"][i, idx[i]] for i in range(N)])

def meanboundary_vs_C(result):
    with np.errstate(all="ignore"):
        return np.nanmean(result["boundary"], axis=0)

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

def plot_kappa_boundary_cbar(kappa_series, output_path):
    n = len(kappa_series)
    cmap = plt.get_cmap("viridis", max(n, 2))

    fig, ax = plt.subplots(figsize=(9, 6))

    for idx, (kappa_val, t, S_star) in enumerate(kappa_series):
        is_base = np.isclose(kappa_val, config.kappa)
        color = "black" if is_base else cmap(idx / max(n - 1, 1))
        lw = 2.5 if is_base else 1.4
        label = f"kappa = {kappa_val:g}" + (" (baseline)" if is_base else "")
        ax.plot(t, S_star, color=color, linewidth=lw, label=label)

    t_base = kappa_series[0][1]
    ax.plot(t_base, config.alpha(t_base), color="dimgray", linestyle=":", linewidth=1.5, label="alpha(t)")
    ax.axhline(config.K, color="dimgray", linestyle="--", linewidth=1, label=f"K = {config.K:.2f}")

    ax.set_xlabel("t (years)")
    ax.set_ylabel("Exercise boundary S* (USD/MMBtu)")
    ax.set_ylim(0, 6)
    ax.legend(loc="best", fontsize=7, ncol=2)

    fig.savefig(output_path, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)

def plot_kappa_meanboundary_vs_C(kappa_series, output_path):
    n = len(kappa_series)
    cmap = plt.get_cmap("viridis", max(n, 2))

    fig, ax = plt.subplots(figsize=(9, 6))

    for idx, (kappa_val, mb_C) in enumerate(kappa_series):
        C_arr = np.arange(len(mb_C))
        is_base = np.isclose(kappa_val, config.kappa)
        color = "black" if is_base else cmap(idx / max(n - 1, 1))
        lw = 2.5 if is_base else 1.4
        label = f"kappa = {kappa_val:g}" + (" (baseline)" if is_base else "")
        ax.plot(C_arr, mb_C, color=color, linewidth=lw, label=label)

    ax.axhline(config.K, color="dimgray", linestyle="--", linewidth=1, label=f"K = {config.K:.2f}")
    ax.set_xlabel("Cumulative exercised volume C")
    ax.set_ylabel("Mean boundary S* over exercise days (USD/MMBtu)")
    ax.legend(loc="best", fontsize=7, ncol=2)

    fig.savefig(output_path, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)

def plot_cmin_forced_onset(cmin_data, output_path):
    N_val = config.N
    q_max = config.q_max
    cmin_vals = [d["C_min"] for d in cmin_data]
    t_dp = [d["forced_onset_t"] for d in cmin_data]
    # Analytic forced onset i = ceil(N - C_min/q_max), t = i / N
    t_analytic = [float(np.ceil(N_val - d["C_min"] / q_max)) / N_val for d in cmin_data]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(cmin_vals, t_dp, "o-", linewidth=1.8, label="DP forced onset")
    ax.plot(cmin_vals, t_analytic, "s--", linewidth=1.4, label="Analytic: ceil(N - C_min/q_max) / N")
    ax.set_xlabel("C_min")
    ax.set_ylabel("Forced onset t (years)")
    ax.legend()

    fig.savefig(output_path, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)

KAPPA_GRID = [1.5, 5.0, 10.5, 20.0, config.kappa, 60.0]
SIGMA_GRID = [2.5, 7.5, config.sigma, 17.5, 25.0]
CMIN_GRID = [0, 200, 525, 700, 850, 1050, 1200]
K_GRID = [2.40, config.K, 4.40]

C_SLICE = 1050

KAPPA0 = config.kappa
SIGMA0 = config.sigma
CMIN0 = config.C_min
K0 = config.K
QMAX0 = config.q_max

def analyse(data, output_dir):
    N = config.N
    K_sub_target = data.get("K_sub", K_SUB)
    plots_dir = os.path.join(output_dir, "plots")

    output = {}

    def _boundary_summary(res):
        K_res = res["params"]["K"]
        mb = mean_boundary(res, 0)
        mb1050 = mean_boundary(res, C_SLICE)
        onset = anticipation_onset(res, K_res)
        return K_res, mb, mb1050, onset

    uniform_ksub = find_uniform_ksub({"kappa": 1.5}, target_K_sub=K_sub_target)

    kappa_sweep_data = []
    mb_kappa = {}
    mb1050_kappa = {}
    mb_base_uniform = None
    mb1050_base_uniform = None
    t_dagger_base_uniform = None
    mb_105 = None
    mb1050_105 = None
    fig1_series = []
    fig2_series = []
    fig1_data = {}
    fig2_data = {"C": list(range(config.C_max + 1))}
    appendix_data = {}

    for kappa in KAPPA_GRID:
        res = solve_for_exact_ksub({"kappa": kappa}, K_sub=uniform_ksub)

        _, mb, mb1050, onset = _boundary_summary(res)
        mba = mean_boundary_alpha_sub(res, 0)
        mb_C = meanboundary_vs_C(res)

        kappa_sweep_data.append({
            "kappa": kappa,
            "mean_boundary_C0": mb,
            "mean_boundary_C1050": mb1050,
            "t_dagger_C0": float(onset[0]) if np.isfinite(onset[0]) else None,
            "mean_boundary_alpha_sub_C0": mba,
            "option_price": res["option_price"],
            "K_sub_used": uniform_ksub,
            "fallback_fraction": res["fallback_fraction"],
        })
        mb_kappa[f"{kappa:g}"] = mb
        mb1050_kappa[f"{kappa:g}"] = mb1050

        if np.isclose(kappa, KAPPA0):
            mb_base_uniform = mb
            mb1050_base_uniform = mb1050
            t_dagger_base_uniform = float(onset[0]) if np.isfinite(onset[0]) else None
        if np.isclose(kappa, 10.5):
            mb_105 = mb
            mb1050_105 = mb1050

        Cbar = simulate_cbar(res, num_paths=250_000, seed=1337)
        S_star = boundary_along_cbar(res, Cbar)
        t = res["t"]
        fig1_series.append((kappa, t.copy(), S_star.copy()))
        fig1_data[f"{kappa:g}"] = {
            "t": _to_py(t), "Cbar": _to_py(Cbar), "S_star": _to_py(S_star),
        }

        fig2_series.append((kappa, mb_C.copy()))
        fig2_data[f"kappa_{kappa:g}"] = _to_py(mb_C)

        # Free the large result before solving the next kappa
        del res
        gc.collect()

    output["kappa_sweep"] = _to_py(kappa_sweep_data)

    spike_diff = (abs(mb_105 - mb_base_uniform) if (mb_105 is not None and mb_base_uniform is not None) else None)
    spike_diff_C1050 = (abs(mb1050_105 - mb1050_base_uniform) if (mb1050_105 is not None and mb1050_base_uniform is not None) else None)
    range_diff = (abs(mb_kappa.get("1.5", float("nan")) - mb_base_uniform) if mb_base_uniform is not None else None)
    range_diff_C1050 = (
        abs(mb1050_kappa.get(f"{KAPPA_GRID[0]:g}", float("nan")) - mb1050_base_uniform)
        if mb1050_base_uniform is not None else None
    )
    output["kappa_report"] = _to_py({
        "uniform_K_sub": uniform_ksub,
        "mb_1.5": mb_kappa.get(f"{KAPPA_GRID[0]:g}"),
        "mb_baseline": mb_base_uniform,
        "mb_C1050_1.5": mb1050_kappa.get(f"{KAPPA_GRID[0]:g}"),
        "mb_C1050_baseline": mb1050_base_uniform,
        "t_dagger_baseline": t_dagger_base_uniform,
        "mb_10.5_spike_excl": mb_105,
        "mb_C1050_10.5_spike_excl": mb1050_105,
        "mb_high": mb_kappa.get(f"{KAPPA_GRID[-1]:g}"),
        "mb_C1050_high": mb1050_kappa.get(f"{KAPPA_GRID[-1]:g}"),
        "spike_exclusion_neutralisation_diff": spike_diff,
        "spike_exclusion_neutralisation_diff_C1050": spike_diff_C1050,
        "range_diff_1.5_to_baseline": range_diff,
        "kappa_swing_1.5_vs_baseline_C1050": range_diff_C1050,
    })
    output["uniform_K_sub"] = uniform_ksub

    # Write JSON outputs
    with open(os.path.join(output_dir, "sensitivity_kappa_boundary_along_cbar.json"), "w") as f:
        json.dump(fig1_data, f, indent=2)
    output["fig1_json"] = "sensitivity_kappa_boundary_along_cbar.json"

    with open(os.path.join(output_dir, "sensitivity_kappa_meanboundary_vs_C.json"), "w") as f:
        json.dump(fig2_data, f, indent=2)
    output["fig2_json"] = "sensitivity_kappa_meanboundary_vs_C.json"

    with open(os.path.join(output_dir, "sensitivity_kappa_samples_appendix.json"), "w") as f:
        json.dump(appendix_data, f, indent=2)
    output["appendix_json"] = "sensitivity_kappa_samples_appendix.json"

    fig1_path = os.path.join(plots_dir, "sensitivity_kappa_boundary_cbar.png")
    plot_kappa_boundary_cbar(fig1_series, fig1_path)

    fig2_path = os.path.join(plots_dir, "sensitivity_kappa_meanboundary_vs_C.png")
    plot_kappa_meanboundary_vs_C(fig2_series, fig2_path)

    sigma_sweep_data = []
    mb_sigma = {}
    mb1050_sigma = {}
    mb_base_K32 = None
    mb1050_base_K32 = None
    t_dagger_base_K32 = None
    onset_i10_K32 = None
    onset_t10_K32 = None
    fallback_base_K32 = None

    for sigma in SIGMA_GRID:
        res = solve_for_exact_ksub({"sigma": sigma}, K_sub=uniform_ksub)

        _, mb, mb1050, onset = _boundary_summary(res)
        mba = mean_boundary_alpha_sub(res, 0)
        sigma_sweep_data.append({
            "sigma": sigma,
            "mean_boundary_C0": mb,
            "mean_boundary_C1050": mb1050,
            "t_dagger_C0": float(onset[0]) if np.isfinite(onset[0]) else None,
            "mean_boundary_alpha_sub_C0": mba,
            "option_price": res["option_price"],
            "fallback_fraction": res["fallback_fraction"],
        })
        mb_sigma[f"{sigma:g}"] = mb
        mb1050_sigma[f"{sigma:g}"] = mb1050

        if np.isclose(sigma, SIGMA0):
            mb_base_K32 = mb
            mb1050_base_K32 = mb1050
            t_dagger_base_K32 = float(onset[0]) if np.isfinite(onset[0]) else None
            onset_i10_K32, onset_t10_K32 = forced_onset(res, C=0)
            fallback_base_K32 = res["fallback_fraction"]

        del res
        gc.collect()

    output["sigma_sweep"] = _to_py(sigma_sweep_data)
    output["sigma_report"] = _to_py({
        "mb_lo": mb_sigma.get(f"{SIGMA_GRID[0]:g}"),
        "mb_baseline": mb_sigma.get(f"{SIGMA0:g}"),
        "mb_hi": mb_sigma.get(f"{SIGMA_GRID[-1]:g}"),
        "mb_C1050_lo": mb1050_sigma.get(f"{SIGMA_GRID[0]:g}"),
        "mb_C1050_baseline": mb1050_base_K32,
        "mb_C1050_hi": mb1050_sigma.get(f"{SIGMA_GRID[-1]:g}"),
        "t_dagger_baseline": t_dagger_base_K32,
    })

    kappa_lo = 0.9 * KAPPA0
    kappa_hi = 1.1 * KAPPA0
    sigma_lo = 0.9 * SIGMA0
    sigma_hi = 1.1 * SIGMA0

    res_klo = solve_for_exact_ksub({"kappa": kappa_lo}, K_sub=uniform_ksub)
    _, mb_klo, mb1050_klo, onset_klo = _boundary_summary(res_klo)
    del res_klo; gc.collect()

    res_khi = solve_for_exact_ksub({"kappa": kappa_hi}, K_sub=uniform_ksub)
    _, mb_khi, mb1050_khi, onset_khi = _boundary_summary(res_khi)
    del res_khi; gc.collect()

    res_slo = solve_for_exact_ksub({"sigma": sigma_lo}, K_sub=uniform_ksub)
    _, mb_slo, mb1050_slo, onset_slo = _boundary_summary(res_slo)
    del res_slo; gc.collect()

    res_shi = solve_for_exact_ksub({"sigma": sigma_hi}, K_sub=uniform_ksub)
    _, mb_shi, mb1050_shi, onset_shi = _boundary_summary(res_shi)
    del res_shi; gc.collect()

    d_kappa = (mb_khi - mb_klo) / 2.0
    d_sigma = (mb_shi - mb_slo) / 2.0
    d_kappa_C1050 = (mb1050_khi - mb1050_klo) / 2.0
    d_sigma_C1050 = (mb1050_shi - mb1050_slo) / 2.0

    output["local_slopes"] = _to_py({
        "kappa_lo": kappa_lo, "kappa_hi": kappa_hi,
        "mb_kappa_lo": mb_klo,
        "mb_kappa_hi": mb_khi,
        "d_kappa_10pct": d_kappa,
        "mb_C1050_kappa_lo": mb1050_klo,
        "mb_C1050_kappa_hi": mb1050_khi,
        "t_dagger_kappa_lo": float(onset_klo[0]) if np.isfinite(onset_klo[0]) else None,
        "t_dagger_kappa_hi": float(onset_khi[0]) if np.isfinite(onset_khi[0]) else None,
        "d_kappa_10pct_C1050": d_kappa_C1050,
        "sigma_lo": sigma_lo, "sigma_hi": sigma_hi,
        "mb_sigma_lo": mb_slo,
        "mb_sigma_hi": mb_shi,
        "d_sigma_10pct": d_sigma,
        "mb_C1050_sigma_lo": mb1050_slo,
        "mb_C1050_sigma_hi": mb1050_shi,
        "t_dagger_sigma_lo": float(onset_slo[0]) if np.isfinite(onset_slo[0]) else None,
        "t_dagger_sigma_hi": float(onset_shi[0]) if np.isfinite(onset_shi[0]) else None,
        "d_sigma_10pct_C1050": d_sigma_C1050,
    })

    cmin_sweep_data = []

    for cmin in CMIN_GRID:
        res = solve_for_exact_ksub({"C_min": cmin}, K_sub=uniform_ksub)

        _, mb, mb1050, onset = _boundary_summary(res)
        onset_i, onset_t = forced_onset(res, C=0)
        forced_at_slice = bool(np.any(res["forced_mask"][:, C_SLICE]))
        analytic_i = N - cmin / QMAX0
        cmin_sweep_data.append({
            "C_min": cmin,
            "forced_onset_i": onset_i,
            "forced_onset_t": onset_t,
            "analytic_last_voluntary_i": analytic_i,
            "mean_boundary_C0": mb,
            "mean_boundary_C1050": mb1050,
            "t_dagger_C0": float(onset[0]) if np.isfinite(onset[0]) else None,
            "forced_at_C1050": forced_at_slice,
            "fallback_fraction": res["fallback_fraction"],
        })

        del res
        gc.collect()

    output["cmin_sweep"] = _to_py(cmin_sweep_data)

    finite_cmin_C0 = [row["mean_boundary_C0"] for row in cmin_sweep_data if row["mean_boundary_C0"] is not None]
    finite_cmin_C1050 = [row["mean_boundary_C1050"] for row in cmin_sweep_data if row["mean_boundary_C1050"] is not None]
    output["cmin_report"] = _to_py({
        "max_spread_C0": float(max(finite_cmin_C0) - min(finite_cmin_C0)) if len(finite_cmin_C0) > 1 else None,
        "max_spread_C1050": float(max(finite_cmin_C1050) - min(finite_cmin_C1050)) if len(finite_cmin_C1050) > 1 else None,
        "any_forced_at_C1050": any(row["forced_at_C1050"] for row in cmin_sweep_data),
    })

    cmin_plot_path = os.path.join(plots_dir, "sensitivity_cmin_forced_onset.png")
    plot_cmin_forced_onset(cmin_sweep_data, cmin_plot_path)

    k_boundaries = []  # [(K, boundary[:,0], boundary[:,C_SLICE], mb, mb1050, t_dagger_C0, option_price, fallback_fraction)]
    mb_base_K = None
    mb1050_base_K = None

    for K in K_GRID:
        res = solve_for_exact_ksub({"K": K}, K_sub=uniform_ksub)
        b0 = res["boundary"][:, 0].copy()
        b1050 = res["boundary"][:, C_SLICE].copy()
        _, mb, mb1050, onset = _boundary_summary(res)
        price = res["option_price"]
        fallback = res["fallback_fraction"]
        k_boundaries.append((K, b0, b1050, mb, mb1050, float(onset[0]) if np.isfinite(onset[0]) else None, price, fallback))
        if np.isclose(K, K0):
            mb_base_K = mb
            mb1050_base_K = mb1050
        del res
        gc.collect()

    b_itm = next(b for K, b, _, _, _, _, _, _ in k_boundaries if np.isclose(K, 2.40))
    b_base = next(b for K, b, _, _, _, _, _, _ in k_boundaries if np.isclose(K, K0))
    b_otm = next(b for K, b, _, _, _, _, _, _ in k_boundaries if np.isclose(K, 4.40))

    common_mask = np.isfinite(b_itm) & np.isfinite(b_base) & np.isfinite(b_otm)
    n_common = int(common_mask.sum())

    if n_common > 0:
        dSdK_itm = float(np.mean(
            (b_itm[common_mask] - b_base[common_mask]) / (2.40 - K0)
        ))
        dSdK_otm = float(np.mean(
            (b_otm[common_mask] - b_base[common_mask]) / (4.40 - K0)
        ))
    else:
        dSdK_itm = dSdK_otm = None

    b1050_itm = next(b for K, _, b, _, _, _, _, _ in k_boundaries if np.isclose(K, 2.40))
    b1050_base = next(b for K, _, b, _, _, _, _, _ in k_boundaries if np.isclose(K, K0))
    b1050_otm = next(b for K, _, b, _, _, _, _, _ in k_boundaries if np.isclose(K, 4.40))

    common_mask_C1050 = np.isfinite(b1050_itm) & np.isfinite(b1050_base) & np.isfinite(b1050_otm)
    n_common_C1050 = int(common_mask_C1050.sum())

    if n_common_C1050 > 0:
        dSdK_itm_C1050 = float(np.mean(
            (b1050_itm[common_mask_C1050] - b1050_base[common_mask_C1050]) / (2.40 - K0)
        ))
        dSdK_otm_C1050 = float(np.mean(
            (b1050_otm[common_mask_C1050] - b1050_base[common_mask_C1050]) / (4.40 - K0)
        ))
    else:
        dSdK_itm_C1050 = dSdK_otm_C1050 = None

    k_sweep_data = []
    for K, _, _, mb, mb1050, onset, price, fallback in k_boundaries:
        naive_C0 = None if (mb_base_K is None or np.isclose(K, K0)) else (mb - mb_base_K) / (K - K0)
        naive_C1050 = None if (mb1050_base_K is None or np.isclose(K, K0)) else (mb1050 - mb1050_base_K) / (K - K0)
        k_sweep_data.append({
            "K": K, "mean_boundary_C0": mb, "mean_boundary_C1050": mb1050, "t_dagger_C0": onset,
            "option_price": price,
            "dS_star_dK_naive_C0": naive_C0, "dS_star_dK_naive_C1050": naive_C1050,
            "fallback_fraction": fallback,
        })
    output["strike_sweep"] = _to_py(k_sweep_data)
    output["strike_report"] = _to_py({
        "n_common_voluntary_dates_C0": n_common,
        "dS_star_dK_ITM_arm_C0": dSdK_itm,
        "dS_star_dK_OTM_arm_C0": dSdK_otm,
        "n_common_voluntary_dates_C1050": n_common_C1050,
        "dS_star_dK_ITM_arm_C1050": dSdK_itm_C1050,
        "dS_star_dK_OTM_arm_C1050": dSdK_otm_C1050,
        "headline_dS_star_dK_C0": dSdK_otm,
        "headline_dS_star_dK_C1050": dSdK_otm_C1050,
    })

    res_qmax20 = solve_for_exact_ksub({"q_max": 20}, K_sub=uniform_ksub)
    _, mb_q20, mb1050_q20, onset_q20 = _boundary_summary(res_qmax20)
    onset_i20, onset_t20 = forced_onset(res_qmax20, C=0)
    analytic_i20 = N - CMIN0 / 20.0
    price_qmax20 = res_qmax20["option_price"]
    fallback_qmax20 = res_qmax20["fallback_fraction"]
    del res_qmax20
    gc.collect()

    output["qmax_check"] = _to_py({
        "qmax10": {
            "q_max": 10,
            "Cmin_over_qmax": CMIN0 / 10,
            "Cmax_over_N_qmax": config.C_max / (N * 10),
            "forced_onset_i": onset_i10_K32,
            "forced_onset_t": onset_t10_K32,
            "mean_boundary_C0": mb_base_K32,
            "mean_boundary_C1050": mb1050_base_K32,
            "t_dagger_C0": t_dagger_base_K32,
            "fallback_fraction": fallback_base_K32,
        },
        "qmax20": {
            "q_max": 20,
            "Cmin_over_qmax": CMIN0 / 20.0,
            "Cmax_over_N_qmax": config.C_max / (N * 20),
            "forced_onset_i": onset_i20,
            "forced_onset_t": onset_t20,
            "analytic_last_voluntary_i": analytic_i20,
            "mean_boundary_C0": mb_q20,
            "mean_boundary_C1050": mb1050_q20,
            "t_dagger_C0": float(onset_q20[0]) if np.isfinite(onset_q20[0]) else None,
            "option_price": price_qmax20,
            "fallback_fraction": fallback_qmax20,
        },
    })

    json_path = os.path.join(output_dir, "sensitivity_values.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)

    return output
