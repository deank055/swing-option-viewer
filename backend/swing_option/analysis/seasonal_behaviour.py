# this file generates the plots and figures for the seasoal section
# 1. mostly copy pasted from optimal_strategy.py so a bit messy
# 2. ad hoc

import json
import os

import numpy as np
import matplotlib.pyplot as plt

from swing_option.config import (
    alpha as _alpha_cfg, K as _K_cfg, N as _N_cfg, T as _T_cfg,
    C_min as _C_min_cfg, q_max as _q_max_cfg,
)
from swing_option.core.lattice import transition_probabilities
from swing_option.config import kappa, sigma

def simulate_exercise_events(data, num_paths=250_000, seed=11):
    grid = data["grid"]
    policy = data["policy"]
    m = data["m"]
    col_X0 = data["col_X0"]
    delta_t = data["delta_t"]
    delta_X = data["delta_X"]
    admissible = data["admissible"]  # (N_fine+1, C_max+1, q_max+1) bool
    K_sub = data.get("K_sub", 1)
    K_val = data.get("K", _K_cfg)
    C_min_val = data.get("C_min", _C_min_cfg)
    q_max_val = data.get("q_max", _q_max_cfg)

    N1 = grid.shape[0]
    N_steps = N1 - 1
    N_fine = N_steps  # = N * K_sub
    n_nodes = 2 * m + 1
    C_max_v = policy.shape[2] - 1

    idx = np.arange(n_nodes)
    is_top = idx == n_nodes - 1
    is_bot = idx == 0
    j_up = np.where(is_top, idx, np.where(is_bot, idx + 2, idx + 1))
    j_mid = np.where(is_top, idx - 1, np.where(is_bot, idx + 1, idx))
    j_dn = np.where(is_top, idx - 2, np.where(is_bot, idx, idx - 1))

    pu_arr = np.empty(n_nodes)
    pm_arr = np.empty(n_nodes)
    pd_arr = np.empty(n_nodes)
    for j in range(n_nodes):
        X_j = (j - m) * delta_X
        pu_arr[j], pm_arr[j], pd_arr[j] = transition_probabilities(
            S=X_j, delta_t=delta_t, delta_S=delta_X,
            kappa=kappa, Theta_t=0.0, sigma=sigma,
            is_top=is_top[j], is_bottom=is_bot[j],
        )

    rng = np.random.default_rng(seed)
    cols = np.full(num_paths, col_X0, dtype=int)
    C = np.zeros(num_paths, dtype=int)

    ev_path = []
    ev_i = []
    ev_i_ex = []
    ev_t = []
    ev_q = []
    ev_Cbef = []
    ev_S = []
    ev_mono = []
    ev_forced = []

    for i in range(N_steps):
        actions = policy[i, cols, C]  # (num_paths,)
        exercising = actions > 0

        if exercising.any():
            t_i = i * delta_t
            alpha_i = _alpha_cfg(t_i)
            i_ex_i = i // K_sub

            ex_paths = np.where(exercising)[0]
            S_vals = alpha_i + (cols[exercising] - m) * delta_X
            C_bef = C[exercising]
            q_at = actions[exercising]
            remaining_days = (N_fine - i) // K_sub
            forced_at = (C_bef.astype(np.int64) + remaining_days * q_max_val) <= C_min_val

            ev_path.extend(ex_paths.tolist())
            ev_i.extend(np.full(len(ex_paths), i, dtype=np.int32).tolist())
            ev_i_ex.extend(np.full(len(ex_paths), i_ex_i, dtype=np.int32).tolist())
            ev_t.extend(np.full(len(ex_paths), t_i).tolist())
            ev_q.extend(q_at.tolist())
            ev_Cbef.extend(C_bef.tolist())
            ev_S.extend(S_vals.tolist())
            ev_mono.extend((S_vals - K_val).tolist())
            ev_forced.extend(forced_at.tolist())

        C = np.minimum(C + actions, C_max_v)
        u = rng.random(num_paths)
        pd_p = pd_arr[cols]
        pm_p = pm_arr[cols]
        cols = np.where(u < pd_p, j_dn[cols], np.where(u < pd_p + pm_p, j_mid[cols], j_up[cols]))

    return {
        "path_idx": np.asarray(ev_path, dtype=np.int32),
        "i": np.asarray(ev_i, dtype=np.int32),
        "i_ex": np.asarray(ev_i_ex, dtype=np.int32),
        "t": np.asarray(ev_t, dtype=np.float64),
        "q": np.asarray(ev_q, dtype=np.int16),
        "C_before": np.asarray(ev_Cbef, dtype=np.int16),
        "S": np.asarray(ev_S, dtype=np.float32),
        "moneyness": np.asarray(ev_mono, dtype=np.float32),
        "forced": np.asarray(ev_forced, dtype=bool),
    }

def compute_seasonal_stats(events, K_sub=100, K_val=None, alpha_func=None, N=None, T=None):
    if K_val is None: K_val = _K_cfg
    if alpha_func is None: alpha_func = _alpha_cfg
    if N is None: N = _N_cfg
    if T is None: T = _T_cfg

    q = events["q"].astype(np.int64)
    t = events["t"]
    mono = events["moneyness"].astype(np.float64)
    forced = events["forced"]
    i_ex = events["i_ex"]  # exercise-day index in [0, N]

    vol_forced = int(q[forced].sum())
    vol_volunt = int(q[~forced].sum())
    vol_total = vol_forced + vol_volunt

    frac_forced = vol_forced / vol_total if vol_total > 0 else float("nan")
    frac_volunt = vol_volunt / vol_total if vol_total > 0 else float("nan")

    t_ex_days = np.arange(N) / N  # exercise day times [0, N-1/N]
    alpha_ex = alpha_func(t_ex_days)
    thresh_top = float(np.percentile(alpha_ex, 200.0 / 3.0))  # 66.67th pctile
    in_peak_days = alpha_ex >= thresh_top
    peak_t_lo = float(t_ex_days[in_peak_days].min())
    peak_t_hi = float(t_ex_days[in_peak_days].max())

    in_peak_ev = alpha_func(t) >= thresh_top
    vol_volunt_peak = int(q[~forced & in_peak_ev].sum())
    vol_volunt_shoulder = vol_volunt - vol_volunt_peak
    frac_volunt_in_peak = vol_volunt_peak / vol_volunt if vol_volunt > 0 else float("nan")
    frac_volunt_in_shoulder = vol_volunt_shoulder / vol_volunt if vol_volunt > 0 else float("nan")

    in_final10 = t > 0.9
    vol_forced_f10 = int(q[forced & in_final10].sum())
    frac_forced_f10 = vol_forced_f10 / vol_forced if vol_forced > 0 else float("nan")

    mean_mono_v = float(mono[~forced].mean()) if (~forced).any() else float("nan")
    mean_mono_f = float(mono[forced].mean()) if forced.any() else float("nan")

    t_v = t[~forced]
    t_f = t[forced]
    mean_t_volunt = float(np.average(t_v, weights=q[~forced])) if (~forced).any() else float("nan")
    median_t_volunt = float(np.median(t_v)) if (~forced).any() else float("nan")
    mean_t_forced = float(np.average(t_f, weights=q[forced])) if forced.any() else float("nan")
    median_t_forced = float(np.median(t_f)) if forced.any() else float("nan")

    vol_per_day_v = np.bincount(
        i_ex[~forced], weights=q[~forced].astype(float), minlength=N
    )[:N]
    alpha_per_day = alpha_func(np.arange(N) / N)
    corr_va = float(np.corrcoef(vol_per_day_v, alpha_per_day)[0, 1])

    return {
        "num_events": int(len(q)),
        "vol_total": vol_total,
        "vol_voluntary": vol_volunt,
        "vol_forced": vol_forced,
        "frac_voluntary": frac_volunt,
        "frac_forced": frac_forced,
        "alpha_top_tercile_threshold": thresh_top,
        "winter_peak_t_lo": peak_t_lo,
        "winter_peak_t_hi": peak_t_hi,
        "vol_voluntary_in_peak": vol_volunt_peak,
        "vol_voluntary_in_shoulder": vol_volunt_shoulder,
        "frac_voluntary_in_peak": frac_volunt_in_peak,
        "frac_voluntary_in_shoulder": frac_volunt_in_shoulder,
        "vol_forced_in_final10pct": vol_forced_f10,
        "frac_forced_in_final10pct": frac_forced_f10,
        "mean_moneyness_voluntary": mean_mono_v,
        "mean_moneyness_forced": mean_mono_f,
        "mean_t_voluntary": mean_t_volunt,
        "median_t_voluntary": median_t_volunt,
        "mean_t_forced": mean_t_forced,
        "median_t_forced": median_t_forced,
        "corr_voluntary_volume_alpha": corr_va,
    }

def plot_exercise_concentration(events, alpha_func=None, N=None, output_path="seasonal_exercise_concentration.png", n_bins=63):
    q = events["q"].astype(float)
    t = events["t"]
    forced = events["forced"]

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bar_w = (bin_edges[1] - bin_edges[0]) * 0.88

    idx_arr = np.searchsorted(bin_edges[1:-1], t, side="right")  # 0-indexed into n_bins

    total_vol = q.sum()
    vol_volunt = np.bincount(idx_arr[~forced], weights=q[~forced], minlength=n_bins)[:n_bins]
    vol_forced = np.bincount(idx_arr[forced], weights=q[forced], minlength=n_bins)[:n_bins]

    pct_volunt = vol_volunt / total_vol * 100.0
    pct_forced = vol_forced / total_vol * 100.0

    alpha_vals = alpha_func(bin_centers)

    fig = plt.figure(figsize=(10, 6))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.06)
    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1], sharex=ax_top)

    col_vol = "#2166ac"
    ax_top.bar(bin_centers, pct_volunt, width=bar_w, color=col_vol, alpha=0.85, label="Voluntary")
    ax_top.set_ylabel("Voluntary volume (% of total)")

    ax_a = ax_top.twinx()
    col_a = "#d95f02"
    ax_a.plot(bin_centers, alpha_vals, linestyle=":", color=col_a, linewidth=2.0, label="alpha(t)")
    ax_a.set_ylabel("alpha(t) (USD/MMBtu)")

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=col_vol, alpha=0.85),
        plt.Line2D([0], [0], linestyle=":", color=col_a, linewidth=2.0),
    ]
    ax_top.legend(handles, ["Voluntary", "alpha(t)"], loc="upper left", fontsize=9)
    plt.setp(ax_top.get_xticklabels(), visible=False)

    col_frc = "#b2182b"
    ax_bot.bar(bin_centers, pct_forced, width=bar_w, color=col_frc, alpha=0.85, label="Forced")
    ax_bot.set_xlabel("t (years)")
    ax_bot.set_ylabel("Forced volume (% of total)")
    ax_bot.legend(loc="upper left", fontsize=9)

    fig.savefig(output_path, dpi=200, bbox_inches="tight", format="png")
    plt.close(fig)

def analyse(data, output_dir, num_paths=250_000, seed=1337):
    K_sub = data.get("K_sub", 1)

    events = simulate_exercise_events(data, num_paths=num_paths, seed=seed)
    stats = compute_seasonal_stats(events, K_sub=K_sub)

    stats_path = os.path.join(output_dir, "seasonal_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Wrote to {stats_path}")

    plot_path = os.path.join(output_dir, "seasonal_exercise_concentration.png")
    plot_exercise_concentration(events, output_path=plot_path)
    print(f"Wrot e to {plot_path}")

    return stats
