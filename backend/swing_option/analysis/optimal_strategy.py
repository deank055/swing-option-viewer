import json
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from swing_option.core.lattice import transition_probabilities
from swing_option.config import K, C_min, kappa, sigma, alpha, r

def simulate_capacity_path(data=None, kappa=None, sigma=None, num_paths=0, seed=1337, return_col_history=False):
    grid, policy = data["grid"], data["policy"]
    m, col_X0 = data["m"], data["col_X0"]
    delta_t, delta_X = data["delta_t"], data["delta_X"]
    N1 = grid.shape[0]
    N_steps = N1 - 1
    n_nodes = 2 * m + 1
    C_max = policy.shape[2] - 1

    idx = np.arange(n_nodes)
    is_top = idx == n_nodes - 1
    is_bot = idx == 0
    j_up = np.where(is_top, idx, np.where(is_bot, idx + 2, idx + 1))
    j_mid = np.where(is_top, idx - 1, np.where(is_bot, idx + 1, idx))
    j_dn = np.where(is_top, idx - 2, np.where(is_bot, idx, idx - 1))

    pu_arr = np.zeros(n_nodes)
    pm_arr = np.zeros(n_nodes)
    pd_arr = np.zeros(n_nodes)
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
    C_history = np.zeros((num_paths, N1), dtype=int)
    col_history = np.zeros((num_paths, N1), dtype=int)
    col_history[:, 0] = col_X0

    for i in range(N_steps):
        actions = policy[i, cols, C]
        C = np.minimum(C + actions, C_max)
        C_history[:, i + 1] = C

        u = rng.random(num_paths)
        pd_p, pm_p = pd_arr[cols], pm_arr[cols]
        cols = np.where(u < pd_p, j_dn[cols], np.where(u < pd_p + pm_p, j_mid[cols], j_up[cols]))
        col_history[:, i + 1] = cols

    if return_col_history:
        return C_history.mean(axis=0), col_history
    return C_history.mean(axis=0)

def extract_boundary(grid, policy, admissible, alpha_func, delta_t):
    N1, _, C1 = policy.shape
    boundary = np.full((N1, C1), np.nan)
    forced_mask = ~admissible[:, :, 0]

    for i in range(N1):
        reachable_cols = np.where(~np.isnan(grid[i]))[0]
        if reachable_cols.size == 0:
            continue
        alpha_i = alpha_func(i * delta_t)

        pol_block = policy[i, reachable_cols, :] > 0
        has_exercise = pol_block.any(axis=0)
        all_exercise = pol_block.all(axis=0)
        first_idx = pol_block.argmax(axis=0)
        j_cols = reachable_cols[first_idx]
        S_vals = alpha_i + grid[i, j_cols]

        valid = has_exercise & ~all_exercise
        boundary[i, valid] = S_vals[valid]

    return boundary, forced_mask

def interpolate_boundary(data, boundary_grid, kappa, sigma, K, r, alpha_func):
    grid, policy = data["grid"], data["policy"]
    V, m = data["V"], data["m"]
    delta_X, delta_t = data["delta_X"], data["delta_t"]

    N1, n_nodes = grid.shape[0], grid.shape[1]
    C_max = policy.shape[2] - 1
    discount = np.exp(-r * delta_t)

    idx = np.arange(n_nodes)
    is_top = idx == n_nodes - 1
    is_bot = idx == 0
    j_up = np.where(is_top, idx, np.where(is_bot, idx + 2, idx + 1))
    j_mid = np.where(is_top, idx - 1, np.where(is_bot, idx + 1, idx))
    j_dn = np.where(is_top, idx - 2, np.where(is_bot, idx, idx - 1))

    pu_arr = np.zeros(n_nodes)
    pm_arr = np.zeros(n_nodes)
    pd_arr = np.zeros(n_nodes)
    for j in range(n_nodes):
        X_j = (j - m) * delta_X
        pu_arr[j], pm_arr[j], pd_arr[j] = transition_probabilities(
            S=X_j, delta_t=delta_t, delta_S=delta_X,
            kappa=kappa, Theta_t=0.0, sigma=sigma,
            is_top=bool(is_top[j]), is_bottom=bool(is_bot[j]),
        )

    result = boundary_grid.copy()
    C_arr = np.arange(C_max)

    n_considered = 0
    n_fallback = 0

    for i in range(N1 - 1):  # terminal date has no V[i+1] so skip
        reachable_cols = np.where(~np.isnan(grid[i]))[0]
        if reachable_cols.size == 0:
            continue
        alpha_i = alpha_func(i * delta_t)

        pol_block = policy[i, reachable_cols, :] > 0
        has_exercise = pol_block.any(axis=0)
        all_exercise = pol_block.all(axis=0)
        valid = has_exercise & ~all_exercise

        first_idx = pol_block.argmax(axis=0)
        j_ex_cols = reachable_cols[first_idx]
        col_min = reachable_cols[0]

        considered_i = int(valid[:C_max].sum())
        n_considered += considered_i
        if considered_i == 0:
            continue

        can_interp = valid & (j_ex_cols > col_min) & (np.arange(C_max + 1) < C_max)
        if not can_interp.any():
            n_fallback += considered_i  # every considered cell here keeps the raw value
            continue

        E_V = (pu_arr[:, None] * V[i + 1, j_up, :]
             + pm_arr[:, None] * V[i + 1, j_mid, :]
             + pd_arr[:, None] * V[i + 1, j_dn, :])

        S_nodes = alpha_i + (idx - m) * delta_X
        g = (S_nodes[:, None] - K) + discount * (E_V[:, 1:] - E_V[:, :-1])

        j_ex_C = j_ex_cols[:C_max]
        j_hold_C = np.maximum(j_ex_C - 1, 0)

        g_ex = g[j_ex_C, C_arr]
        g_hold = g[j_hold_C, C_arr]

        sign_ok = (can_interp[:C_max]
                   & (g_hold <= 0)
                   & (g_ex > 0)
                   & ((g_ex - g_hold) > 1e-12))

        X_hold = (j_hold_C - m) * delta_X
        denom = np.where(sign_ok, g_ex - g_hold, 1.0)
        frac = np.where(sign_ok, -g_hold / denom, 0.0)
        X_star = X_hold + delta_X * frac

        result[i, :C_max] = np.where(sign_ok, alpha_i + X_star, result[i, :C_max])

        n_fallback += considered_i - int(sign_ok.sum())

    fallback_fraction = (n_fallback / n_considered) if n_considered > 0 else float("nan")
    diagnostics = dict(
        n_cells_considered=n_considered,
        n_fallback=n_fallback,
        fallback_fraction=fallback_fraction,
    )
    return result, diagnostics

def report_boundary_fallback(diagnostics):
    n_considered = diagnostics["n_cells_considered"]
    frac = diagnostics["fallback_fraction"]

def _cell_edges(centers):
    edges = np.empty(len(centers) + 1)
    edges[1:-1] = (centers[:-1] + centers[1:]) / 2
    edges[0] = centers[0] - (centers[1] - centers[0]) / 2
    edges[-1] = centers[-1] + (centers[-1] - centers[-2]) / 2
    return edges

def build_spot_mesh(N, m, delta_X, alpha_func, delta_t):
    n_nodes = 2 * m + 1
    t = np.arange(N + 1) * delta_t
    X_full = (np.arange(n_nodes) - m) * delta_X

    t_edges = _cell_edges(t)
    X_edges = _cell_edges(X_full)
    T_edges = np.tile(t_edges.reshape(-1, 1), (1, n_nodes + 1))
    S_edges = alpha_func(t_edges).reshape(-1, 1) + X_edges.reshape(1, -1)

    return t, T_edges, S_edges

def boundary_y_limits(boundary, alpha_t, K, C_values=None, margin_frac=0.15):
    cols = boundary if C_values is None else boundary[:, C_values]
    finite = cols[np.isfinite(cols)]
    vals = np.concatenate([finite, alpha_t, [K]])
    lo, hi = vals.min(), vals.max()
    pad = (hi - lo) * margin_frac
    return lo - pad, hi + pad

def plot_policy_heatmaps(data, boundary, output_path):
    grid, policy, delta_t = data["grid"], data["policy"], data["delta_t"]
    N = grid.shape[0] - 1
    q_max = data["admissible"].shape[2] - 1
    admissible = data["admissible"]
    forced_mask = ~admissible[:, :, 0]

    t, T_edges, S_edges = build_spot_mesh(N, data["m"], data["delta_X"], alpha, delta_t)
    alpha_t = alpha(t)
    unreachable = np.isnan(grid)
    C_values = [0, 525, 1050, 1120, 1190, 1250]

    # lowest reachable node per row, for the "no genuine threshold" check below.
    lowest_reachable_col = np.full(N + 1, -1, dtype=int)
    for i in range(N + 1):
        rc = np.where(~unreachable[i])[0]
        if rc.size:
            lowest_reachable_col[i] = rc[0]
    has_floor = lowest_reachable_col >= 0

    cmap = plt.get_cmap("viridis", q_max + 1).copy()
    cmap.set_bad(color="lightgrey")
    norm = mcolors.BoundaryNorm(np.arange(-0.5, q_max + 1.5, 1), cmap.N)

    ylim = boundary_y_limits(boundary, alpha_t, K, C_values)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=False)
    mesh = None
    for ax, C in zip(axes.ravel(), C_values):
        min_i = -(-C // q_max)  # ceil(C / q_max): earliest step C is reachable
        t_lo = min_i * delta_t

        if C >= C_min:
            i_hi = N
        else:
            voluntary_rows = np.where(~forced_mask[:, C])[0]
            i_hi = int(voluntary_rows[-1]) if voluntary_rows.size else min_i
            i_hi = min(i_hi, N)
        t_hi = i_hi * delta_t

        pol_C = np.where(unreachable, np.nan, policy[:, :, C].astype(float))
        pol_C[forced_mask[:, C], :] = np.nan
        pol_C[:min_i, :] = np.nan
        if i_hi < N:
            pol_C[i_hi + 1:, :] = np.nan

        mesh = ax.pcolormesh(
            T_edges, S_edges, np.ma.masked_invalid(pol_C),
            cmap=cmap, norm=norm, shading="flat",
        )

        floor_exercises = np.zeros(N + 1, dtype=bool)
        floor_exercises[has_floor] = policy[np.arange(N + 1)[has_floor], lowest_reachable_col[has_floor], C] > 0
        no_threshold = forced_mask[:, C] | floor_exercises

        in_window = (t >= t_lo) & (t <= t_hi)
        boundary_C = np.where(no_threshold | ~in_window, np.nan, boundary[:, C])
        alpha_clipped = np.where(in_window, alpha_t, np.nan)

        ax.plot(t, boundary_C, color="red", linewidth=1.5, label="S*(t, C)")
        ax.axhline(K, color="white", linestyle="--", linewidth=1, label="K")
        ax.plot(t, alpha_clipped, color="black", linestyle=":", linewidth=1, label=r"$\alpha(t)$")

        ax.set_xlabel("t (years)")
        ax.set_ylabel("Spot price S")
        ax.set_xlim(t_lo, t_hi)
        ax.set_ylim(ylim)

    axes.ravel()[0].legend(loc="upper left", fontsize=8)
    fig.colorbar(mesh, ax=axes.ravel().tolist(), ticks=np.arange(0, q_max + 1, 2), label="optimal volume q*")

    fig.savefig(output_path, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)

def plot_boundary_average_path(data, boundary, output_path, num_paths=1, capacity_seed=1337, Cbar=None, cbar_label=None, boundary_label=None, S_path=None):
    delta_t = data["delta_t"]
    N1 = boundary.shape[0]
    C_max = boundary.shape[1] - 1
    t = np.arange(N1) * delta_t
    alpha_t = alpha(t)

    if Cbar is None:
        Cbar = simulate_capacity_path(data, kappa=kappa, sigma=sigma, num_paths=num_paths, seed=capacity_seed)
    Cbar_idx = np.clip(np.round(Cbar).astype(int), 0, C_max)
    S_star_path = boundary[np.arange(N1), Cbar_idx]

    forced_mask_full = ~data["admissible"][:, :, 0]
    forced_path = forced_mask_full[np.arange(N1), Cbar_idx]

    if boundary_label is None:
        boundary_label = r"S*(t, $\bar{C}(t)$)"
    if cbar_label is None:
        cbar_label = r"$\bar{C}(t)$"

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(t, S_star_path, color="red", linewidth=2, label=boundary_label)
    ax.plot(t, alpha_t, color="black", linestyle=":", linewidth=1.5, label=r"$\alpha(t)$")
    if S_path is not None:
        ax.plot(t, S_path, color="tab:blue", linewidth=1, alpha=0.7, label=r"realised $S_t$")
    ax.axhline(K, color="gray", linestyle="--", linewidth=1, label=f"K = {K}")

    forced_rows = np.where(forced_path)[0]
    if forced_rows.size:
        ax.axvspan(t[forced_rows[0]], t[-1], color="red", alpha=0.12, label="forced exercise")

    ax.set_xlabel("t (years)")
    ax.set_ylabel("Spot price S")
    # ax.set_ylim(boundary_y_limits(S_star_path[np.isfinite(S_star_path)], alpha_t, K))
    ax.set_ylim(0, 8)

    ax2 = ax.twinx()
    ax2.plot(t, Cbar, color="tab:green", linewidth=1.5, label=cbar_label)
    ax2.axhline(C_min, color="tab:green", linestyle="--", linewidth=1, label="C_min")
    ax2.set_ylabel("cumulative volume C")
    ax2.set_ylim(0, C_max)

    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, loc="upper left", fontsize=8)

    fig.savefig(output_path, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)

def plot_boundary_surface_3d(data, boundary, output_path):
    delta_t = data["delta_t"]
    q_max = data["admissible"].shape[2] - 1
    forced_mask = ~data["admissible"][:, :, 0]
    N1, C1 = boundary.shape
    t = np.arange(N1) * delta_t
    C_arr = np.arange(C1)

    Z = boundary.copy()
    for C in range(C1):
        min_i = -(-C // q_max)
        Z[:min_i, C] = np.nan
    Z[forced_mask] = np.nan

    TT, CC = np.meshgrid(t, C_arr, indexing="ij")

    finite = Z[np.isfinite(Z)]
    z_min, z_max = float(finite.min()), float(finite.max())

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(TT, CC, Z, cmap="viridis", antialiased=False)

    ax.set_zlim(z_min, z_max)
    fig.colorbar(surf, ax=ax, label="S*", shrink=0.5, pad=0.1)

    ax.set_xlabel("t (years)")
    ax.set_ylabel("C")
    ax.set_zlabel("S*")
    ax.view_init(elev=25, azim=-45)

    fig.tight_layout()
    fig.savefig(output_path, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)

def _to_py(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [_to_py(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {k: _to_py(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_py(x) for x in obj]
    return obj

def print_rule_anchors(boundary, forced_mask, data, boundary_interp=None, Cbar=None):
    delta_t = data["delta_t"]
    N1 = boundary.shape[0]
    N_day = N1 - 1
    t = np.arange(N1) * delta_t

    C_max = data["policy"].shape[2] - 1
    q_max = data["admissible"].shape[2] - 1

    report = {}

    premium = boundary[:, 0] - alpha(t)
    delta_X = data["delta_X"]
    print(f"delta_X = {delta_X:.4f} USD/MMBtu")
    print(f"Grid-snapped mean boundary S*(t,0): {np.nanmean(boundary[:, 0]):.4f} USD/MMBtu")
    print(f"Grid-snapped mean premium S*(t,0) - alpha(t): {np.nanmean(premium):.4f}  "
          f"({np.nanmean(premium)/delta_X:.2f} grid steps)")
    if boundary_interp is not None:
        print(f"Interpolated mean boundary S*(t,0): {np.nanmean(boundary_interp[:, 0]):.4f} USD/MMBtu")
        premium_i = boundary_interp[:, 0] - alpha(t)
        print(f"Interpolated  mean premium S*(t,0) - alpha(t): {np.nanmean(premium_i):.4f}  "
              f"({np.nanmean(premium_i)/delta_X:.2f} grid steps)")

        if Cbar is not None:
            Cbar_idx = np.clip(np.round(Cbar).astype(int), 0, C_max)
            S_path_interp = boundary_interp[np.arange(N1), Cbar_idx]
            S_path_raw = boundary[np.arange(N1), Cbar_idx]
            voluntary = ~forced_mask[np.arange(N1), Cbar_idx]

            valid_interp = np.isfinite(S_path_interp) & voluntary
            n_valid = int(valid_interp.sum())
            mean_path_interp = float(np.mean(S_path_interp[valid_interp])) if n_valid else None
            t_lo = float(t[valid_interp].min()) if n_valid else None
            t_hi = float(t[valid_interp].max()) if n_valid else None

            valid_raw = np.isfinite(S_path_raw) & voluntary
            mean_path_raw = float(np.mean(S_path_raw[valid_raw])) if valid_raw.any() else None

            premium_path = (mean_path_interp - float(K)) if mean_path_interp is not None else None
            
            near_K = np.isfinite(S_path_interp) & (np.abs(S_path_interp - float(K)) < 1e-6)
            lock_i = int(np.argmax(near_K)) if near_K.any() else None
            stays_at_K = None

            remaining_capacity_needed = (N_day - np.arange(N1)) * q_max
            slack = C_max - np.asarray(Cbar)
            decouples = remaining_capacity_needed < slack
            first_decouple_i = int(np.argmax(decouples)) if decouples.any() else None

            report["mean_path_boundary"] = {
                "mean_interpolated": mean_path_interp,
                "mean_grid_snapped": mean_path_raw,
                "mean_premium_over_K": premium_path,
                "n_finite_voluntary_dates": n_valid,
                "t_range": [t_lo, t_hi] if n_valid else None,
                "lock_onto_K_i": lock_i,
                "lock_onto_K_t": float(t[lock_i]) if lock_i is not None else None,
                "stays_at_K_thereafter": stays_at_K,
                "decouple_condition_first_i": first_decouple_i,
                "decouple_condition_first_t": float(t[first_decouple_i]) if first_decouple_i is not None else None,
                "lock_vs_decouple_diff_steps": (
                    (lock_i - first_decouple_i) if (lock_i is not None and first_decouple_i is not None) else None
                ),
            }

    forced_rows = np.where(forced_mask.any(axis=1))[0]
    if forced_rows.size:
        i_first = forced_rows[0]
        c_forced = np.where(forced_mask[i_first])[0]
        print(
            f"Forced exercise first appears at t={i_first * delta_t:.3f} (i={i_first}), "
            f"C in [{c_forced.min()}, {c_forced.max()}]"
        )
    else:
        print("No forced exercise states found in baseline.")

    return report

def analyse(data, output_dir):
    K_sub = data.get("K_sub", 1)

    boundary_full, forced_mask_full = extract_boundary(data["grid"], data["policy"], data["admissible"], alpha, data["delta_t"])
    boundary_interp_full, boundary_diag = interpolate_boundary(data, boundary_full, kappa=kappa, sigma=sigma, K=K, r=r, alpha_func=alpha)
    report_boundary_fallback(boundary_diag)

    boundary_full[-1, :] = np.nan
    boundary_interp_full[-1, :] = np.nan

    if K_sub > 1:
        ex_idx = np.arange(0, data["grid"].shape[0], K_sub)
        boundary = boundary_full[ex_idx]
        boundary_interp = boundary_interp_full[ex_idx]
        forced_mask = forced_mask_full[ex_idx]
        plot_data = {
            **data,
            "grid": data["grid"][ex_idx],
            "policy": data["policy"][ex_idx],
            "admissible": data["admissible"][ex_idx],
            "delta_t": data["delta_t"] * K_sub,
            "K_sub": 1,
            "N_fine": len(ex_idx) - 1,
        }
    else:
        boundary = boundary_full
        boundary_interp = boundary_interp_full
        forced_mask = forced_mask_full
        plot_data = data

    boundary_for_plots = boundary_interp

    Cbar_full = simulate_capacity_path(data, kappa=kappa, sigma=sigma, num_paths=250_000, seed=1337)
    Cbar_plot = Cbar_full[ex_idx] if K_sub > 1 else Cbar_full

    Cbar_single_full, col_history_full = simulate_capacity_path(data, kappa=kappa, sigma=sigma, num_paths=1, seed=1337, return_col_history=True)
    Cbar_single_plot = Cbar_single_full[ex_idx] if K_sub > 1 else Cbar_single_full

    t_full = np.arange(data["grid"].shape[0]) * data["delta_t"]
    S_single_full = alpha(t_full) + (col_history_full[0] - data["m"]) * data["delta_X"]
    S_single_plot = S_single_full[ex_idx] if K_sub > 1 else S_single_full

    plot_policy_heatmaps(
        plot_data,
        boundary_for_plots,
        os.path.join(output_dir, "policy_heatmaps.png")
    )
    plot_boundary_average_path(
        plot_data,
        boundary_for_plots,
        os.path.join(output_dir, "boundary_lead.png"),
        Cbar=Cbar_plot
    )
    plot_boundary_average_path(
        plot_data, boundary_for_plots, 
        os.path.join(output_dir, "boundary_lead_sample.png"),
        Cbar=Cbar_single_plot,
        S_path=S_single_plot,
        boundary_label=r"S*(t, C(t))",
        cbar_label=r"$C(t)$"
    )
    plot_boundary_surface_3d(
        plot_data,
        boundary_for_plots,
        os.path.join(output_dir, "whole_surface.png")
    )

    report = print_rule_anchors(
        boundary,
        forced_mask,
        plot_data,
        boundary_interp,
        Cbar=Cbar_plot
    )

    json_path = os.path.join(output_dir, "optimal_strategy_report.json")
    with open(json_path, "w") as f:
        json.dump(_to_py(report), f, indent=2)
    print(f"\nJSON written: {json_path}")
