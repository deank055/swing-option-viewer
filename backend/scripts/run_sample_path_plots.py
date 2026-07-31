# this file generates sample path plots + boundary, used in results chapter
# 1. mostly copy pasted from optimal_strategy.py so a bit messy
# 2. ad hoc

import os

import numpy as np
from run_analysis import compute_data
from swing_option.analysis import optimal_strategy
from swing_option.config import K, kappa, sigma, alpha, r
    
OUTPUT_DIR = "data/output/sample_path_plots"

def main():
    data = compute_data(K_sub=100)
    print("Sanity check: Swing option price = {:.4f}".format(data["option_price"]))

    K_sub = data.get("K_sub", 1)

    boundary_full, forced_mask_full = optimal_strategy.extract_boundary(
        data["grid"], data["policy"], data["admissible"], alpha, data["delta_t"]
    )
    boundary_interp_full, boundary_diag = optimal_strategy.interpolate_boundary(
        data, boundary_full, kappa=kappa, sigma=sigma, K=K, r=r, alpha_func=alpha,
    )
    optimal_strategy.report_boundary_fallback(boundary_diag)

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
            "delta_t": data["delta_t"] * K_sub,   # one contract day
            "K_sub": 1,
            "N_fine": len(ex_idx) - 1,
        }
    else:
        boundary = boundary_full
        boundary_interp = boundary_interp_full
        forced_mask = forced_mask_full
        plot_data = data

    boundary_for_plots = boundary_interp

    # plot away for 1000 seeds
    for seed in range(1000):
        Cbar_single_full, col_history_full = optimal_strategy.simulate_capacity_path(
            data, kappa=kappa, sigma=sigma, num_paths=1, seed=seed, return_col_history=True,
        )
        Cbar_single_plot = Cbar_single_full[ex_idx] if K_sub > 1 else Cbar_single_full

        t_full = np.arange(data["grid"].shape[0]) * data["delta_t"]
        S_single_full = alpha(t_full) + (col_history_full[0] - data["m"]) * data["delta_X"]
        S_single_plot = S_single_full[ex_idx] if K_sub > 1 else S_single_full

        optimal_strategy.plot_boundary_average_path(
            plot_data, boundary_for_plots, os.path.join(OUTPUT_DIR, f"boundary_lead_sample_seed{seed}.png"),
            Cbar=Cbar_single_plot,
            S_path=S_single_plot,
            boundary_label=r"S*(t, C(t))",
            cbar_label=r"$C(t)$"
        )
    
if __name__ == "__main__":
    main()
