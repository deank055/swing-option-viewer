# this file runs all the analyses
# 1. generates all plots for chapter 5
# 2. generates all numbers and figure for chapter 6

import gc

from swing_option.analysis import convergence, optimal_strategy, policy_analysis, sensitivity, structural_properties, data_exploration, seasonal_behaviour, validation, policy_validation
from swing_option.core.lattice import build_price_grid
from swing_option.core.pricing import compute_admissible_actions, compute_option_value
from swing_option.core.memory import find_feasible_ksub
from swing_option.config import T, N, sigma, r, kappa, alpha, S_0, K, C_max, C_min, q_max

TARGET_K_SUB = 100   # actual K_sub may downscale

def resolve_k_sub(target_K_sub=TARGET_K_SUB):   # find highest feasible k_sub
    params = {
        "T": T, "N": N, "sigma": sigma, "kappa": 1.5, "S_0": S_0, "C_max": C_max,
    }
    print(f"Finding highest feasible K_sub (target={target_K_sub}).")
    resolved = find_feasible_ksub(params, target_K_sub)
    if resolved < target_K_sub:
        print(f"    target={target_K_sub} resolved to {resolved}")
    return resolved

def _add_parameters(data):
    data["params"] = dict(
        T=T,
        N=N,
        sigma=sigma,
        r=r,
        kappa=kappa,
        alpha=alpha,
        S_0=S_0,
        K=K,
        C_max=C_max,
        C_min=C_min,
        q_max=q_max,
    )
    return data

def compute_data(K_sub=1):
    X_0 = S_0 - alpha(0)
    grid, delta_t, delta_X, m, col_X0 = build_price_grid(
        T=T, sigma=sigma, X_0=X_0, N=N, kappa=kappa, K_sub=K_sub
    )

    admissible = compute_admissible_actions(T=T, N=N, C_min=C_min, C_max=C_max, q_max=q_max, K_sub=K_sub)

    V, policy, option_price = compute_option_value(
        delta_t=delta_t,
        delta_X=delta_X,
        m=m,
        col_X0=col_X0,
        N=N,
        admissible=admissible,
        K=K,
        r=r,
        kappa=kappa,
        alpha_func=alpha,
        sigma=sigma,
        C_max=C_max,
        q_max=q_max,
        K_sub=K_sub,
    )

    data = {
        "grid": grid,
        "delta_t": delta_t,
        "delta_X": delta_X,
        "m": m,
        "col_X0": col_X0,
        "admissible": admissible,
        "V": V,
        "policy": policy,
        "option_price": option_price,
        "K_sub": K_sub,
        "N_fine": N * K_sub,
    }
    data = _add_parameters(data)

    return data

def analyse_data(data, output_dir):
    # chapter 5 (calibration chapter)
    # analyse/plot underlying data
    data_exploration.analyse(data, output_dir)

    # chapter 6 (results chapter)
    # Optimal strategy analysis (6.1)
    optimal_strategy.analyse(data, output_dir)

    # Structural properties (6.2)
    structural_properties.analyse(data, output_dir)

    del data["V"]   # free V so we can allocate memory for the parameter sweeps
    gc.collect()

    # Sensitivity analysis (6.3)
    sensitivity.analyse(data, output_dir)

    # Seasonal analysis (6.4)
    seasonal_behaviour.analyse(data, output_dir)

    # Naive-policy comparison (thesis 6.5)
    policy_analysis.analyse(data, output_dir)

    # Convergence ladder (6.6)
    convergence.analyse(data, output_dir)

    # Value-function validation (6.6.1)
    validation.analyse(data, output_dir)

    # Optimal-policy validation (6.6.2)
    # recompute because we need the value function again, not the most efficient but oh well "if it works don't touch it"
    policy_validation.analyse(compute_data(K_sub=data["K_sub"]), output_dir)

def main():
    OUTPUT_DIR = "data/output"

    # resolve K_sub
    K_sub = resolve_k_sub()

    # compute all data
    data = compute_data(K_sub=K_sub)

    print("Swing option price = {:.4f}".format(data["option_price"]))
    analyse_data(data, OUTPUT_DIR)

if __name__ == "__main__":
    main()
