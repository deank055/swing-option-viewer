# this file contains memory probing uses to prevent OOMs or killed processes
# 1. keeps every K_sub identical for the thesis experiments, especially important for the parameter sweeps (low kappa greatly increases size of state space)
# 2. value function is the biggest allocation (policy is smaller, float32 vs int8))
# 3. probe builds cheap price grid to get n_nodes first

from swing_option import config
from swing_option.core.lattice import build_price_grid

MEMORY_LIMIT_GB = 64
K_SUB_CANDIDATES = [100, 64, 32, 16, 8, 4, 2, 1] # i ensured K_sub = 100 for my actual results

def v_memory_gb(params, K_sub): # estimate size of value function V
    C1 = params["C_max"] + 1
    grid_probe, _, _, m_probe, _ = build_price_grid(
        T=params["T"], 
        sigma=params["sigma"], 
        X_0=params["S_0"] - config.alpha(0), 
        N=params["N"],
        kappa=params["kappa"], 
        K_sub=K_sub,
    )
    n_nodes = 2 * m_probe + 1
    N_fine = params["N"] * K_sub
    del grid_probe # remove so we dont fill up memory anyways
    return (N_fine + 1) * n_nodes * C1 * 4 / 1e9  # V is float32 so 4 bytes per element

def find_feasible_ksub(params, target_K_sub, memory_limit_gb=MEMORY_LIMIT_GB): # find largest feasible K_sub
    candidates = [target_K_sub] + [k for k in K_SUB_CANDIDATES if k < target_K_sub]
    for k in candidates:
        mem = v_memory_gb(params, k)
        # print(f"Approximate memory for K_sub={k}: V≈{mem:.1f} GB")
        if mem <= memory_limit_gb:
            return k
    raise MemoryError(
        f"No K_sub <= {target_K_sub} fits within {memory_limit_gb} GB. Run this on a computer, not on a smart fridge."
    )
