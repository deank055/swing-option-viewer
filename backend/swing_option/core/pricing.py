# this file contains the core pricing program
# 1. computes option value
# 2. computes admissible action

import time as _time
import numpy as np

from swing_option.core.lattice import transition_probabilities

def compute_admissible_actions(T, N, C_min, C_max, q_max, K_sub=1):
    N_fine = N * K_sub
    admissible = np.zeros((N_fine + 1, C_max + 1, q_max + 1), dtype=bool)

    for i in range(N_fine + 1): # on subsetps only q=0 is allowed
        if i % K_sub != 0:
            admissible[i, :, 0] = True

    # vectorise so it runs faster
    C_arr = np.arange(C_max + 1)[:, None]  # (C_max+1, 1)
    q_arr = np.arange(q_max + 1)[None, :]  # (1, q_max+1)
    Cq = C_arr + q_arr  # (C_max+1, q_max+1)
    upper_bound_holds = Cq <= C_max

    # main loop
    for i in range(0, N_fine + 1, K_sub):
        remaining_days = (N_fine - i) // K_sub
        lower_bound_holds = Cq + remaining_days * q_max >= C_min
        admissible[i] = upper_bound_holds & lower_bound_holds

    return admissible

def compute_option_value(delta_t, delta_X, m, col_X0, N, admissible, K, r, kappa, alpha_func, sigma, C_max, q_max, K_sub=1, clip=True): # vectorised over (j, C), clip=False disables the negativity guard on transition probabilities
    t0 = _time.time()

    N_fine = admissible.shape[0] - 1
    n_nodes = 2 * m + 1

    V = np.zeros((N_fine + 1, n_nodes, C_max + 1), dtype=np.float32) # float32 is accurate enough to capture value
    policy = np.zeros((N_fine + 1, n_nodes, C_max + 1), dtype=np.int8) # float8 is also accurate (we never exceed 20)
    discount = np.exp(-r * delta_t)

    # precompute transition probabilities for each node (doesnt care about time in X, only in alpha)
    idx = np.arange(n_nodes)
    pu_arr = np.zeros(n_nodes)
    pm_arr = np.zeros(n_nodes)
    pd_arr = np.zeros(n_nodes)
    for j in range(n_nodes):
        X_j = (j - m) * delta_X
        probs = transition_probabilities(
            S=X_j, 
            delta_t=delta_t, 
            delta_S=delta_X,
            kappa=kappa, 
            Theta_t=0.0, 
            sigma=sigma,
            is_top=(j == n_nodes - 1), 
            is_bottom=(j == 0),
            clip=clip,
        )
        pu_arr[j], pm_arr[j], pd_arr[j] = probs

    # precompute neighbour indices for all nodes
    is_top = idx == n_nodes - 1
    is_bot = idx == 0
    j_up = np.where(is_top, idx, np.where(is_bot, idx + 2, idx + 1))
    j_mid = np.where(is_top, idx - 1, np.where(is_bot, idx + 1, idx))
    j_dn = np.where(is_top, idx - 2, np.where(is_bot, idx, idx - 1))

    C_indices = np.arange(C_max + 1)

    alpha_N = alpha_func(N_fine * delta_t)
    S_N = alpha_N + (idx - m) * delta_X
    raw_unit = S_N - K  # signed payoff
    payoff_pu = raw_unit[:, None]

    best = np.full((n_nodes, C_max + 1), -np.inf)
    best_q = np.zeros((n_nodes, C_max + 1), dtype=int)
    for q in range(q_max + 1):
        adm_q = admissible[N_fine, :, q]
        if not adm_q.any():
            continue
        candidate = np.where(adm_q[None, :], payoff_pu * q, -np.inf)
        update = candidate > best
        best = np.where(update, candidate, best)
        best_q = np.where(update, q, best_q)

    V[N_fine] = np.where(best > -np.inf, best, 0.0)
    policy[N_fine] = best_q

    for i in range(N_fine - 1, -1, -1): # backward induction routine
        E_cont = discount * (
            pu_arr[:, None] * V[i + 1, j_up, :]
            + pm_arr[:, None] * V[i + 1, j_mid, :]
            + pd_arr[:, None] * V[i + 1, j_dn, :]
        )

        if i % K_sub != 0:  # substepping logic
            V[i] = E_cont
            continue

        alpha_t = alpha_func(i * delta_t)
        S_nodes = alpha_t + (idx - m) * delta_X
        raw_unit = S_nodes - K
        payoff_pu = raw_unit[:, None]

        best = np.full((n_nodes, C_max + 1), -np.inf)
        best_q = np.zeros((n_nodes, C_max + 1), dtype=int)
        for q in range(q_max + 1):
            adm_q = admissible[i, :, q]
            if not adm_q.any():
                continue
            C_next = np.minimum(C_indices + q, C_max)
            total = np.where(adm_q[None, :], payoff_pu * q + E_cont[:, C_next], -np.inf)
            update = total > best
            best = np.where(update, total, best)
            best_q = np.where(update, q, best_q)

        V[i] = np.where(best > -np.inf, best, 0.0)
        policy[i] = best_q

    option_price = V[0, col_X0, 0]
    elapsed = _time.time() - t0
    print(f"K_sub={K_sub}  N_fine={N_fine}  delta_X={delta_X:.4f}  clip={clip}  price={option_price:.4f}  runtime={elapsed:.1f}s")
    return V, policy, option_price
