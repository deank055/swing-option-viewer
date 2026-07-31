# this file contains the core lattice methods
# 1. builds the grid
# 2. computes paths on the grid

import numpy as np

def build_price_grid(T, sigma, X_0, N, kappa, n_std=4, K_sub=1):
    """
    Grid is centered at X=0, half-width covers standard deviations and initial deivation (X_0 = S_0 - alpha(0))
    Note that initial deviation is grid snapped (not exact) so will introduce discretisation error
    Unreachable nodes are np.nan

    K_sub sub-steps per contract day: N_fine = N*K_sub lattice steps, finer delta_t and delta_X.
    Grid is returned as 2d np.ndarray (N_fine+1, 2*m+1)
    """
    N_fine = N * K_sub
    delta_t = T / N_fine
    delta_X = sigma * np.sqrt(3 * delta_t)

    stationary_std = sigma / np.sqrt(2 * kappa)
    m_from_vol = int(np.ceil(n_std * stationary_std / delta_X))
    m_from_X0 = int(np.ceil(abs(X_0) / delta_X))
    m = max(m_from_vol, m_from_X0) + 2

    grid_width = 2 * m + 1
    grid = np.full((N_fine + 1, grid_width), np.nan)

    center = m

    j_X0 = int(round(X_0 / delta_X))
    col_X0 = center + j_X0

    for i in range(N_fine + 1):  # main loop, fill in reachable nodes
        left = max(col_X0 - i, 0)
        right = min(col_X0 + i, 2 * m)
        for col in range(left, right + 1):
            j = col - center
            grid[i, col] = j * delta_X

    return grid, delta_t, delta_X, m, col_X0

def compute_M_V(S, delta_t, kappa, Theta_t, sigma): # Computes conditional mean M and variance V of X_{t+delta_t} given X_t = S
    M = (Theta_t - S) * (1 - np.exp(-kappa * delta_t))
    V = (sigma**2 / (2 * kappa)) * (1 - np.exp(-2 * kappa * delta_t))
    return M, V

def transition_probabilities(S, delta_t, delta_S, kappa, Theta_t, sigma,is_top=False, is_bottom=False, clip=True):
    # clip=False returns the unclipped probabilities (so in the event we have negative probabilities, we leave it instead of renomralising)
    M, V = compute_M_V(S, delta_t, kappa, Theta_t, sigma)  # cond mean & var

    if is_top:
        pu = 1 + (V + M**2) / (2 * delta_S**2) + 3 * M / (2 * delta_S)
        pm = -(V + M**2) / (delta_S**2) - 2 * M / delta_S
        pd = (V + M**2) / (2 * delta_S**2) + M / (2 * delta_S)

    elif is_bottom:
        pu = (V + M**2) / (2 * delta_S**2) - M / (2 * delta_S)
        pm = -(V + M**2) / (delta_S**2) + 2 * M / delta_S
        pd = 1 + (V + M**2) / (2 * delta_S**2) - 3 * M / (2 * delta_S)

    else:
        pu = (V + M**2) / (2 * delta_S**2) + M / (2 * delta_S)
        pm = 1 - (V + M**2) / (delta_S**2)
        pd = (V + M**2) / (2 * delta_S**2) - M / (2 * delta_S)

    # moment matching
    probs = np.array([pu, pm, pd], dtype=float)

    if clip:  # renormalise and remove negative probabilities
        probs = np.maximum(probs, 0.0)
        probs /= probs.sum()

    return probs

def compute_process_paths_on_grid(num_paths, grid, N, m, col_X0, delta_t, delta_X, kappa, sigma): # Simulate X_t paths and construct S_t
    paths = np.full((num_paths, N + 1), np.nan)

    current_cols = np.full(num_paths, col_X0, dtype=int)
    paths[:, 0] = grid[0, col_X0]

    top_col = 2 * m
    bot_col = 0

    # iterate over time steps, then over num_paths
    for i in range(N):
        next_cols = np.empty(num_paths, dtype=int)

        for p in range(num_paths):
            col = current_cols[p]
            X = grid[i, col]

            is_top_ = (col == top_col)
            is_bottom_ = (col == bot_col)

            probs = transition_probabilities(
                S=X,
                delta_t=delta_t,
                delta_S=delta_X,
                kappa=kappa,
                Theta_t=0.0,  # X_t always reverts to 0
                sigma=sigma,
                is_top=is_top_,
                is_bottom=is_bottom_,
            )

            # handle boundary conditions
            if is_top_:
                candidates = [col, col - 1, col - 2]
            elif is_bottom_:
                candidates = [col + 2, col + 1, col]
            else:
                candidates = [col + 1, col, col - 1]

            next_col = np.random.choice(candidates, p=probs)
            next_cols[p] = next_col
            paths[p, i + 1] = grid[i + 1, next_col]

        current_cols = next_cols

    return paths