# this file simulates some asset paths
# 1. used it for my presentation, more ad hoc than anything else

import numpy as np
import matplotlib.pyplot as plt

from swing_option.core.lattice import build_price_grid, compute_process_paths_on_grid
from swing_option.config import alpha, S_0

T = 5
N = 252 * 5
KAPPA = 5
SIGMA = 5
K_SUB = 100
NUM_PATHS = 5
SEED = 1337

COLORS = ["#1b4f72", "#a93226", "#1e8449", "#b9770e", "#6c3483"]
OUTPUT_PATH = "data/output/asset_price_paths_illustration.png"

def simulate_price_paths(num_paths, K_sub, seed):
    X_0 = S_0 - alpha(0)
    grid, delta_t, delta_X, m, col_X0 = build_price_grid(
        T=T, sigma=SIGMA, X_0=X_0, N=N, kappa=KAPPA, K_sub=K_sub,
    )
    N_fine = N * K_sub

    np.random.seed(seed)
    X_paths = compute_process_paths_on_grid(
        num_paths=num_paths, grid=grid, N=N_fine, m=m, col_X0=col_X0,
        delta_t=delta_t, delta_X=delta_X, kappa=KAPPA, sigma=SIGMA,
    )

    t = np.linspace(0.0, T, N_fine + 1)
    alpha_curve = alpha(t)
    S_paths = alpha_curve[np.newaxis, :] + X_paths
    return t, S_paths, alpha_curve

def plot_price_paths(t, S_paths, alpha_curve, output_path):
    fig, ax = plt.subplots(figsize=(20, 5))

    for i in range(S_paths.shape[0]):
        ax.plot(t, S_paths[i], color=COLORS[i % len(COLORS)], linewidth=1)
    ax.plot(t, alpha_curve, color="0.4", linestyle="--", linewidth=1.0, label=r"$\alpha(t)$")

    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel(r"$S_t$", fontsize=12)
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(0, 6)    # mean is 3 so best to center around there
    ax.tick_params(labelsize=10)    
    ax.grid(True, linestyle=":", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", frameon=False, fontsize=11)

    fig.tight_layout()
    fig.savefig(output_path, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def main():
    t, S_paths, alpha_curve = simulate_price_paths(NUM_PATHS, K_SUB, SEED)
    plot_price_paths(t, S_paths, alpha_curve, OUTPUT_PATH)
    print(f"Written to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
