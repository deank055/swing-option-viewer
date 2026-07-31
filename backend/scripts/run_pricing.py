# this file prices the option
# 1. prices the option and prints to terminal
# 2. note that your computer will sound like an airplane and explode if you set K_sub too high

from swing_option.core.lattice import build_price_grid
from swing_option.core.pricing import compute_admissible_actions, compute_option_value
from swing_option.config import T, N, sigma, r, kappa, alpha, S_0, K, C_max, C_min, q_max
from run_analysis import resolve_k_sub

def main():
    K_sub = resolve_k_sub()

    X_0 = S_0 - alpha(0)
    _, delta_t, delta_X, m, col_X0 = build_price_grid(
        T=T, sigma=sigma, X_0=X_0, N=N, kappa=kappa, K_sub=K_sub
    )

    admissible = compute_admissible_actions(T=T, N=N, C_min=C_min, C_max=C_max, q_max=q_max, K_sub=K_sub)

    _, _, option_price = compute_option_value(
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
        K_sub=K_sub
    )

    print(f"Swing option price = {option_price:.4f}")

if __name__ == "__main__":
    main()
