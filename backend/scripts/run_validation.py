# this file generates runs only the validation section of run_analysis

from swing_option import config
from swing_option.analysis import validation, policy_validation
from swing_option.core.memory import find_feasible_ksub, MEMORY_LIMIT_GB
from run_analysis import compute_data

OUTPUT_DIR = "data/output"

def _baseline_params():
    return dict(
        T=config.T, N=config.N, sigma=config.sigma, r=config.r,
        kappa=config.kappa, alpha=config.alpha, S_0=config.S_0, K=config.K,
        C_max=config.C_max, C_min=config.C_min, q_max=config.q_max,
    )

def _baseline_data():
    return {"params": _baseline_params()}

def _solve_policy_baseline(target_K_sub=100):
    K_sub = find_feasible_ksub(_baseline_params(), target_K_sub, memory_limit_gb=MEMORY_LIMIT_GB)
    return compute_data(K_sub=K_sub)

def main():
    validation.analyse(_baseline_data(), OUTPUT_DIR)

    policy_data = _solve_policy_baseline()
    policy_validation.analyse(policy_data, OUTPUT_DIR)


if __name__ == "__main__":
    main()
