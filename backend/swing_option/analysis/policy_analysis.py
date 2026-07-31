# this file generates everything requierd for section 5 of results chapter (naive policies)
import gc
import json
import os

import numpy as np

from swing_option.core.pricing import compute_admissible_actions

REQUIRED_PARAMS = ("T", "N", "sigma", "r", "kappa", "alpha", "S_0", "K", "C_max", "C_min", "q_max")

def _to_py(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, float):
        return None if (obj != obj) else obj
    if isinstance(obj, np.ndarray):
        return [_to_py(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {k: _to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_py(x) for x in obj]
    return obj

def simulate_paths(data, num_paths, seed):
    params = data["params"]
    kappa, sigma, alpha_func = params["kappa"], params["sigma"], params["alpha"]
    N, T, S_0 = params["N"], params["T"], params["S_0"]

    dt = T / N
    decay = np.exp(-kappa * dt)
    std = sigma * np.sqrt((1 - np.exp(-2 * kappa * dt)) / (2 * kappa))

    rng = np.random.default_rng(seed)
    X_0 = S_0 - alpha_func(0.0)

    X = np.empty((num_paths, N + 1))
    X[:, 0] = X_0
    for i in range(N):
        X[:, i + 1] = decay * X[:, i] + std * rng.standard_normal(num_paths)

    t = np.arange(N + 1) * dt
    return X + alpha_func(t)[None, :]

def clamp_admissible(q, i, C, data):
    admissible_ex = data["admissible_ex"]
    q_max_v = admissible_ex.shape[2] - 1

    q_req = np.clip(np.round(np.asarray(q)).astype(int), 0, q_max_v)

    adm_at_C = admissible_ex[i, C, :]
    q_min_forced = adm_at_C.argmax(axis=1)
    q_max_allowed = q_max_v - adm_at_C[:, ::-1].argmax(axis=1)

    q_clamped = np.clip(q_req, q_min_forced, q_max_allowed)
    was_raised = q_clamped > q_req
    return q_clamped, was_raised

def evaluate_policy(paths, policy_fn, data):
    params = data["params"]
    K, r, N, T = params["K"], params["r"], params["N"], params["T"]
    dt = T / N
    disc = np.exp(-r * dt * np.arange(N + 1))

    num_paths = paths.shape[0]
    C = np.zeros(num_paths, dtype=int)
    value = np.zeros(num_paths, dtype=float)
    total_q = np.zeros(num_paths, dtype=float)
    raised_q = np.zeros(num_paths, dtype=float)

    for i in range(N + 1):
        q_req = policy_fn(i, paths[:, i], C)
        q, raised = clamp_admissible(q_req, i, C, data)
        value += disc[i] * q * (paths[:, i] - K)
        raised_q += np.where(raised, q, 0)
        total_q += q
        C += q

    denom = total_q.sum()
    forced_volume_frac = float(raised_q.sum() / denom) if denom > 0 else 0.0
    return value, forced_volume_frac

def make_dp_policy(data):
    K_sub = data["K_sub"]
    policy_ex = data["policy"][::K_sub]
    m = data["m"]
    delta_X = data["delta_X"]
    alpha_func = data["params"]["alpha"]
    dt = data["params"]["T"] / data["params"]["N"]
    n_nodes = policy_ex.shape[1]

    snap_stats = {"sum_abs_err": 0.0, "n": 0}

    def policy_fn(i, S_vec, C_vec):
        alpha_i = alpha_func(i * dt)
        X_vec = S_vec - alpha_i
        j = np.clip(np.round(X_vec / delta_X).astype(int) + m, 0, n_nodes - 1)
        S_node = alpha_i + (j - m) * delta_X
        snap_stats["sum_abs_err"] += float(np.sum(np.abs(S_vec - S_node)))
        snap_stats["n"] += S_vec.size
        return policy_ex[i, j, C_vec].astype(float)

    policy_fn.snap_stats = snap_stats
    return policy_fn

def make_pro_rata(data):
    params = data["params"]
    rate = params["C_max"] / params["N"]

    def policy_fn(i, S_vec, C_vec):
        return np.full_like(S_vec, rate, dtype=float)

    return policy_fn

def make_deterministic_seasonal(data):
    params = data["params"]
    alpha_func, N, T = params["alpha"], params["N"], params["T"]
    q_max, C_max = params["q_max"], params["C_max"]
    n_top = C_max // q_max

    dt = T / N
    t_ex = np.arange(N + 1) * dt
    alpha_vals = alpha_func(t_ex)
    top_days = frozenset(np.argsort(alpha_vals)[-n_top:].tolist())

    def policy_fn(i, S_vec, C_vec):
        q = q_max if i in top_days else 0
        return np.full_like(S_vec, q, dtype=float)

    return policy_fn

def make_fixed_threshold(data, S_bar):
    q_max = data["params"]["q_max"]

    def policy_fn(i, S_vec, C_vec):
        return np.where(S_vec >= S_bar, q_max, 0).astype(float)

    return policy_fn

def fit_best_constant(data, seed_opt, num_paths_opt):
    K = data["params"]["K"]
    opt_paths = simulate_paths(data, num_paths=num_paths_opt, seed=seed_opt)

    coarse_grid = np.round(np.arange(K, K + 2.0 + 1e-9, 0.05), 2)
    coarse_means = np.array([
        evaluate_policy(opt_paths, make_fixed_threshold(data, S_bar), data)[0].mean()
        for S_bar in coarse_grid
    ])
    i_best = int(np.argmax(coarse_means))
    coarse_best = float(coarse_grid[i_best])

    fine_lo = max(K, coarse_best - 0.05)
    fine_hi = coarse_best + 0.05
    fine_grid = np.round(np.arange(fine_lo, fine_hi + 1e-9, 0.01), 2)
    fine_means = np.array([
        evaluate_policy(opt_paths, make_fixed_threshold(data, S_bar), data)[0].mean()
        for S_bar in fine_grid
    ])
    j_best = int(np.argmax(fine_means))
    S_bar_opt = float(fine_grid[j_best])
    in_sample_optimum = float(fine_means[j_best])

    grid = dict(
        coarse=list(zip(coarse_grid.tolist(), coarse_means.tolist())),
        fine=list(zip(fine_grid.tolist(), fine_means.tolist())),
    )

    del opt_paths
    gc.collect()
    return S_bar_opt, grid, in_sample_optimum

def analyse(data, output_dir, num_paths=250_000, seed=1337, seed_opt=420, num_paths_opt=250_000):
    params = data.get("params")
    assert params is not None and all(k in params for k in REQUIRED_PARAMS), ("needs params")

    data = dict(data)
    data["admissible_ex"] = compute_admissible_actions(
        T = params["T"], 
        N = params["N"],
        C_min = params["C_min"],
        C_max = params["C_max"],
        q_max = params["q_max"],
        K_sub=1,
    )

    print(f"Simulating {num_paths:,} evaluation paths (seed={seed}) ...")
    paths = simulate_paths(data, num_paths=num_paths, seed=seed)

    print("Validating MC test against the DP lattice price ...")
    dp_policy_fn = make_dp_policy(data)
    dp_value, dp_forced_frac = evaluate_policy(paths, dp_policy_fn, data)
    gc.collect()

    mc_mean = float(dp_value.mean())
    mc_se = float(dp_value.std(ddof=1) / np.sqrt(len(dp_value)))
    ci_lo, ci_hi = mc_mean - 1.96 * mc_se, mc_mean + 1.96 * mc_se
    lattice_price = float(data["option_price"])
    ci_half_width_pct = 100 * 1.96 * mc_se / lattice_price if lattice_price else float("nan")
    passed = ci_lo <= lattice_price <= ci_hi
    snap = dp_policy_fn.snap_stats
    mean_snap_error = (snap["sum_abs_err"] / snap["n"]) if snap["n"] else float("nan")

    validation = dict(
        n_paths=num_paths, seed=seed, mc_mean=mc_mean, mc_se=mc_se,
        ci_lo=ci_lo, ci_hi=ci_hi, ci_half_width_pct=ci_half_width_pct,
        lattice_price=lattice_price, passed=bool(passed),
        mean_snap_error=mean_snap_error, forced_volume_frac=dp_forced_frac,
    )

    # actual paths
    results = {"optimal_dp": dict(value=dp_value, forced_volume_frac=dp_forced_frac)}

    pr_value, pr_forced = evaluate_policy(paths, make_pro_rata(data), data)
    results["pro_rata"] = dict(value=pr_value, forced_volume_frac=pr_forced)
    gc.collect()

    ds_value, ds_forced = evaluate_policy(paths, make_deterministic_seasonal(data), data)
    results["deterministic_seasonal"] = dict(value=ds_value, forced_volume_frac=ds_forced)
    gc.collect()

    fk_value, fk_forced = evaluate_policy(paths, make_fixed_threshold(data, params["K"]), data)
    results["fixed_threshold_K"] = dict(value=fk_value, forced_volume_frac=fk_forced)
    gc.collect()

    S_bar_opt, sbar_grid, in_sample_optimum = fit_best_constant(data, seed_opt, num_paths_opt)
    bc_value, bc_forced = evaluate_policy(paths, make_fixed_threshold(data, S_bar_opt), data)
    results["best_constant_threshold"] = dict(value=bc_value, forced_volume_frac=bc_forced, S_bar=S_bar_opt, in_sample_optimum=in_sample_optimum)
    gc.collect()

    # create table
    row_order = [
        ("pro_rata", "Pro-rata"),
        ("deterministic_seasonal", "Deterministic seasonal"),
        ("fixed_threshold_K", "Fixed threshold at K"),
        ("best_constant_threshold", "Best constant threshold"),
        ("optimal_dp", "Optimal (DP)"),
    ]

    table_rows = []
    for key, label in row_order:
        v = results[key]["value"]
        mean_v = float(v.mean())
        se_v = float(v.std(ddof=1) / np.sqrt(len(v)))
        delta = dp_value - v
        gap = float(delta.mean())
        gap_se = float(delta.std(ddof=1) / np.sqrt(len(delta))) if key != "optimal_dp" else 0.0
        gap_pct = 100 * gap / mc_mean if mc_mean else float("nan")
        table_rows.append(dict(
            policy=label,
            mc_value=mean_v,
            mc_se=se_v,
            gap_vs_dp=gap,
            gap_pct_of_dp=gap_pct,
            paired_se_of_gap=gap_se,
            clamp_raised_volume_pct=100 * results[key]["forced_volume_frac"],
        ))

    K = params["K"]
    offset = S_bar_opt - K
    print(f"\nFitted S_bar = {S_bar_opt:.4f}  (K = {K:.4f}, offset = {offset:+.4f})")

    out = dict(
        validation=validation,
        table=table_rows,
        S_bar_grid_search=sbar_grid,
        S_bar_fitted=S_bar_opt,
        S_bar_in_sample_optimum=in_sample_optimum,
        n_paths=num_paths,
        num_paths_opt=num_paths_opt,
        seed=seed,
        seed_opt=seed_opt,
        K_sub=data["K_sub"],
    )
    json_path = os.path.join(output_dir, "naive_policies.json")
    with open(json_path, "w") as f:
        json.dump(_to_py(out), f, indent=2)
    print(f"\nJSON written: {json_path}")

    return out
