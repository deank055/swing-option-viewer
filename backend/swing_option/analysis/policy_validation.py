# this file validates teh policy, for the validation section in the results chapter

import gc
import json
import os

import numpy as np

from swing_option.core.lattice import build_price_grid
from swing_option.core.pricing import compute_admissible_actions, compute_option_value
from swing_option.analysis.optimal_strategy import extract_boundary, interpolate_boundary, report_boundary_fallback
from swing_option.analysis.policy_analysis import simulate_paths, clamp_admissible, evaluate_policy, make_dp_policy

REQUIRED_PARAM_KEYS = ("T", "N", "sigma", "r", "kappa", "alpha", "S_0", "K", "C_max", "C_min", "q_max")

PERT_SE_MULT = 2.0
SHRINK_Z_THRESHOLD = 3.0
LSMC_SE_MULT = 2.0

def _require_params(params):
    missing = [k for k in REQUIRED_PARAM_KEYS if k not in params]
    if missing:
        raise KeyError(f"missing required keys: {missing}")
    return params

def _to_py(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float):
        return None if (obj != obj) else obj
    if isinstance(obj, np.ndarray):
        return [_to_py(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {k: _to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_py(x) for x in obj]
    return obj

def _paired_gap(a, b):
    diff = np.asarray(a) - np.asarray(b)
    n = diff.size
    mean = float(diff.mean())
    se = float(diff.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return mean, se

def _load_naive_policies(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"policy_validation: {path} not found")
    with open(path) as f:
        return json.load(f)

def _mc_under_dp_policy_block(naive, source_path):
    val = naive["validation"]
    fields = ("n_paths", "seed", "mc_mean", "mc_se", "ci_lo", "ci_hi", "ci_half_width_pct", "lattice_price", "passed", "mean_snap_error")
    block = {k: val[k] for k in fields if k in val}
    block["source"] = source_path
    return block

def _best_constant_row(naive):
    for row in naive.get("table", []):
        if row.get("policy") == "Best constant threshold":
            return row
    return None

def _boundary_interp_ex(work_data):
    params = work_data["params"]
    grid, policy, admissible, V = work_data["grid"], work_data["policy"], work_data["admissible"], work_data["V"]
    m, delta_X, delta_t, K_sub = work_data["m"], work_data["delta_X"], work_data["delta_t"], work_data["K_sub"]

    boundary_full, _ = extract_boundary(grid, policy, admissible, params["alpha"], delta_t)
    interp_data = {"grid": grid, "policy": policy, "V": V, "m": m, "delta_X": delta_X, "delta_t": delta_t}
    boundary_interp_full, diag = interpolate_boundary(
        interp_data, boundary_full, kappa=params["kappa"], sigma=params["sigma"],
        K=params["K"], r=params["r"], alpha_func=params["alpha"],
    )
    boundary_interp_full = boundary_interp_full.copy()
    boundary_interp_full[-1, :] = np.nan

    ex_idx = np.arange(0, grid.shape[0], K_sub)
    return boundary_interp_full[ex_idx], diag

def _evaluate_policy_with_actions(paths, policy_fn, data):
    params = data["params"]
    K, r, N, T = params["K"], params["r"], params["N"], params["T"]
    dt = T / N
    disc = np.exp(-r * dt * np.arange(N + 1))

    num_paths = paths.shape[0]
    C = np.zeros(num_paths, dtype=int)
    value = np.zeros(num_paths, dtype=float)
    total_q = np.zeros(num_paths, dtype=float)
    raised_q = np.zeros(num_paths, dtype=float)
    actions = np.zeros((num_paths, N + 1), dtype=np.int16)

    for i in range(N + 1):
        q_req = policy_fn(i, paths[:, i], C)
        q, raised = clamp_admissible(q_req, i, C, data)
        actions[:, i] = q
        value += disc[i] * q * (paths[:, i] - K)
        raised_q += np.where(raised, q, 0)
        total_q += q
        C += q

    denom = total_q.sum()
    forced_volume_frac = float(raised_q.sum() / denom) if denom > 0 else 0.0
    return value, forced_volume_frac, actions

def _make_threshold_policy(boundary_ex, delta_fn, q_max):
    def policy_fn(i, S_vec, C_vec):
        S_star = boundary_ex[i, C_vec]
        thresh = S_star + delta_fn(i, C_vec)
        return np.where(np.isnan(S_star), 0.0, np.where(S_vec >= thresh, float(q_max), 0.0))
    return policy_fn

def _make_shrink_policy(boundary_ex, lam, S_bar, q_max):
    def policy_fn(i, S_vec, C_vec):
        S_star = boundary_ex[i, C_vec]
        thresh = lam * S_star + (1.0 - lam) * S_bar
        return np.where(np.isnan(S_star), 0.0, np.where(S_vec >= thresh, float(q_max), 0.0))
    return policy_fn

def _member_row(label, baseline_value, value, baseline_actions, actions):
    gap, se = _paired_gap(baseline_value, value)
    frac_changed = float(np.mean(actions != baseline_actions))
    return dict(
        label=label, mc_mean=float(np.mean(value)),
        mc_se=float(np.std(value, ddof=1) / np.sqrt(len(value))),
        gap_vs_baseline=gap, paired_se=se,
        fraction_decisions_changed=frac_changed,
    )

def _run_perturbation_test(work_data, eval_paths, boundary_ex, delta_X, S_bar, best_constant_row, families, level_steps):
    params = work_data["params"]
    q_max, C_max = params["q_max"], params["C_max"]

    baseline_fn = _make_threshold_policy(boundary_ex, lambda i, C: 0.0, q_max)
    baseline_value, _, baseline_actions = _evaluate_policy_with_actions(eval_paths, baseline_fn, work_data)

    report = {"level": None, "tilt_C": None, "tilt_t": None, "shrink": None}
    fail_reasons = []

    if "level" in families:
        rows = []
        for d in sorted(set(level_steps) | {0}):
            delta_usd = d * delta_X
            fn = _make_threshold_policy(boundary_ex, lambda i, C, dd=delta_usd: dd, q_max)
            val, _, act = _evaluate_policy_with_actions(eval_paths, fn, work_data)
            row = _member_row(f"d={d:+d}", baseline_value, val, baseline_actions, act)
            row["d_steps"] = d
            row["delta_usd"] = delta_usd
            rows.append(row)
            del val, act
            gc.collect()
        rows.sort(key=lambda r: r["d_steps"])
        report["level"] = rows
        for r in rows:
            if r["gap_vs_baseline"] < -PERT_SE_MULT * r["paired_se"]:
                fail_reasons.append(f"level {r['label']}: gap={r['gap_vs_baseline']:.4f} "
                                     f"< -{PERT_SE_MULT}*SE={-PERT_SE_MULT * r['paired_se']:.4f}")
        best = max(rows, key=lambda r: r["mc_mean"])
        if best["d_steps"] != 0:
            fail_reasons.append(f"level family maximum at d={best['d_steps']} "
                                 f"(mc_mean={best['mc_mean']:.4f}), expected d=0")

    if "tilt_C" in families:
        rows = []
        for s in (1, 2, 4):
            for sign in (+1, -1):
                target = sign * s
                beta = sign * s * delta_X / (C_max / 2.0)
                fn = _make_threshold_policy(boundary_ex, lambda i, C, b=beta: b * (C - C_max / 2.0), q_max)
                val, _, act = _evaluate_policy_with_actions(eval_paths, fn, work_data)
                row = _member_row(f"target_steps={target:+d}", baseline_value, val, baseline_actions, act)
                row["target_steps"] = target
                row["beta"] = beta
                rows.append(row)
                del val, act
                gc.collect()
        report["tilt_C"] = rows
        for r in rows:
            if r["gap_vs_baseline"] < -PERT_SE_MULT * r["paired_se"]:
                fail_reasons.append(f"tilt_C {r['label']}: gap={r['gap_vs_baseline']:.4f} "
                                     f"< -{PERT_SE_MULT}*SE={-PERT_SE_MULT * r['paired_se']:.4f}")

    if "tilt_t" in families:
        rows = []
        T_val, N_val = params["T"], params["N"]
        for s in (1, 2, 4):
            for sign in (+1, -1):
                target = sign * s
                gamma = sign * 2.0 * s * delta_X
                fn = _make_threshold_policy(
                    boundary_ex, lambda i, C, g=gamma, T_=T_val, N_=N_val: g * ((i * T_ / N_) - 0.5), q_max,
                )
                val, _, act = _evaluate_policy_with_actions(eval_paths, fn, work_data)
                row = _member_row(f"target_steps={target:+d}", baseline_value, val, baseline_actions, act)
                row["target_steps"] = target
                row["gamma"] = gamma
                rows.append(row)
                del val, act
                gc.collect()
        report["tilt_t"] = rows
        for r in rows:
            if r["gap_vs_baseline"] < -PERT_SE_MULT * r["paired_se"]:
                fail_reasons.append(f"tilt_t {r['label']}: gap={r['gap_vs_baseline']:.4f} "
                                     f"< -{PERT_SE_MULT}*SE={-PERT_SE_MULT * r['paired_se']:.4f}")

    if "shrink" in families:
        rows = []
        for lam in (0.0, 0.25, 0.5, 0.75, 1.0, 1.25):
            fn = _make_shrink_policy(boundary_ex, lam, S_bar, q_max)
            val, _, act = _evaluate_policy_with_actions(eval_paths, fn, work_data)
            row = _member_row(f"lambda={lam}", baseline_value, val, baseline_actions, act)
            row["lam"] = lam
            rows.append(row)
            del val, act
            gc.collect()
        report["shrink"] = rows
        for r in rows:
            if r["gap_vs_baseline"] < -PERT_SE_MULT * r["paired_se"]:
                fail_reasons.append(f"shrink {r['label']}: gap={r['gap_vs_baseline']:.4f} "
                                     f"< -{PERT_SE_MULT}*SE={-PERT_SE_MULT * r['paired_se']:.4f}")
        best_shrink = max(rows, key=lambda r: r["mc_mean"])
        if abs(best_shrink["lam"] - 1.0) > 1e-9:
            fail_reasons.append(f"shrink family maximum at lambda={best_shrink['lam']}, expected lambda=1.0")

        lam0 = next(r for r in rows if r["lam"] == 0.0)
        if best_constant_row is not None:
            diff = lam0["mc_mean"] - best_constant_row["mc_value"]
            combined_se = float(np.sqrt(lam0["mc_se"] ** 2 + best_constant_row["mc_se"] ** 2))
            z = diff / combined_se if combined_se > 0 else float("nan")
            lam0["diff_vs_naive_best_constant"] = diff
            lam0["combined_se_vs_naive_best_constant"] = combined_se
            lam0["z_vs_naive_best_constant"] = z
            if not np.isnan(z) and abs(z) > SHRINK_Z_THRESHOLD:
                fail_reasons.append(
                    f"shrink lambda=0 ({lam0['mc_mean']:.4f}) vs naive-policies best-constant "
                    f"({best_constant_row['mc_value']:.4f}): diff={diff:.4f}, z={z:.2f} "
                    f"(> {SHRINK_Z_THRESHOLD} SE)"
                )
        else:
            lam0["diff_vs_naive_best_constant"] = None
            fail_reasons.append("shrink lambda=0 vs naive best-constant: naive_policies.json "
                                 "had no 'Best constant threshold' row to compare against")

    report["pass"] = len(fail_reasons) == 0
    report["fail_reasons"] = fail_reasons

    return report, baseline_value, baseline_actions

def _fit_lsmc(work_data, admissible_ex, fit_paths, degree, N_slots, C_max, q_max):
    params = work_data["params"]
    K, r, T, N = params["K"], params["r"], params["T"], params["N"]
    alpha_func = params["alpha"]
    dt = T / N
    disc = np.exp(-r * dt * np.arange(N + 1))
    num_paths = fit_paths.shape[0]
    t_arr = np.arange(N + 1) * dt
    X_fit = fit_paths - alpha_func(t_arr)[None, :]

    mean_by_i = X_fit.mean(axis=0)
    std_by_i = X_fit.std(axis=0, ddof=0)
    std_safe = np.where(std_by_i > 1e-12, std_by_i, 1.0)

    n_arr = np.arange(N_slots + 1)
    C_of_n = C_max - n_arr * q_max
    wait_admissible = admissible_ex[np.arange(N + 1)[:, None], C_of_n[None, :], 0]  # (N+1, N_slots+1)

    coeffs_by_i = [None] * N

    S_N = fit_paths[:, N]
    exercise_pv_N = disc[N] * q_max * (S_N - K)
    forced_row_N = ~wait_admissible[N, :]
    decide_N = forced_row_N[None, :] | (exercise_pv_N[:, None] > 0)
    decide_N[:, 0] = False
    V = np.where(decide_N, exercise_pv_N[:, None], 0.0)

    for i in range(N - 1, -1, -1):
        Xs = (X_fit[:, i] - mean_by_i[i]) / std_safe[i]
        Phi = np.vstack([Xs ** d for d in range(degree + 1)]).T
        coeffs, *_ = np.linalg.lstsq(Phi, V, rcond=None)
        coeffs_by_i[i] = coeffs

        cont_hat = Phi @ coeffs
        S_i = fit_paths[:, i]
        exercise_pv = disc[i] * q_max * (S_i - K)

        shifted_hat = np.concatenate([np.zeros((num_paths, 1)), cont_hat[:, :-1]], axis=1)
        fitted_exercise = exercise_pv[:, None] + shifted_hat

        forced_row = ~wait_admissible[i, :]
        decide = forced_row[None, :] | (fitted_exercise > cont_hat)
        decide[:, 0] = False

        shifted_V = np.concatenate([np.zeros((num_paths, 1)), V[:, :-1]], axis=1)
        realized_exercise = exercise_pv[:, None] + shifted_V
        V = np.where(decide, realized_exercise, V)

    return dict(coeffs_by_i=coeffs_by_i, mean_by_i=mean_by_i, std_by_i=std_safe, wait_admissible=wait_admissible, in_sample_value=float(V[:, N_slots].mean()))

def make_lsmc_policy(fit, params, C_max, q_max, N_slots):
    K, r, T, N = params["K"], params["r"], params["T"], params["N"]
    alpha_func = params["alpha"]
    dt = T / N
    disc = np.exp(-r * dt * np.arange(N + 1))
    coeffs_by_i, mean_by_i, std_by_i = fit["coeffs_by_i"], fit["mean_by_i"], fit["std_by_i"]

    def policy_fn(i, S_vec, C_vec):
        n_vec = np.clip(np.round((C_max - C_vec) / q_max).astype(int), 0, N_slots)
        X_vec = S_vec - alpha_func(i * dt)
        Xs = (X_vec - mean_by_i[i]) / std_by_i[i]
        exercise_pv = disc[i] * q_max * (S_vec - K)
        idx = np.arange(S_vec.shape[0])

        if i == N:
            decide = exercise_pv > 0
        else:
            coeffs = coeffs_by_i[i]
            degree = coeffs.shape[0] - 1
            Phi = np.vstack([Xs ** d for d in range(degree + 1)]).T
            cont_hat = Phi @ coeffs
            n_minus1 = np.clip(n_vec - 1, 0, N_slots)
            fitted_wait = cont_hat[idx, n_vec]
            fitted_exercise = exercise_pv + cont_hat[idx, n_minus1]
            decide = fitted_exercise > fitted_wait

        decide = decide & (n_vec > 0)
        return np.where(decide, float(q_max), 0.0)

    return policy_fn

def _make_snapped_decision_wrapper(policy_fn, work_data):
    m, delta_X = work_data["m"], work_data["delta_X"]
    alpha_func = work_data["params"]["alpha"]
    dt = work_data["params"]["T"] / work_data["params"]["N"]
    n_nodes = 2 * m + 1

    def wrapped(i, S_vec, C_vec):
        alpha_i = alpha_func(i * dt)
        X_vec = S_vec - alpha_i
        j = np.clip(np.round(X_vec / delta_X).astype(int) + m, 0, n_nodes - 1)
        S_node = alpha_i + (j - m) * delta_X
        return policy_fn(i, S_node, C_vec)

    return wrapped

def _run_lsmc(work_data, admissible_ex, num_paths, seed_fit, eval_paths, dp_value_eval, degrees):
    params = work_data["params"]
    C_max, q_max = params["C_max"], params["q_max"]
    N_slots = C_max // q_max

    fit_paths = simulate_paths(work_data, num_paths=num_paths, seed=seed_fit)

    out = {}
    flagged = []
    for degree in degrees:
        fit = _fit_lsmc(work_data, admissible_ex, fit_paths, degree, N_slots, C_max, q_max)
        policy_fn = make_lsmc_policy(fit, params, C_max, q_max, N_slots)
        lsmc_value, lsmc_forced_frac = evaluate_policy(eval_paths, policy_fn, work_data)
        gap, se = _paired_gap(dp_value_eval, lsmc_value)  # DP - LSMC: should be >= 0 within noise
        exceeds = gap < -LSMC_SE_MULT * se

        snapped_comparison = None
        if exceeds:
            flagged.append(degree)
            snapped_fn = _make_snapped_decision_wrapper(policy_fn, work_data)
            lsmc_value_snapped, lsmc_forced_snapped = evaluate_policy(eval_paths, snapped_fn, work_data)
            gap_snapped, se_snapped = _paired_gap(dp_value_eval, lsmc_value_snapped)
            exceeds_snapped = gap_snapped < -LSMC_SE_MULT * se_snapped
            snapped_comparison = dict(
                lsmc_value=float(np.mean(lsmc_value_snapped)), gap_vs_dp=gap_snapped,
                paired_se=se_snapped, exceeds_dp_flag=bool(exceeds_snapped),
                forced_volume_frac=lsmc_forced_snapped,
            )
            del lsmc_value_snapped
        out[f"degree_{degree}"] = dict(
            lsmc_value=float(np.mean(lsmc_value)),
            lsmc_se=float(np.std(lsmc_value, ddof=1) / np.sqrt(len(lsmc_value))),
            gap_vs_dp=gap, paired_se=se, exceeds_dp_flag=bool(exceeds),
            forced_volume_frac=lsmc_forced_frac, in_sample_value=fit["in_sample_value"],
            snapped_comparison=snapped_comparison,
        )
        del fit, lsmc_value
        gc.collect()

    resolution_explained = [
        d for d in flagged
        if out[f"degree_{d}"]["snapped_comparison"] is not None
        and not out[f"degree_{d}"]["snapped_comparison"]["exceeds_dp_flag"]
    ]
    out["pass"] = len(flagged) == 0
    out["degrees_run"] = list(degrees)
    out["flagged_degrees"] = flagged
    out["flagged_degrees_resolution_explained"] = resolution_explained
    out["flagged_degrees_unexplained"] = [d for d in flagged if d not in resolution_explained]
    return out

FALLBACK_WARN_THRESHOLD = 0.10
POSITIVE_CONTROL_HEALTHY_MAX = 0.01
POSITIVE_CONTROL_FAULT_MIN = 0.10
POSITIVE_CONTROL_FAULT_K = 2.40

def _load_json_or_none(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

def _check_monotonicity_S(structure_path):
    struct = _load_json_or_none(structure_path)
    mono = struct["monotonicity_S"]
    block = {
        "states_checked": mono.get("states_checked"),
        "violations": mono.get("violation_count"),
        "source": structure_path,
    }
    return block

def _check_independent_threshold(naive, naive_path, opt_strategy_path):
    opt = _load_json_or_none(opt_strategy_path)
    s_bar = naive.get("S_bar_fitted")
    mean_path = None
    if opt is not None:
        mean_path = opt.get("mean_path_boundary", {}).get("mean_interpolated")

    diff = float(mean_path) - float(s_bar)

    return {
        "s_bar_lsmc_free_fit": float(s_bar), "mean_path_boundary": float(mean_path),
        "difference": diff, "sources": [naive_path, opt_strategy_path],
    }

def _fallback_aggregate(structure_path, sensitivity_path):
    rows = []

    struct = _load_json_or_none(structure_path)
    if struct is not None and "boundary_fallback" in struct:
        rows.append({"run": "baseline (structure_properties.json)", "fallback_fraction": struct["boundary_fallback"].get("fallback_fraction")})

    sens = _load_json_or_none(sensitivity_path)
    if sens is not None:
        for entry in sens.get("kappa_sweep", []) or []:
            rows.append({"run": f"kappa={entry.get('kappa')}", "fallback_fraction": entry.get("fallback_fraction")})
        for entry in sens.get("sigma_sweep", []) or []:
            rows.append({"run": f"sigma={entry.get('sigma')}", "fallback_fraction": entry.get("fallback_fraction")})
        for entry in sens.get("cmin_sweep", []) or []:
            rows.append({"run": f"C_min={entry.get('C_min')}", "fallback_fraction": entry.get("fallback_fraction")})
        for entry in sens.get("strike_sweep", []) or []:
            rows.append({"run": f"K={entry.get('K')}", "fallback_fraction": entry.get("fallback_fraction")})
        for label, entry in (sens.get("qmax_check", {}) or {}).items():
            rows.append({"run": label, "fallback_fraction": entry.get("fallback_fraction")})

    rows = [r for r in rows if r["fallback_fraction"] is not None]
    sources = [structure_path, sensitivity_path]

    worst = max(rows, key=lambda r: r["fallback_fraction"])

    return {"runs": rows, "max_fraction": worst["fallback_fraction"], "max_run": worst["run"], "sources": sources}

def _solve_for_positive_control(params, K_sub):
    alpha_func = params["alpha"]
    X_0 = params["S_0"] - alpha_func(0)
    grid, delta_t, delta_X, m, col_X0 = build_price_grid(
        T=params["T"], sigma=params["sigma"], X_0=X_0, N=params["N"],
        kappa=params["kappa"], K_sub=K_sub,
    )
    admissible = compute_admissible_actions(
        T=params["T"], N=params["N"], C_min=params["C_min"], C_max=params["C_max"], q_max=params["q_max"], K_sub=K_sub,
    )
    V, policy, option_price = compute_option_value(
        delta_t=delta_t,
        delta_X=delta_X,
        m=m,
        col_X0=col_X0,
        N=params["N"],
        admissible=admissible,
        K=params["K"],
        r=params["r"],
        kappa=params["kappa"],
        alpha_func=alpha_func,
        sigma=params["sigma"],
        C_max=params["C_max"],
        q_max=params["q_max"],
        K_sub=K_sub,
    )
    return dict(grid=grid, delta_t=delta_t, delta_X=delta_X, m=m, col_X0=col_X0, admissible=admissible, V=V, policy=policy, K_sub=K_sub)

def _fallback_positive_control(params):
    solved = _solve_for_positive_control(params, K_sub=100)

   
    boundary_full, _ = extract_boundary(
        solved["grid"], solved["policy"], solved["admissible"], params["alpha"], solved["delta_t"],
    )
    interp_data = {"grid": solved["grid"], "policy": solved["policy"], "V": solved["V"], "m": solved["m"], "delta_X": solved["delta_X"], "delta_t": solved["delta_t"]}

    _, healthy_diag = interpolate_boundary(
        interp_data, boundary_full, kappa=params["kappa"], sigma=params["sigma"],
        K=params["K"], r=params["r"], alpha_func=params["alpha"],
    )
    healthy_fraction = healthy_diag["fallback_fraction"]

    fault_boundary_interp, fault_diag = interpolate_boundary(
        interp_data, boundary_full, kappa=params["kappa"], sigma=params["sigma"],
        K=POSITIVE_CONTROL_FAULT_K, r=params["r"], alpha_func=params["alpha"],
    )
    fault_fraction = fault_diag["fallback_fraction"]
    del fault_boundary_interp

    del solved, boundary_full, interp_data
    gc.collect()

    passed = (
        healthy_fraction < POSITIVE_CONTROL_HEALTHY_MAX
        and fault_fraction > POSITIVE_CONTROL_FAULT_MIN
    )

    return {
        "healthy_fraction": healthy_fraction, "fault_fraction": fault_fraction,
        "fault_description": (
            f"strike mismatch K={POSITIVE_CONTROL_FAULT_K} vs true K={params['K']:.4f}, reproducing "
            "the historical parameter-threading bug signature (solver and interpolator disagreeing "
            "on a physical parameter)"
        ),
        "K_sub": 100, "pass": passed,
    }

def _check_extracted_boundary(params, naive, naive_path, output_dir):
    structure_path = os.path.join(output_dir, "structure_properties.json")
    opt_strategy_path = os.path.join(output_dir, "optimal_strategy_report.json")
    sensitivity_path = os.path.join(output_dir, "sensitivity_values.json")

    monotonicity_S = _check_monotonicity_S(structure_path)

    independent_threshold = _check_independent_threshold(naive, naive_path, opt_strategy_path)
    fallback_aggregate = _fallback_aggregate(structure_path, sensitivity_path)
    positive_control = _fallback_positive_control(params)

    return {
        "monotonicity_S": monotonicity_S,
        "independent_threshold": independent_threshold,
        "fallback_aggregate": fallback_aggregate,
        "positive_control": positive_control,
    }

DEGENERATE_LSMC_SE_MULT = 2.0

def _run_degenerate_lsmc(baseline_params, K_sub=100, num_paths=250_000, seed=1337, seed_fit=420, degrees=(2, 3, 4)):
    from swing_option.analysis import validation as _validation_mod

    check_C_result = _validation_mod._single_right_bound_check(baseline_params, [K_sub])
    ok_resolutions = [r for r in check_C_result["resolutions"] if r["K_sub"] == K_sub and r["status"] == "ok"]
    prod_resolution = ok_resolutions[0]
    lattice_price = prod_resolution["V_dp"]
    lower_bound_analytic = check_C_result["lower_bound"]
    upper_bound_analytic = check_C_result["upper_bound"]
    X_0_snap = prod_resolution["col_X0_snap"]["snapped_X0"]

    single_params = dict(baseline_params)
    single_params["q_max"] = 1
    single_params["C_max"] = 1
    single_params["C_min"] = 0
    C_max, q_max = single_params["C_max"], single_params["q_max"]
    N_slots = C_max // q_max

    T, N, K, r = single_params["T"], single_params["N"], single_params["K"], single_params["r"]
    kappa, sigma, alpha_func = single_params["kappa"], single_params["sigma"], single_params["alpha"]
    t = _validation_mod._exercise_dates(T, N)
    discount = np.exp(-r * t)
    mean_snap, sd_t = _validation_mod._ou_law(t, X_0_snap, kappa, sigma)
    mu_snap = alpha_func(t) + mean_snap
    call_snap = _validation_mod._bachelier_call(mu_snap, sd_t, K)
    lower_bound_at_snap = float(np.max(discount * call_snap))
    snap_allowance = abs(lower_bound_at_snap - lower_bound_analytic)

    admissible_ex = compute_admissible_actions(
        T=T, N=N, C_min=single_params["C_min"], C_max=C_max, q_max=q_max, K_sub=1,
    )
    work_data = {"params": single_params, "admissible_ex": admissible_ex}

    fit_paths = simulate_paths(work_data, num_paths=num_paths, seed=seed_fit)
    eval_paths = simulate_paths(work_data, num_paths=num_paths, seed=seed)

    degree_out = {}
    for degree in degrees:
        fit = _fit_lsmc(work_data, admissible_ex, fit_paths, degree, N_slots, C_max, q_max)
        policy_fn = make_lsmc_policy(fit, single_params, C_max, q_max, N_slots)
        lsmc_value, lsmc_forced_frac = evaluate_policy(eval_paths, policy_fn, work_data)
        lsmc_mean = float(np.mean(lsmc_value))
        lsmc_se = float(np.std(lsmc_value, ddof=1) / np.sqrt(len(lsmc_value)))
        diff = lattice_price - lsmc_mean
        upper_tol = DEGENERATE_LSMC_SE_MULT * lsmc_se + snap_allowance
        within_bound = bool(-1e-9 <= diff <= upper_tol)  # tiny float-noise floor at 0
        exceeds_lattice_flag = bool(diff < -DEGENERATE_LSMC_SE_MULT * lsmc_se)
        denom = lattice_price - lower_bound_analytic
        gap_closed_fraction = ((lsmc_mean - lower_bound_analytic) / denom) if denom != 0 else float("nan")

        degree_out[f"degree_{degree}"] = dict(
            lsmc_value=lsmc_mean, lsmc_se=lsmc_se, diff_vs_lattice=diff,
            within_bound=within_bound, exceeds_lattice_flag=exceeds_lattice_flag,
            gap_closed_fraction=gap_closed_fraction, forced_volume_frac=lsmc_forced_frac,
            in_sample_value=fit["in_sample_value"],
        )
        del fit, lsmc_value
        gc.collect()

    passed = all(degree_out[f"degree_{d}"]["within_bound"] for d in degrees)

    return {
        "K_sub": K_sub, "n_paths": num_paths, "seed_eval": seed, "seed_fit": seed_fit,
        "single_right_params": {"q_max": q_max, "C_max": C_max, "C_min": single_params["C_min"]},
        "lattice_price": lattice_price,
        "analytic_lower_bound": lower_bound_analytic, "analytic_upper_bound": upper_bound_analytic,
        "snap_allowance": snap_allowance,
        "degrees_run": list(degrees),
        **degree_out,
        "pass": passed,
    }

def analyse(data, output_dir, num_paths=250_000, seed=1337, seed_fit=420, naive_policies_path=None, run_degenerate_lsmc=True):
    params = _require_params(data["params"])

    families = ["level", "tilt_C", "tilt_t", "shrink"]
    level_steps = [-8, -4, -2, -1, 1, 2, 4, 8]
    lsmc_degrees = [2, 3, 4]

    naive_path = naive_policies_path or os.path.join(output_dir, "naive_policies.json")
    naive = _load_naive_policies(naive_path)

    mc_under_dp = _mc_under_dp_policy_block(naive, naive_path)

    admissible_ex = compute_admissible_actions(T=params["T"], N=params["N"], C_min=params["C_min"], C_max=params["C_max"], q_max=params["q_max"], K_sub=1)
    work_data = dict(data)
    work_data["admissible_ex"] = admissible_ex
    work_data["params"] = params

    boundary_ex, boundary_diag = _boundary_interp_ex(work_data)
    report_boundary_fallback(boundary_diag)
    delta_X = work_data["delta_X"]
    S_bar = float(naive.get("S_bar_fitted"))
    best_constant_row = _best_constant_row(naive)

    eval_paths = simulate_paths(work_data, num_paths=num_paths, seed=seed)
    dp_value_eval, _, _ = _evaluate_policy_with_actions(eval_paths, make_dp_policy(work_data), work_data)

    perturbation, baseline_value, _ = _run_perturbation_test(
        work_data, eval_paths, boundary_ex, delta_X, S_bar, best_constant_row, families, level_steps,
    )
    recon_gap, recon_se = _paired_gap(dp_value_eval, baseline_value)
    reconstruction_cost = dict(
        gap=recon_gap, paired_se=recon_se,
        dp_mean=float(np.mean(dp_value_eval)), reconstruction_mean=float(np.mean(baseline_value)),
    )

    if lsmc_degrees:
        lsmc = _run_lsmc(work_data, admissible_ex, num_paths, seed_fit, eval_paths, dp_value_eval, lsmc_degrees)
    else:
        lsmc = {"skipped": True, "degrees_run": [], "pass": True}

    extracted_boundary = _check_extracted_boundary(params, naive, naive_path, output_dir)

    degenerate_lsmc = _run_degenerate_lsmc(params)

    out = {
        "optimal_policy": {
            "mc_under_dp_policy": mc_under_dp,
            "reconstruction_cost": reconstruction_cost,
            "perturbation": perturbation,
            "lsmc": lsmc,
            "extracted_boundary": extracted_boundary,
            "K_sub": work_data["K_sub"], "n_paths": num_paths, "seed_eval": seed, "seed_fit": seed_fit,
        },
        "value_function": {
            "degenerate_lsmc": degenerate_lsmc,
        },
    }

    json_path = os.path.join(output_dir, "validation_optimal_policy.json")
    with open(json_path, "w") as f:
        json.dump(_to_py(out), f, indent=2)

    return out
