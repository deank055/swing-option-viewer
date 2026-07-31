# this file does all the value function validation (chapter 6.6)

import gc
import json
import os

import numpy as np
from scipy.stats import norm

from swing_option.core.lattice import build_price_grid, transition_probabilities
from swing_option.core.pricing import compute_admissible_actions, compute_option_value
from swing_option.core.memory import v_memory_gb, MEMORY_LIMIT_GB
from swing_option.analysis.convergence import _peak_rss_gb, run_ladder as _convergence_run_ladder
from swing_option.analysis import convergence as _convergence_mod
from swing_option.analysis.optimal_strategy import extract_boundary, interpolate_boundary

LADDER = [1, 4, 8, 16, 32, 64, 100]
# LADDER = [1, 4, 8, 16, 32]
MEM_BUDGET_GB = MEMORY_LIMIT_GB

REL_TOL = 0.01
NOISE_FLOOR_ABS = 1e-2
BOUND_REL_EPS = 1e-6
REAL_MISMATCH_FRACTION_THRESHOLD = 0.05

REQUIRED_PARAM_KEYS = ("T", "N", "sigma", "r", "kappa", "alpha", "S_0", "K", "C_max", "C_min", "q_max")

def _require_params(params):
    missing = [k for k in REQUIRED_PARAM_KEYS if k not in params]
    if missing:
        raise KeyError(f"missing required keys: {missing}")
    return params

def _exercise_dates(T, N):
    return np.arange(N + 1) * (T / N)

def _ou_law(t, X_0, kappa, sigma):
    mean = X_0 * np.exp(-kappa * t)
    var = (sigma ** 2) * (1.0 - np.exp(-2.0 * kappa * t)) / (2.0 * kappa)
    return mean, np.sqrt(np.maximum(var, 0.0))

def _bachelier_call(mu, sd, K):
    mu = np.asarray(mu, dtype=float)
    sd = np.asarray(sd, dtype=float)
    positive = sd > 0
    d = np.zeros_like(mu)
    np.divide(mu - K, sd, out=d, where=positive)
    deterministic = np.maximum(mu - K, 0.0)
    return np.where(positive, sd * norm.pdf(d) + (mu - K) * norm.cdf(d), deterministic)

def _col_X0_snap(params, K_sub):
    N_fine = params["N"] * K_sub
    delta_t = params["T"] / N_fine
    delta_X = params["sigma"] * np.sqrt(3.0 * delta_t)
    X_0 = params["S_0"] - params["alpha"](0)
    j_X0 = int(round(X_0 / delta_X))
    snapped_X0 = j_X0 * delta_X
    return {"delta_X": float(delta_X), "X_0": float(X_0), "snapped_X0": float(snapped_X0), "snap_bias": float(snapped_X0 - X_0)}

def _solve(params, K_sub):
    alpha_func = params["alpha"]
    X_0 = params["S_0"] - alpha_func(0)
    grid, delta_t, delta_X, m, col_X0 = build_price_grid(
        T=params["T"], sigma=params["sigma"], X_0=X_0, N=params["N"],
        kappa=params["kappa"], K_sub=K_sub,
    )
    admissible = compute_admissible_actions(
        T=params["T"], N=params["N"], C_min=params["C_min"],
        C_max=params["C_max"], q_max=params["q_max"], K_sub=K_sub,
    )
    V, policy, option_price = compute_option_value(
        delta_t=delta_t, delta_X=delta_X, m=m, col_X0=col_X0, N=params["N"],
        admissible=admissible, K=params["K"], r=params["r"], kappa=params["kappa"],
        alpha_func=alpha_func, sigma=params["sigma"], C_max=params["C_max"],
        q_max=params["q_max"], K_sub=K_sub,
    )
    return {
        "grid": grid, "delta_t": delta_t, "delta_X": delta_X, "m": m, "col_X0": col_X0,
        "admissible": admissible, "V": V, "policy": policy, "option_price": float(option_price),
    }

def _run_ladder_with_memory_guard(params, K_sub_list, per_resolution_fn):
    rows = []
    for K_sub in K_sub_list:
        predicted_gb = v_memory_gb(params, K_sub)
        if predicted_gb > MEM_BUDGET_GB:
            rows.append({"K_sub": K_sub, "status": "skipped_oom", "predicted_gb": predicted_gb, "peak_rss_gb": None})
            continue
        row = per_resolution_fn(params, K_sub)
        row["predicted_gb"] = predicted_gb
        rows.append(row)
    return rows

def _verdict_snapped(rows):
    if any(r["status"] == "invalid_oracle" for r in rows):
        return False
    ok_rows = [r for r in rows if r["status"] == "ok"]
    if len(ok_rows) < 2:
        return False
    abs_errs = [abs(r["err_vs_snapped"]) for r in ok_rows]
    shrinking = True
    for i in range(len(abs_errs) - 1):
        cur, nxt = abs_errs[i], abs_errs[i + 1]
        if cur <= NOISE_FLOOR_ABS and nxt <= NOISE_FLOOR_ABS:
            continue
        if nxt > cur + 1e-9:
            shrinking = False
    final_rel = ok_rows[-1].get("rel_err_snapped")
    small = abs_errs[-1] <= NOISE_FLOOR_ABS or (
        final_rel is not None and not (isinstance(final_rel, float) and np.isnan(final_rel))
        and abs(final_rel) <= REL_TOL
    )
    return bool(shrinking and small)

def _verify_forced_path(admissible, N, K_sub, q_max):
    C = 0
    violations = []
    for k in range(N + 1):
        i = k * K_sub
        allowed = np.flatnonzero(admissible[i, C, :]).tolist()
        if allowed != [q_max]:
            violations.append({"i": i, "k": k, "C": C, "allowed_q": allowed})
        C += q_max
    return violations

def _fully_forced_check(params, K_sub_list):
    N, T, q_max, K = params["N"], params["T"], params["q_max"], params["K"]
    alpha_func = params["alpha"]
    X_0_true = params["S_0"] - alpha_func(0)

    forced_budget = (N + 1) * q_max
    naive_budget = N * q_max
    forced_params = dict(params)
    forced_params["C_min"] = forced_budget
    forced_params["C_max"] = forced_budget

    t = _exercise_dates(T, N)
    discount = np.exp(-params["r"] * t)

    def _V_analytic(X_0):
        mean_t, _ = _ou_law(t, X_0, params["kappa"], params["sigma"])
        return float(np.sum(discount * q_max * (alpha_func(t) + mean_t - K)))

    V_analytic_true = _V_analytic(X_0_true)

    def _resolution(p, K_sub):
        solved = _solve(p, K_sub)
        violations = _verify_forced_path(solved["admissible"], N, K_sub, q_max)
        peak_rss_gb = _peak_rss_gb()
        V_dp = solved["option_price"]
        X_0_snap = float(solved["grid"][0, solved["col_X0"]])
        del solved["V"], solved["admissible"], solved["grid"], solved["policy"]
        gc.collect()

        V_analytic_snapped = _V_analytic(X_0_snap)

        if violations:
            return {"K_sub": K_sub, "status": "invalid_oracle", "V_dp": V_dp,
                    "violations": violations, "peak_rss_gb": peak_rss_gb,
                    "X_0_true": X_0_true, "X_0_snap": X_0_snap}

        err_vs_snapped = V_dp - V_analytic_snapped
        err_vs_true = V_dp - V_analytic_true
        snap_gap = V_analytic_snapped - V_analytic_true
        rel_err_snapped = err_vs_snapped / V_analytic_snapped if V_analytic_snapped != 0 else float("nan")
        rel_err_true = err_vs_true / V_analytic_true if V_analytic_true != 0 else float("nan")

        return {"K_sub": K_sub, "status": "ok", "V_dp": V_dp,
                "V_analytic_snapped": V_analytic_snapped, "V_analytic_true": V_analytic_true,
                "X_0_true": X_0_true, "X_0_snap": X_0_snap, "X_0_diff": abs(X_0_snap - X_0_true),
                "err_vs_snapped": err_vs_snapped, "err_vs_true": err_vs_true, "snap_gap": snap_gap,
                "rel_err_snapped": rel_err_snapped, "rel_err_true": rel_err_true,
                "peak_rss_gb": peak_rss_gb}

    rows = _run_ladder_with_memory_guard(forced_params, K_sub_list, _resolution)
    return {
        "naive_budget_N_qmax": naive_budget,
        "budget_used": forced_budget,
        "V_analytic_true": V_analytic_true,
        "resolutions": rows,
        "pass": _verdict_snapped(rows),
    }

def _advantage_at_state(V, i, j, C, q_dp, q_rule, S, K, C_max, params, N_fine, m, delta_X, delta_t):
    if i == N_fine:
        val_rule = q_rule * (S - K)
    else:
        n_nodes = V.shape[1]
        is_top = j == n_nodes - 1
        is_bot = j == 0
        X_j = (j - m) * delta_X
        pu, pm, pd = transition_probabilities(
            S=X_j, delta_t=delta_t, delta_S=delta_X, kappa=params["kappa"], Theta_t=0.0,
            sigma=params["sigma"], is_top=is_top, is_bottom=is_bot, clip=True,
        )
        j_up = j if is_top else (j + 2 if is_bot else j + 1)
        j_mid = (j - 1) if is_top else (j + 1 if is_bot else j)
        j_dn = (j - 2) if is_top else (j if is_bot else j - 1)
        discount = np.exp(-params["r"] * delta_t)
        C_next_rule = min(C + q_rule, C_max)
        cont_rule = discount * (pu * V[i + 1, j_up, C_next_rule]
                                 + pm * V[i + 1, j_mid, C_next_rule]
                                 + pd * V[i + 1, j_dn, C_next_rule])
        val_rule = q_rule * (S - K) + float(cont_rule)
    return float(V[i, j, C]) - val_rule

def _decoupling_mismatch_report(solved, params, K_sub, q_max, K, C_max_used):
    policy = solved["policy"]
    admissible = solved["admissible"]
    grid = solved["grid"]
    V = solved["V"]
    delta_X = solved["delta_X"]
    delta_t = solved["delta_t"]
    m = solved["m"]
    N = params["N"]
    N_fine = N * K_sub
    alpha_func = params["alpha"]

    n_checked = 0
    n_mismatch = 0
    n_checked_reachable = 0
    n_mismatch_reachable = 0
    S_minus_K_chunks, k_chunks, C_chunks, reach_chunks, dir_chunks, ceiling_chunks = [], [], [], [], [], []
    examples = []

    C_arr = np.arange(admissible.shape[1])

    for k in range(N + 1):
        i = k * K_sub
        S_i = grid[i] + alpha_func(i * delta_t)
        valid = np.isfinite(S_i)
        expected_q = np.where(S_i > K, q_max, 0).astype(np.int64)

        adm_i = admissible[i]  # (C_max+1, q_max+1)
        adm_expected = adm_i[:, expected_q].T  # (n_nodes, C_max+1)
        ok_mask = valid[:, None] & adm_expected

        reachable_C_row = (C_arr % q_max == 0) & (C_arr <= k * q_max)
        reachable_mask = ok_mask & reachable_C_row[None, :]

        policy_i = policy[i]  # (n_nodes, C_max+1)
        mism = ok_mask & (policy_i != expected_q[:, None])
        mism_reachable = mism & reachable_C_row[None, :]

        n_checked += int(ok_mask.sum())
        n_mismatch += int(mism.sum())
        n_checked_reachable += int(reachable_mask.sum())
        n_mismatch_reachable += int(mism_reachable.sum())

        js, Cs = np.nonzero(mism)
        if js.size == 0:
            continue

        S_vals = S_i[js]
        dp_q = policy_i[js, Cs]
        rule_q = expected_q[js]

        S_minus_K_chunks.append(S_vals - K)
        k_chunks.append(np.full(js.shape, k))
        C_chunks.append(Cs)
        reach_chunks.append(reachable_C_row[Cs])
        dir_chunks.append(np.where(dp_q > rule_q, 1, -1))
        ceiling_chunks.append((Cs + q_max) > C_max_used)

        for idx in range(js.size):
            if len(examples) >= 5:
                break
            j = int(js[idx]); C = int(Cs[idx]); S = float(S_vals[idx])
            q_dp = int(dp_q[idx]); q_rule = int(rule_q[idx])
            g = _advantage_at_state(V, i, j, C, q_dp, q_rule, S, K, C_max_used,
                                     params, N_fine, m, delta_X, delta_t)
            examples.append({"i": int(i), "k": int(k), "j": j, "S": S, "C": C,
                              "q_dp": q_dp, "q_rule": q_rule, "advantage_g": g})

    if n_mismatch == 0:
        return {"n_checked": n_checked, "n_mismatch": 0, "mismatch_fraction": 0.0,
                "n_checked_reachable": n_checked_reachable, "n_mismatch_reachable": 0,
                "mismatch_fraction_reachable": 0.0, "verdict": "NONE", "examples": []}

    S_minus_K = np.concatenate(S_minus_K_chunks)
    k_arr = np.concatenate(k_chunks)
    C_arr_m = np.concatenate(C_chunks)
    reach_arr = np.concatenate(reach_chunks)
    dir_arr = np.concatenate(dir_chunks)
    ceiling_arr = np.concatenate(ceiling_chunks)

    def _dist(x):
        return {"min": float(np.min(x)), "q1": float(np.percentile(x, 25)),
                "median": float(np.median(x)), "q3": float(np.percentile(x, 75)),
                "max": float(np.max(x))}

    frac_near_strike = float(np.mean(np.abs(S_minus_K) < delta_X))
    frac_terminal = float(np.mean(k_arr == N))
    frac_i0 = float(np.mean(k_arr == 0))
    frac_unreachable = float(np.mean(~reach_arr))
    frac_ceiling_could_bind = float(np.mean(ceiling_arr))
    mismatch_fraction_reachable = (n_mismatch_reachable / n_checked_reachable) if n_checked_reachable else 0.0

    n_real = int(np.sum(reach_arr & (k_arr != 0) & (k_arr != N) & (np.abs(S_minus_K) >= delta_X)))
    frac_real_of_mismatches = n_real / n_mismatch
    verdict = "REAL" if frac_real_of_mismatches > REAL_MISMATCH_FRACTION_THRESHOLD else "BENIGN_EDGE"

    return {
        "n_checked": n_checked, "n_mismatch": n_mismatch,
        "mismatch_fraction": n_mismatch / n_checked if n_checked else 0.0,
        "n_checked_reachable": n_checked_reachable, "n_mismatch_reachable": n_mismatch_reachable,
        "mismatch_fraction_reachable": mismatch_fraction_reachable,
        "S_minus_K_dist": _dist(S_minus_K), "frac_near_strike_within_delta_X": frac_near_strike,
        "k_dist": _dist(k_arr), "frac_at_terminal_i_N": frac_terminal, "frac_at_i0": frac_i0,
        "C_dist": _dist(C_arr_m), "frac_unreachable_C": frac_unreachable,
        "frac_ceiling_could_bind": frac_ceiling_could_bind,
        "direction_dp_exercises_rule_waits": int(np.sum(dir_arr == 1)),
        "direction_dp_waits_rule_exercises": int(np.sum(dir_arr == -1)),
        "n_real_reachable_interior_far_from_strike": n_real,
        "frac_real_of_mismatches": frac_real_of_mismatches,
        "verdict": verdict,
        "examples": examples,
    }

def _decoupled_strip_check(params, K_sub_list):
    N, T, q_max, K = params["N"], params["T"], params["q_max"], params["K"]
    alpha_func = params["alpha"]
    X_0_true = params["S_0"] - alpha_func(0)

    decoupled_params = dict(params)
    decoupled_params["C_min"] = 0
    decoupled_params["C_max"] = (N + 1) * q_max

    t = _exercise_dates(T, N)
    discount = np.exp(-params["r"] * t)

    def _V_analytic(X_0):
        mean_t, sd_t = _ou_law(t, X_0, params["kappa"], params["sigma"])
        mu_t = alpha_func(t) + mean_t
        call_t = _bachelier_call(mu_t, sd_t, K)
        return float(q_max * np.sum(discount * call_t))

    V_analytic_true = _V_analytic(X_0_true)

    def _resolution(p, K_sub):
        solved = _solve(p, K_sub)
        X_0_snap = float(solved["grid"][0, solved["col_X0"]])
        mismatch_report = _decoupling_mismatch_report(
            solved, p, K_sub, q_max, K, decoupled_params["C_max"],
        )
        peak_rss_gb = _peak_rss_gb()
        V_dp = solved["option_price"]
        del solved["V"], solved["admissible"], solved["grid"], solved["policy"]
        gc.collect()

        V_analytic_snapped = _V_analytic(X_0_snap)

        err_vs_snapped = V_dp - V_analytic_snapped
        err_vs_true = V_dp - V_analytic_true
        snap_gap = V_analytic_snapped - V_analytic_true
        rel_err_snapped = err_vs_snapped / V_analytic_snapped if V_analytic_snapped != 0 else float("nan")
        rel_err_true = err_vs_true / V_analytic_true if V_analytic_true != 0 else float("nan")

        return {"K_sub": K_sub, "status": "ok", "V_dp": V_dp,
                "V_analytic_snapped": V_analytic_snapped, "V_analytic_true": V_analytic_true,
                "X_0_true": X_0_true, "X_0_snap": X_0_snap, "X_0_diff": abs(X_0_snap - X_0_true),
                "err_vs_snapped": err_vs_snapped, "err_vs_true": err_vs_true, "snap_gap": snap_gap,
                "rel_err_snapped": rel_err_snapped, "rel_err_true": rel_err_true,
                "peak_rss_gb": peak_rss_gb, "policy_mismatch_fraction": mismatch_report["mismatch_fraction"],
                "mismatch_report": mismatch_report}

    rows = _run_ladder_with_memory_guard(decoupled_params, K_sub_list, _resolution)
    any_mismatch = any(r.get("policy_mismatch_fraction", 0.0) > 0 for r in rows if r["status"] == "ok")
    decoupled_result = {
        "C_max_used": decoupled_params["C_max"],
        "V_analytic_true": V_analytic_true,
        "resolutions": rows,
        "any_policy_mismatch": any_mismatch,
        "pass": _verdict_snapped(rows),
    }

    binding_result = _binding_strip_check(params, K_sub_list)
    decoupled_result["binding_case"] = binding_result
    decoupled_result["pass"] = decoupled_result["pass"] and binding_result["pass"]
    return decoupled_result

def _binding_strip_check(params, K_sub_list):
    N, T, q_max, K = params["N"], params["T"], params["q_max"], params["K"]
    alpha_func = params["alpha"]
    X_0_true = params["S_0"] - alpha_func(0)
    C_max = params["C_max"]
    N_slots = C_max // q_max

    binding_params = dict(params)
    binding_params["C_min"] = 0
    binding_params["C_max"] = C_max

    t = _exercise_dates(T, N)
    discount = np.exp(-params["r"] * t)

    def _bounds(X_0):
        mean_t, sd_t = _ou_law(t, X_0, params["kappa"], params["sigma"])
        mu_t = alpha_func(t) + mean_t
        call_t = _bachelier_call(mu_t, sd_t, K)
        disc_call_t = discount * call_t
        upper = float(q_max * np.sum(disc_call_t))
        top_vals = np.sort(disc_call_t)[::-1][:N_slots]
        lower = float(q_max * np.sum(top_vals))
        return lower, upper

    lower_true, upper_true = _bounds(X_0_true)

    def _resolution(p, K_sub):
        solved = _solve(p, K_sub)
        X_0_snap = float(solved["grid"][0, solved["col_X0"]])
        peak_rss_gb = _peak_rss_gb()
        V_dp = solved["option_price"]
        del solved["V"], solved["admissible"], solved["grid"], solved["policy"]
        gc.collect()

        lower_snap, upper_snap = _bounds(X_0_snap)
        lower_ok = V_dp >= lower_snap - BOUND_REL_EPS * max(abs(lower_snap), 1.0)
        upper_ok = V_dp <= upper_snap + BOUND_REL_EPS * max(abs(upper_snap), 1.0)

        return {"K_sub": K_sub, "status": "ok", "V_dp": V_dp,
                "lower_bound_snapped": lower_snap, "upper_bound_snapped": upper_snap,
                "lower_bound_true": lower_true, "upper_bound_true": upper_true,
                "X_0_true": X_0_true, "X_0_snap": X_0_snap,
                "slack_lower": V_dp - lower_snap, "slack_upper": upper_snap - V_dp,
                "lower_ok": bool(lower_ok), "upper_ok": bool(upper_ok),
                "peak_rss_gb": peak_rss_gb}

    rows = _run_ladder_with_memory_guard(binding_params, K_sub_list, _resolution)
    ok_rows = [r for r in rows if r["status"] == "ok"]
    verdict = bool(ok_rows) and all(r["lower_ok"] and r["upper_ok"] for r in ok_rows)
    return {"C_max": int(C_max), "N_slots": int(N_slots),
            "lower_bound_true": lower_true, "upper_bound_true": upper_true,
            "resolutions": rows, "pass": verdict}

def _single_right_bound_check(params, K_sub_list):
    N, T, K = params["N"], params["T"], params["K"]
    alpha_func = params["alpha"]
    X_0 = params["S_0"] - alpha_func(0)

    single_params = dict(params)
    single_params["q_max"] = 1
    single_params["C_max"] = 1
    single_params["C_min"] = 0

    t = _exercise_dates(T, N)
    mean_t, sd_t = _ou_law(t, X_0, params["kappa"], params["sigma"])
    mu_t = alpha_func(t) + mean_t
    call_t = _bachelier_call(mu_t, sd_t, K)
    discount = np.exp(-params["r"] * t)
    disc_calls = discount * call_t
    lower_bound = float(np.max(disc_calls))
    upper_bound = float(np.sum(disc_calls))

    def _resolution(p, K_sub):
        solved = _solve(p, K_sub)
        peak_rss_gb = _peak_rss_gb()
        V_dp = solved["option_price"]
        del solved["V"], solved["admissible"], solved["grid"], solved["policy"]
        gc.collect()

        lower_ok = V_dp >= lower_bound - BOUND_REL_EPS * max(abs(lower_bound), 1.0)
        upper_ok = V_dp <= upper_bound + BOUND_REL_EPS * max(abs(upper_bound), 1.0)
        return {"K_sub": K_sub, "status": "ok", "V_dp": V_dp,
                "lower_bound": lower_bound, "upper_bound": upper_bound,
                "slack_lower": V_dp - lower_bound, "slack_upper": upper_bound - V_dp,
                "lower_ok": bool(lower_ok), "upper_ok": bool(upper_ok), "peak_rss_gb": peak_rss_gb,
                "col_X0_snap": _col_X0_snap(p, K_sub)}

    rows = _run_ladder_with_memory_guard(single_params, K_sub_list, _resolution)
    ok_rows = [r for r in rows if r["status"] == "ok"]
    verdict = bool(ok_rows) and all(r["lower_ok"] and r["upper_ok"] for r in ok_rows)
    return {"lower_bound": lower_bound, "upper_bound": upper_bound, "resolutions": rows, "pass": verdict}

def _dSdK_oracle_check(params, K_sub_list):
    N, q_max, K0 = params["N"], params["q_max"], params["K"]
    decoupled_C_max = (N + 1) * q_max
    K_GRID = [2.40, K0, 4.40]

    rows = []
    for K_sub in K_sub_list:
        probe_params = dict(params)
        probe_params["C_max"] = decoupled_C_max
        predicted_gb = v_memory_gb(probe_params, K_sub)
        if predicted_gb > MEM_BUDGET_GB:
            print(f"    K_sub={K_sub}: SKIPPED (predicted V~={predicted_gb:.2f} GB > {MEM_BUDGET_GB:.1f} GB budget)")
            rows.append({"K_sub": K_sub, "status": "skipped_oom", "predicted_gb": predicted_gb})
            continue

        boundaries_ex = {}
        delta_X_used = None
        for K_val in K_GRID:
            p = dict(params)
            p["C_min"] = 0
            p["C_max"] = decoupled_C_max
            p["K"] = K_val
            solved = _solve(p, K_sub)

            boundary_full, _ = extract_boundary(
                solved["grid"], solved["policy"], solved["admissible"], p["alpha"], solved["delta_t"],
            )
            interp_data = {"grid": solved["grid"], "policy": solved["policy"], "V": solved["V"],
                            "m": solved["m"], "delta_X": solved["delta_X"], "delta_t": solved["delta_t"]}
            boundary_interp, _ = interpolate_boundary(
                interp_data, boundary_full, kappa=p["kappa"], sigma=p["sigma"], K=K_val, r=p["r"],
                alpha_func=p["alpha"],
            )
            boundary_interp[-1, :] = np.nan  # terminal date has no continuation

            ex_idx = np.arange(0, solved["grid"].shape[0], K_sub)  # thin fine steps to exercise days
            boundaries_ex[K_val] = boundary_interp[ex_idx, 0].copy()  # C=0 slice
            delta_X_used = solved["delta_X"]

            del solved["V"], solved["admissible"], solved["grid"], solved["policy"]
            gc.collect()

        b_itm, b_base, b_otm = boundaries_ex[2.40], boundaries_ex[K0], boundaries_ex[4.40]
        common_mask = np.isfinite(b_itm) & np.isfinite(b_base) & np.isfinite(b_otm)
        n_common = int(common_mask.sum())

        d_itm_K = 2.40 - K0
        d_otm_K = 4.40 - K0
        dSdK_itm = float(np.mean((b_itm[common_mask] - b_base[common_mask]) / d_itm_K)) if n_common else float("nan")
        dSdK_otm = float(np.mean((b_otm[common_mask] - b_base[common_mask]) / d_otm_K)) if n_common else float("nan")

        tol_itm = 2.0 * delta_X_used / abs(d_itm_K)
        tol_otm = 2.0 * delta_X_used / abs(d_otm_K)

        S_minus_K_base = b_base[np.isfinite(b_base)] - K0
        s_minus_k_stats = {
            "mean": float(np.mean(S_minus_K_base)) if S_minus_K_base.size else None,
            "std": float(np.std(S_minus_K_base)) if S_minus_K_base.size else None,
            "max_abs": float(np.max(np.abs(S_minus_K_base))) if S_minus_K_base.size else None,
            "n": int(S_minus_K_base.size),
        }

        itm_ok = bool(np.isfinite(dSdK_itm) and abs(dSdK_itm - 1.0) <= tol_itm)
        otm_ok = bool(np.isfinite(dSdK_otm) and abs(dSdK_otm - 1.0) <= tol_otm)

        rows.append({
            "K_sub": K_sub, "status": "ok", "n_common": n_common, "delta_X": delta_X_used,
            "dSdK_itm": dSdK_itm, "dSdK_otm": dSdK_otm, "tol_itm": tol_itm, "tol_otm": tol_otm,
            "itm_ok": itm_ok, "otm_ok": otm_ok, "S_minus_K_baseline": s_minus_k_stats,
            "predicted_gb": predicted_gb,
        })

    ok_rows = [r for r in rows if r["status"] == "ok"]
    verdict = bool(ok_rows) and all(r["itm_ok"] and r["otm_ok"] for r in ok_rows)
    return {"K_GRID": K_GRID, "decoupled_C_max": decoupled_C_max, "resolutions": rows, "pass": verdict}

def _convergence_ladder_check(data, output_dir):
    reference = None
    if "K_sub" in data and "option_price" in data:
        reference = (data["K_sub"], data["option_price"])

    result = _convergence_run_ladder(output_dir, _convergence_mod.MEM_BUDGET_GB, reference=reference)
    summary = result["summary"]
    ladder_by_ksub = {r["K_sub"]: r for r in result["ladder"]}
    production_resolution = _convergence_mod.PRODUCTION_K_SUB
    prod_row = ladder_by_ksub.get(production_resolution)

    price_conv = summary["price_convergence"]
    richardson = price_conv.get("richardson_estimate")
    price_error_at_production = None
    if prod_row is not None and prod_row.get("status") == "ok" and richardson is not None:
        price_error_at_production = prod_row["price"] - richardson

    boundary_conv = summary["mean_boundary_convergence"]
    boundary_richardson = boundary_conv.get("richardson_estimate")
    boundary_error_at_production = None
    if prod_row is not None and prod_row.get("status") == "ok" and boundary_richardson is not None:
        boundary_error_at_production = prod_row["mean_boundary_C0"] - boundary_richardson

    per_resolution_table = []
    for r in result["ladder"]:
        if r.get("status") != "ok":
            per_resolution_table.append({"K_sub": r["K_sub"], "status": r.get("status", "skipped_oom")})
            continue
        price_err = (r["price"] - richardson) if richardson is not None else None
        boundary_err = (r["mean_boundary_C0"] - boundary_richardson) if boundary_richardson is not None else None
        per_resolution_table.append({
            "K_sub": r["K_sub"], "status": "ok",
            "price": r["price"], "price_error_vs_richardson": price_err,
            "mean_boundary_C0": r["mean_boundary_C0"], "boundary_error_vs_richardson": boundary_err,
        })

    return {
        "per_resolution_table": per_resolution_table,
        "source_json": "convergence_ladder.json",
        "price_richardson_estimate": richardson,
        "price_order_p": price_conv.get("order_p"),
        "price_discretisation_error_K_sub_1": summary["price_discretisation_error"]["at_K_sub_1"]
            if summary["price_discretisation_error"] else None,
        "price_discretisation_error_top_feasible": summary["price_discretisation_error"]["at_top_feasible"]
            if summary["price_discretisation_error"] else None,
        "boundary_richardson_estimate": boundary_richardson,
        "boundary_order_p": boundary_conv.get("order_p"),
        "boundary_discretisation_error_K_sub_1": summary["mean_boundary_discretisation_error"]["at_K_sub_1"]
            if summary["mean_boundary_discretisation_error"] else None,
        "boundary_discretisation_error_top_feasible": summary["mean_boundary_discretisation_error"]["at_top_feasible"]
            if summary["mean_boundary_discretisation_error"] else None,
        "production_resolution": production_resolution,
        "price_at_production_resolution": prod_row["price"] if prod_row and prod_row.get("status") == "ok" else None,
        "price_error_at_production_resolution": price_error_at_production,
        "boundary_at_production_resolution": prod_row["mean_boundary_C0"] if prod_row and prod_row.get("status") == "ok" else None,
        "boundary_error_at_production_resolution": boundary_error_at_production,
    }

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
    if isinstance(obj, list):
        return [_to_py(x) for x in obj]
    return obj

def _params_snapshot(params):
    snap = {k: v for k, v in params.items() if k != "alpha"}
    snap["alpha(0)"] = params["alpha"](0)
    return snap

def analyse(data, output_dir):
    params = _require_params(data["params"])
    K_sub_list = list(LADDER)

    result = {
        "params_used": _to_py(_params_snapshot(params)),
        "K_sub_ladder": K_sub_list,
        "deterministic": True,
        "seeds": None,
    }

    fully_forced = _fully_forced_check(params, K_sub_list)
    result["check_A"] = fully_forced

    decoupled_strip = _decoupled_strip_check(params, K_sub_list)
    result["check_B"] = decoupled_strip

    single_right_bound = _single_right_bound_check(params, K_sub_list)
    result["check_C"] = single_right_bound

    dSdK_oracle = _dSdK_oracle_check(params, K_sub_list)
    result["check_E"] = dSdK_oracle

    convergence_ladder = _convergence_ladder_check(data, output_dir)
    result["check_D"] = convergence_ladder

    overall_pass = fully_forced["pass"] and decoupled_strip["pass"] and single_right_bound["pass"] and dSdK_oracle["pass"]
    result["overall_pass"] = overall_pass

    json_path = os.path.join(output_dir, "validation_value_function.json")
    with open(json_path, "w") as f:
        json.dump(_to_py(result), f, indent=2)

    return result
