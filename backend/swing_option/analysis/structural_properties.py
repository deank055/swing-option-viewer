# this file handles the structural properties of the boundary

import json
import os

import numpy as np

from swing_option.config import alpha, K, C_min, kappa, sigma, r
from swing_option.analysis.optimal_strategy import extract_boundary, interpolate_boundary, report_boundary_fallback

def _make_serializable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    return obj

def _frac(n, d):
    return n / d if d else float("nan")

TOL_STRICT = 1e-9

def _tier(slice_max, viol_list, tol):
    n = slice_max.size
    satisfying = int((slice_max <= tol).sum())
    n_viol = sum(1 for v in viol_list if v[-1] > tol)
    return dict(tol=float(tol), violations_above_tol=n_viol, slices_satisfying=satisfying, fraction=_frac(satisfying, n))

def _tiered_summary(slice_max, viol_list, tol_noise, tol_grid):
    eps = 1.0 + 1e-9
    viol_above_noise = [v for v in viol_list if v[-1] > tol_noise]
    return dict(
        strict=_tier(slice_max, viol_list, TOL_STRICT),
        noise_tol=_tier(slice_max, viol_list, tol_noise),
        grid_tol=_tier(slice_max, viol_list, tol_grid * eps),
        max_violation=float(slice_max.max()) if slice_max.size else 0.0,
        violations_above_tol_noise=viol_above_noise,
    )

def analyse(data, output_dir):
    K_sub = data.get("K_sub", 1)
    ex_idx = np.arange(0, data["grid"].shape[0], K_sub)

    grid = data["grid"][ex_idx]
    policy = data["policy"][ex_idx]
    adm = data["admissible"][ex_idx]
    delta_t = data["delta_t"] * K_sub
    N_day = len(ex_idx) - 1

    C1 = policy.shape[2]
    q_max_v = adm.shape[2] - 1
    C_arr = np.arange(C1)

    min_i_C = -(-C_arr // q_max_v)

    boundary_full, _ = extract_boundary(
        data["grid"], data["policy"], data["admissible"], alpha, data["delta_t"]
    )
    boundary_interp_full, boundary_diag = interpolate_boundary(
        data, boundary_full, kappa=kappa, sigma=sigma, K=K, r=r, alpha_func=alpha,
    )
    report_boundary_fallback(boundary_diag)
    boundary_full[-1, :] = np.nan
    boundary_interp_full[-1, :] = np.nan

    boundary = boundary_full[ex_idx]  # (N_day+1, C1)
    boundary_interp = boundary_interp_full[ex_idx]  # (N_day+1, C1)

    delta_X = data["delta_X"]
    tol_noise = 0.02 * delta_X
    tol_grid = 1.0 * delta_X

    results = {}

    total_checked = 0
    total_satisfying = 0
    violations_s = []
    viol_by_i = {}
    viol_by_C = {}

    for i in range(N_day):  # exclude terminal date
        rj = np.where(~np.isnan(grid[i]))[0]
        if rj.size < 2:
            continue

        pol = policy[i, rj, :]  # (n_reach, C1)
        ind = (pol > 0).astype(np.int8)
        diffs = np.diff(ind, axis=0)  # (n_reach-1, C1)
        monotone = (diffs >= 0).all(axis=0)  # (C1,)

        reachable_C = min_i_C <= i
        any_adm = adm[i, :, :].any(axis=1)
        voluntary = adm[i, :, 0]  # q=0 admissible => voluntary
        check = reachable_C & any_adm & voluntary

        total_checked += int(check.sum())
        total_satisfying += int((monotone & check).sum())

        for C in np.where(check & ~monotone)[0]:
            if len(violations_s) < 50:
                violations_s.append((int(i), round(float(i * delta_t), 4), int(C)))
            viol_by_i[int(i)] = viol_by_i.get(int(i), 0) + 1
            viol_by_C[int(C)] = viol_by_C.get(int(C), 0) + 1

    n_viol_s = total_checked - total_satisfying
    frac_s = _frac(total_satisfying, total_checked)

    results["monotonicity_S"] = dict(
        states_checked=total_checked,
        states_satisfying=total_satisfying,
        fraction=frac_s,
        violation_count=n_viol_s,
        first_50_violations=violations_s,
        violations_by_i={str(k): v for k, v in viol_by_i.items()},
        violations_by_C={str(k): v for k, v in viol_by_C.items()},
    )

    def _check_mono_C(bnd):
        dates_data = 0
        slice_max = []
        viols = []  # (i, t, C_lo, C_hi, drop)
        for i in range(N_day):
            b = bnd[i]
            fidx = np.where(np.isfinite(b))[0]
            if fidx.size < 2:
                continue
            dates_data += 1
            d = np.diff(b[fidx])
            drops = np.where(d < 0, -d, 0.0)
            slice_max.append(float(drops.max()) if drops.size else 0.0)
            bad = np.where(drops > TOL_STRICT)[0]
            for k in bad:
                drop = float(drops[k])
                viols.append((int(i), round(float(i * delta_t), 4), int(fidx[k]), int(fidx[k + 1]), drop))
        slice_max_arr = np.array(slice_max) if slice_max else np.zeros(0)
        return dates_data, slice_max_arr, viols

    dd_i, slicemax_i, viol_i = _check_mono_C(boundary_interp)
    dd_r, slicemax_r, viol_r = _check_mono_C(boundary)
    summ_i = _tiered_summary(slicemax_i, viol_i, tol_noise, tol_grid)
    summ_r = _tiered_summary(slicemax_r, viol_r, tol_noise, tol_grid)

    def _mono_C_json(dd, summ):
        return dict(
            dates_with_data=dd,
            max_decrease=summ["max_violation"],
            strict=summ["strict"],
            noise_tol=summ["noise_tol"],
            grid_tol=summ["grid_tol"],
            first_20_violations=summ["violations_above_tol_noise"][:20],
        )

    results["monotonicity_C"] = dict(
        interpolated=_mono_C_json(dd_i, summ_i),
        raw=_mono_C_json(dd_r, summ_r),
    )

    def _check_mono_T(bnd):
        slices_checked = 0
        n_noncontig = 0
        slice_max = []
        viols = []  # (C, i_a, i_b, t_i_a, increase)
        for C in range(C1):
            forced = ~adm[:N_day, C, 0]
            vol_idx = np.where(~forced)[0]
            if vol_idx.size == 0:
                continue
            last_vol = int(vol_idx[-1])
            window = bnd[:last_vol + 1, C]
            fidx = np.where(np.isfinite(window))[0]
            if fidx.size < 2:
                continue
            slices_checked += 1

            if np.any(np.diff(fidx) > 1):
                n_noncontig += 1

            vals = window[fidx]
            d = np.diff(vals)
            incs = np.where(d > 0, d, 0.0)
            slice_max.append(float(incs.max()) if incs.size else 0.0)

            bad = np.where(incs > TOL_STRICT)[0]
            for k in bad:
                i_a, i_b = int(fidx[k]), int(fidx[k + 1])
                viols.append((int(C), i_a, i_b, round(float(i_a * delta_t), 4), float(incs[k])))

        slice_max_arr = np.array(slice_max) if slice_max else np.zeros(0)
        return slices_checked, n_noncontig, slice_max_arr, viols

    sc_i, ncontig_i, slicemax_i_t, viol_i_t = _check_mono_T(boundary_interp)
    sc_r, ncontig_r, slicemax_r_t, viol_r_t = _check_mono_T(boundary)
    summ_i_t = _tiered_summary(slicemax_i_t, viol_i_t, tol_noise, tol_grid)
    summ_r_t = _tiered_summary(slicemax_r_t, viol_r_t, tol_noise, tol_grid)

    def _dist(arr):
        if arr.size == 0:
            return dict(min=None, Q1=None, median=None, Q3=None, max=None)
        return dict(min=float(arr.min()), Q1=float(np.percentile(arr, 25)),
                    median=float(np.median(arr)), Q3=float(np.percentile(arr, 75)),
                    max=float(arr.max()))

    dist_i_t = _dist(slicemax_i_t)
    dist_r_t = _dist(slicemax_r_t)

    REP_C_VALUES = [0, 525, 1050, 1120, 1190, 1250]

    def _representative_slices(bnd, C_values):
        out = {}
        for C in C_values:
            forced = ~adm[:N_day, C, 0]
            vol_idx = np.where(~forced)[0]
            if vol_idx.size == 0:
                out[str(C)] = dict(status="no voluntary window (always forced)")
                continue
            last_vol = int(vol_idx[-1])
            window = bnd[:last_vol + 1, C]
            fidx = np.where(np.isfinite(window))[0]
            if fidx.size == 0:
                out[str(C)] = dict(status="no finite boundary in voluntary window")
                continue
            i_first, i_last = int(fidx[0]), int(fidx[-1])
            S_first, S_last = float(window[i_first]), float(window[i_last])
            out[str(C)] = dict(
                i_first=i_first, t_first=round(float(i_first * delta_t), 4), S_first=S_first,
                i_last=i_last, t_last=round(float(i_last * delta_t), 4), S_last=S_last,
                decline=S_first - S_last,
            )
        return out

    rep_slices = _representative_slices(boundary_interp, REP_C_VALUES)

    def _mono_T_json(sc, ncontig, summ, dist):
        return dict(
            slices_checked=sc,
            max_increase=summ["max_violation"],
            strict=summ["strict"],
            noise_tol=summ["noise_tol"],
            grid_tol=summ["grid_tol"],
            violations_above_tol_noise=dict(
                count=len(summ["violations_above_tol_noise"]),
                first_20=summ["violations_above_tol_noise"][:20],
            ),
            n_noncontiguous_slices=ncontig,
            max_increase_distribution=dist,
        )

    results["monotonicity_t"] = dict(
        interpolated=_mono_T_json(sc_i, ncontig_i, summ_i_t, dist_i_t),
        raw=_mono_T_json(sc_r, ncontig_r, summ_r_t, dist_r_t),
        representative_slices_interpolated=rep_slices,
    )

    all_total = all_zero = all_qmax = all_int = 0
    vol_total = vol_zero = vol_qmax = vol_int = 0
    interior_C_list = []
    interior_t_list = []
    near_thresh = 0
    total_int_near = 0

    for i in range(N_day):
        rj = np.where(~np.isnan(grid[i]))[0]
        if rj.size == 0:
            continue

        pol = policy[i, rj, :]  # (n_reach, C1)

        reachable_C = min_i_C <= i
        any_adm_i = adm[i, :, :].any(axis=1)
        voluntary = adm[i, :, 0]
        valid_C = reachable_C & any_adm_i
        vol_C = valid_C & voluntary

        pv = pol[:, valid_C]
        pvo = pol[:, vol_C]

        all_total += pv.size
        all_zero += int((pv == 0).sum())
        all_qmax += int((pv == q_max_v).sum())
        all_int += int(((pv > 0) & (pv < q_max_v)).sum())

        vol_total += pvo.size
        vol_zero += int((pvo == 0).sum())
        vol_qmax += int((pvo == q_max_v).sum())
        int_here = (pvo > 0) & (pvo < q_max_v)  # (n_reach, n_vol)
        vol_int += int(int_here.sum())

        int_j, int_c = np.where(int_here)
        if int_j.size:
            vol_C_positions = np.where(vol_C)[0]
            interior_C_list.extend(vol_C_positions[int_c].tolist())
            interior_t_list.extend([float(i * delta_t)] * int_j.size)

        pol_any = pvo > 0
        j_star = pol_any.argmax(axis=0)
        j_idx = np.arange(rj.size)[:, None]
        near = np.abs(j_idx - j_star[None, :]) <= 2
        near_thresh += int((int_here & near).sum())
        total_int_near += int(int_here.sum())
    if interior_C_list:
        Ca = np.array(interior_C_list)
        Ta = np.array(interior_t_list)

    near_frac = _frac(near_thresh, total_int_near)

    results["bang_bang"] = dict(
        all_states=dict(total=all_total, q0=all_zero, qmax=all_qmax, interior=all_int,
                        frac_q0=_frac(all_zero, all_total),
                        frac_qmax=_frac(all_qmax, all_total),
                        frac_interior=_frac(all_int, all_total)),
        voluntary_states=dict(total=vol_total, q0=vol_zero, qmax=vol_qmax, interior=vol_int,
                              frac_q0=_frac(vol_zero, vol_total),
                              frac_qmax=_frac(vol_qmax, vol_total),
                              frac_interior=_frac(vol_int, vol_total)),
        interior_near_threshold_fraction=near_frac,
        interior_C_distribution=dict(
            min=int(Ca.min()) if interior_C_list else None,
            Q1=float(np.percentile(Ca, 25)) if interior_C_list else None,
            median=float(np.median(Ca)) if interior_C_list else None,
            Q3=float(np.percentile(Ca, 75)) if interior_C_list else None,
            max=int(Ca.max()) if interior_C_list else None,
        ) if interior_C_list else {},
        interior_t_distribution=dict(
            min=float(Ta.min()) if interior_t_list else None,
            Q1=float(np.percentile(Ta, 25)) if interior_t_list else None,
            median=float(np.median(Ta)) if interior_t_list else None,
            Q3=float(np.percentile(Ta, 75)) if interior_t_list else None,
            max=float(Ta.max()) if interior_t_list else None,
        ) if interior_t_list else {},
    )

    forced_total = 0
    earliest_i = None
    earliest_C_at_i = None
    forced_C0_onset = None
    forced_i_min = N_day + 1
    forced_i_max = -1
    forced_C_min_val = C1
    forced_C_max_val = -1
    analytic_mismatch = 0
    below_K_count = 0
    min_spot_exercise = float("inf")

    for i in range(N_day + 1):
        reachable_C = min_i_C <= i
        any_adm_i = adm[i, :, :].any(axis=1)
        forced_mask = reachable_C & any_adm_i & ~adm[i, :, 0]

        analytic = (reachable_C & any_adm_i & (C_arr < C_min) & ((C_min - C_arr) > (N_day - i) * q_max_v))
        analytic_mismatch += int((forced_mask != analytic).sum())

        forced_C = np.where(forced_mask)[0]
        if forced_C.size == 0:
            continue

        forced_total += forced_C.size
        forced_i_min = min(forced_i_min, i)
        forced_i_max = max(forced_i_max, i)
        forced_C_min_val = min(forced_C_min_val, int(forced_C.min()))
        forced_C_max_val = max(forced_C_max_val, int(forced_C.max()))

        if earliest_i is None:
            earliest_i = i
            earliest_C_at_i = int(forced_C[0])
        if 0 in forced_C and forced_C0_onset is None:
            forced_C0_onset = i

        rj = np.where(~np.isnan(grid[i]))[0]
        if rj.size == 0:
            continue
        alpha_i = float(alpha(i * delta_t))
        S_j = alpha_i + grid[i, rj]  # (n_reach,)
        pol_f = policy[i, rj, :][:, forced_C]  # (n_reach, n_forced)
        below = (S_j < K)[:, None] & (pol_f > 0)
        below_K_count += int(below.any(axis=0).sum())
        if below.any():
            min_spot_exercise = min(min_spot_exercise, float(S_j[below.any(axis=1)].min()))

    ms = min_spot_exercise if min_spot_exercise < float("inf") else None

    results["forced_region"] = dict(
        total_forced_states=forced_total,
        earliest_forced_i=earliest_i,
        earliest_forced_t=float(earliest_i * delta_t) if earliest_i is not None else None,
        earliest_forced_C=earliest_C_at_i,
        C0_forced_onset_i=forced_C0_onset,
        C0_forced_onset_t=float(forced_C0_onset * delta_t) if forced_C0_onset is not None else None,
        forced_i_range=[forced_i_min, forced_i_max] if forced_i_min <= forced_i_max else None,
        forced_C_range=[forced_C_min_val, forced_C_max_val] if forced_C_min_val <= forced_C_max_val else None,
        analytic_mismatches=analytic_mismatch,
        below_K_exercise_count=below_K_count,
        min_spot_at_exercise=ms,
    )

    results["boundary_fallback"] = boundary_diag

    # =======================================================================
    # Write JSON
    # =======================================================================
    json_path = os.path.join(output_dir, "structure_properties.json")
    with open(json_path, "w") as f:
        json.dump(_make_serializable(results), f, indent=2)
    print(f"\nResults written to: {json_path}")

    return results
