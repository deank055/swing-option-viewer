# this file contains the robustness check for chapter 5

import numpy as np
import pandas as pd
from scipy.stats import norm

from swing_option.calibration.calibrate import (
    _design_matrix,
    estimate_dynamics,
    estimate_seasonal_curve,
    fit_seasonal,
)

DEFAULT_EXCLUDE_WINDOWS = [
    ("2021-02-01", "2021-03-01"),
    ("2022-01-01", "2023-01-01"), 
]

_GAP_THRESHOLD_DAYS = 7

def estimate_dynamics_excluding(path, exclude_windows=None):
    if exclude_windows is None:
        exclude_windows = DEFAULT_EXCLUDE_WINDOWS

    df = pd.read_csv(path)
    df.columns = ["date", "spot"]
    df["date"] = pd.to_datetime(df["date"], format="mixed")
    df = df.dropna(subset=["spot"])
    df = df.sort_values("date").reset_index(drop=True)

    end = df["date"].iloc[-1]
    df = df[df["date"] >= end - pd.DateOffset(years=10)].reset_index(drop=True)

    t_full = (df["date"] - df["date"].iloc[0]).dt.days.to_numpy() / 365.25

    exclude_mask = np.zeros(len(df), dtype=bool)
    for start, end_excl in exclude_windows:
        start = pd.Timestamp(start)
        end_excl = pd.Timestamp(end_excl)
        exclude_mask |= (df["date"] >= start) & (df["date"] < end_excl)

    retained = df.loc[~exclude_mask].reset_index(drop=True)
    t = t_full[~exclude_mask]
    S = retained["spot"].to_numpy()
    dates = retained["date"].to_numpy()

    coeffs_P = fit_seasonal(t, S)
    X = S - _design_matrix(t) @ coeffs_P

    gap_days = np.diff(dates) / np.timedelta64(1, "D")
    segment_break = gap_days > _GAP_THRESHOLD_DAYS

    X_t = X[:-1]
    X_next = X[1:]
    within_segment = ~segment_break
    pairs_dropped = int(np.sum(segment_break))

    X_t = X_t[within_segment]
    X_next = X_next[within_segment]

    beta_1 = np.sum(X_t * X_next) / np.sum(X_t ** 2)
    residuals = X_next - beta_1 * X_t

    delta_t = 1 / 252
    kappa = -(np.log(beta_1) / delta_t)
    sigma = np.sqrt(2 * kappa * np.var(residuals, ddof=1) / (1 - np.exp(-2 * kappa * delta_t)))
    stationary_std = sigma / np.sqrt(2 * kappa)

    return {
        "beta_1": beta_1,
        "kappa": kappa,
        "sigma": sigma,
        "stationary_std": stationary_std,
        "coeffs_P": coeffs_P,
        "pairs_dropped": pairs_dropped,
    }

def negative_price_probability(alpha_min, kappa, sigma):
    stationary_std = sigma / np.sqrt(2 * kappa)
    return norm.cdf(-alpha_min / stationary_std)

def _alpha_min(coeffs_Q, n_grid=1000):
    t_grid = np.linspace(0.0, 1.0, n_grid)
    alpha_grid = _design_matrix(t_grid) @ coeffs_Q
    return float(np.min(alpha_grid))

def compare(historic_data_path, future_data_path, exclude_windows=None):
    kappa_full, sigma_full, _ = estimate_dynamics(historic_data_path)
    beta_1_full = np.exp(-kappa_full / 252)
    stationary_std_full = sigma_full / np.sqrt(2 * kappa_full)

    excl = estimate_dynamics_excluding(historic_data_path, exclude_windows)

    coeffs_Q = estimate_seasonal_curve(future_data_path)
    alpha_min = _alpha_min(coeffs_Q)

    p_full = negative_price_probability(alpha_min, kappa_full, sigma_full)
    p_excl = negative_price_probability(alpha_min, excl["kappa"], excl["sigma"])

    return {
        "full": {"kappa": kappa_full, "sigma": sigma_full, "stationary_std": stationary_std_full, "beta_1": beta_1_full, "p_negative": p_full},
        "excluding_anomalies": {**excl, "p_negative": p_excl},
        "alpha_min": alpha_min,
    }