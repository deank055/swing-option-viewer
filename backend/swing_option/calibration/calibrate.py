# this file handles modle calibration

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from swing_option.config import S_0
import matplotlib.dates as mdates


from statsmodels.graphics.tsaplots import plot_acf

def load_historic(path, years=None):
    df = pd.read_csv(path)
    df.columns = ["date", "spot"]
    df["date"] = pd.to_datetime(df["date"], format="mixed")
    df = df.dropna(subset=["spot"]).sort_values("date").reset_index(drop=True)
    if years is not None:
        end = df["date"].iloc[-1]
        df = df[df["date"] >= end - pd.DateOffset(years=years)].reset_index(drop=True)
    return df

def _exclude_storm_months(df): # drop the two witner storms
    mask = ~(
        ((df["date"].dt.year == 2021) & (df["date"].dt.month == 2)) |
        ((df["date"].dt.year == 2026) & (df["date"].dt.month == 1))
    )
    return df[mask].reset_index(drop=True)

def plot_residual_acf(resid, lags=20):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    plot_acf(resid, lags=lags, ax=ax, alpha=0.05, zero=False)
    ax.set_title("")
    ax.set_xlabel("Lag (trading days)")
    ax.set_ylabel("Autocorrelation")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig("data/output/residual_acf.png", dpi=200, bbox_inches="tight")
    plt.show()

def plot_fitted_seasonal_curves_overlayed(futures_path, historic_data_path, coeffs_Q, coeffs_P, S_0, title):
    import matplotlib.dates as mdates

    df_futures = pd.read_csv(futures_path)
    df_futures = df_futures[df_futures["t_years"] > 0].copy()

    origin = pd.Timestamp("2026-06-30")

    t_smooth = np.linspace(0.0, 1.0, 500)
    dates_smooth = [origin + pd.Timedelta(days=float(t) * 365.25) for t in t_smooth]

    A_smooth = _design_matrix(t_smooth)
    P_smooth = A_smooth @ coeffs_P
    Q_smooth = A_smooth @ coeffs_Q
    risk_premium = float(coeffs_P[0] - coeffs_Q[0])

    t_futures = df_futures["t_years"].to_numpy()
    F_futures = df_futures["prior_settle"].to_numpy()
    dates_futures = [origin + pd.Timedelta(days=float(t) * 365.25) for t in t_futures]

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.fill_between(dates_smooth, P_smooth, Q_smooth, alpha=0.15, color='#888888', label='Risk premium')

    ax.plot(dates_smooth, P_smooth, '-', color='#1f77b4', linewidth=2.0, label=r'Seasonal curve under $\mathbb{P}$')
    ax.plot(dates_smooth, Q_smooth, '--', color='#d62728', linewidth=2.0, label=r'Seasonal curve under $\mathbb{Q}$')

    ax.scatter(dates_futures, F_futures, color='#d62728', s=35, zorder=5, label='Futures strip prices')
    ax.scatter([origin], [S_0], color='#1f77b4', marker='D', s=45, zorder=6, label=r'Spot price $S_0$')

    t_ann = 0.5
    date_ann = origin + pd.Timedelta(days=t_ann * 365.25)
    A_ann = _design_matrix(np.array([t_ann]))
    P_ann = float(A_ann @ coeffs_P)
    Q_ann = float(A_ann @ coeffs_Q)
    y_lo, y_hi = min(P_ann, Q_ann), max(P_ann, Q_ann)

    ax.annotate('', xy=(date_ann, y_lo), xytext=(date_ann, y_hi), arrowprops=dict(arrowstyle='<->', color='#444444', lw=1.2))
    ax.text(date_ann + pd.Timedelta(days=8), (P_ann + Q_ann) / 2, fr'$\max\Delta = {abs(risk_premium):.3f}$ \$/MMBtu', va='center', ha='left', fontsize=9.5, color='#444444')

    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel(r'Price (\$/MMBtu)', fontsize=11)
    ax.legend(fontsize=10, framealpha=0.9, loc='upper right')
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

    ax.set_ylim(2, 5)

    fig.tight_layout()
    fig.savefig(f"data/output/fitted_seasonal_curves_overlayed_{title.replace(' ', '_')}.png", dpi=200, bbox_inches="tight")
    plt.show()

def plot_fitted_seasonal_curve_historic(historic_data_path, coeffs, title):
    df = load_historic(historic_data_path, years=10)

    t = (df["date"] - df["date"].iloc[0]).dt.days.to_numpy() / 365.25  # calendar t, phase only
    S = df["spot"].to_numpy()

    A = _design_matrix(t)
    S_fitted = A @ coeffs

    plt.figure(figsize=(10, 6))
    plt.plot(t, S, 'o', label='Original Data', markersize=1)
    plt.plot(t, S_fitted, '-', label='Fitted Seasonal Curve ', linewidth=2)
    plt.xlabel('Time (years)')
    plt.ylabel('Price')
    plt.title(f'Fitted Curve Under {title}')
    plt.legend()
    plt.grid()
    plt.show()

def plot_fitted_seasonal_curve(futures_path, coeffs, title):
    df = pd.read_csv(futures_path)
    df_fut = df[df["t_years"] > 0]
    t = df_fut["t_years"].to_numpy()
    F = df_fut["prior_settle"].to_numpy()

    t_grid = np.linspace(0.0, float(t.max()), 1000)
    F_fitted = _design_matrix(t_grid) @ coeffs

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, F, 'o', color='#c0392b', markersize=6, label='Forward strip', zorder=3)
    ax.plot(0.0, S_0, 'D', color="#000000", markersize=7, label=r'Spot $S_0$', zorder=4)  # spot shown, NOT fitted
    ax.plot(t_grid, F_fitted, '-', color='#1f5fa8', linewidth=2.5, label='Fitted seasonal curve', zorder=2)

    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Price (USD/MMBtu)')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()

    fig.savefig(f"data/output/fitted_seasonal_curve_{title.replace(' ', '_')}.png", dpi=200, bbox_inches="tight")
    plt.show()

def _design_matrix(t):
    t = np.asarray(t, dtype=float)
    return np.column_stack([
        np.ones_like(t),
        t,
        np.cos(2 * np.pi * t), np.sin(2 * np.pi * t),
        np.cos(4 * np.pi * t), np.sin(4 * np.pi * t),
    ])

def fit_seasonal(t, y):
    A = _design_matrix(t)
    coeffs, *_ = np.linalg.lstsq(A, np.asarray(y, dtype=float), rcond=None)
    return coeffs

def estimate_dynamics(path):
    df = load_historic(path, years=10)

    t = (df["date"] - df["date"].iloc[0]).dt.days.to_numpy() / 365.25  # calendar t, phase only
    S = df["spot"].to_numpy()

    coeffs_P = fit_seasonal(t, S)

    X = S - _design_matrix(t) @ coeffs_P

    X_t = X[:-1]
    X_next = X[1:]
    beta_1 = np.sum(X_t * X_next) / np.sum(X_t ** 2)
    residuals = X_next - beta_1 * X_t

    from statsmodels.stats.diagnostic import acorr_ljungbox

    resid = X_next - beta_1 * X_t

    plot_residual_acf(resid, lags=20)

    lb = acorr_ljungbox(resid, lags=[10], return_df=True)
    p_lb = lb["lb_pvalue"].iloc[0]

    D = _design_matrix(t[1:])
    coeffs_r, *_ = np.linalg.lstsq(D, resid, rcond=None)
    r2_resid = 1 - np.sum((resid - D @ coeffs_r)**2) / np.sum((resid - resid.mean())**2)

    print(f"Ljung-Box(10) p = {p_lb:.3g}")
    print(f"residual-Fourier R2 = {r2_resid:.4f}")

    delta_t = 1 / 252
    kappa = -(np.log(beta_1) / delta_t)

    sigma = np.sqrt(2 * kappa * np.var(residuals, ddof=1) / (1 - np.exp(-2 * kappa * delta_t)))

    # n = len(X_t)
    # se_beta1 = np.sqrt(np.sum(residuals**2) / (n - 1) / np.sum(X_t**2))
    # se_kappa = se_beta1 / (beta_1 * delta_t)
    # se_sigma = sigma * np.sqrt(1 / (2 * n))

    # print(f"\nEstimated dynamics parameters:")
    # print(f"kappa = {kappa:.6f} (SE = {se_kappa:.6f})")
    # print(f"sigma = {sigma:.6f} (SE = {se_sigma:.6f})")

    # res_P = S - _design_matrix(t) @ coeffs_P
    # r2_P = 1 - np.sum(res_P**2) / np.sum((S - S.mean())**2)
    # print(f"alpha_P: R2 = {r2_P:.3f}")

    return kappa, sigma, coeffs_P

def estimate_seasonal_curve(futures_path):
    df = pd.read_csv(futures_path)

    df["delivery_month"] = pd.to_datetime(df["delivery_month"])

    # start_dm = df['delivery_month'].iloc[0]
    # end_dm = df['delivery_month'].iloc[-1]

    # total_days = (end_dm - start_dm).days if pd.notna(start_dm) and pd.notna(end_dm) else 'unknown'

    df = df[df["t_years"] > 0]  # delivery contracts only
    t = df["t_years"].to_numpy()
    F = df["prior_settle"].to_numpy()

    coeffs_Q, *_ = np.linalg.lstsq(_design_matrix(t), F, rcond=None)

    # res_Q = F - _design_matrix(t) @ coeffs_Q
    # r2_Q = 1 - np.sum(res_Q**2) / np.sum((F - F.mean())**2)
    # rmse_Q = np.sqrt(np.mean(res_Q**2))
    # print(f"alpha_Q: R2 = {r2_Q:.3f}, RMSE = {rmse_Q:.3f} USD/MMBtu")

    return coeffs_Q

def fit_window(df_win, max_gap_days=None):
    t = (df_win["date"] - df_win["date"].iloc[0]).dt.days.to_numpy() / 365.25
    S = df_win["spot"].to_numpy()
    coeffs = fit_seasonal(t, S)
    X = S - _design_matrix(t) @ coeffs
    Xt, Xn = X[:-1], X[1:]
    if max_gap_days is not None:
        dates = df_win["date"].to_numpy()
        gaps = (dates[1:] - dates[:-1]) / np.timedelta64(1, "D")
        keep = gaps <= max_gap_days
        Xt, Xn = Xt[keep], Xn[keep]
    beta = np.sum(Xt * Xn) / np.sum(Xt**2)
    resid = Xn - beta * Xt
    dt = 1 / 252
    kappa = -np.log(beta) / dt
    sigma = np.sqrt(2 * kappa * np.var(resid, ddof=1) / (1 - np.exp(-2 * kappa * dt)))
    n = len(Xt)
    se_beta = np.sqrt(np.sum(resid**2) / (n - 1) / np.sum(Xt**2))
    se_kappa = se_beta / (beta * dt)
    se_sigma = sigma * np.sqrt(1 / (2 * n))
    return kappa, sigma, se_kappa, se_sigma

def window_stability(df, t0, roll_years=5, min_obs=250, max_gap_days=None, label=""):
    df = df.sort_values("date").reset_index(drop=True)
    starts = pd.date_range(df["date"].iloc[0], t0 - pd.DateOffset(years=1), freq="MS")

    roll_k, roll_s, roll_sk, roll_ss, roll_mid = [], [], [], [], []
    exp_k, exp_s, exp_sk, exp_ss, exp_start = [], [], [], [], []

    for st in starts:
        w = df[(df["date"] >= st) & (df["date"] < st + pd.DateOffset(years=roll_years))]
        if len(w) >= min_obs:
            k, s, sk, ss = fit_window(w, max_gap_days=max_gap_days)
            roll_k.append(k); roll_s.append(s)
            roll_sk.append(sk); roll_ss.append(ss)
            roll_mid.append(st + pd.Timedelta(days=int(roll_years * 365.25 / 2)))
        # expanding start to t0
        w = df[(df["date"] >= st) & (df["date"] <= t0)]
        if len(w) >= min_obs:
            k, s, sk, ss = fit_window(w, max_gap_days=max_gap_days)
            exp_k.append(k); exp_s.append(s)
            exp_sk.append(sk); exp_ss.append(ss)
            exp_start.append(st)

    roll_k = np.array(roll_k); roll_s = np.array(roll_s)
    roll_sk = np.array(roll_sk); roll_ss = np.array(roll_ss)
    exp_k = np.array(exp_k); exp_s = np.array(exp_s)
    exp_sk = np.array(exp_sk); exp_ss = np.array(exp_ss)

    fig, ax = plt.subplots(2, 2, figsize=(8, 8), sharex="col")

    ax[0,0].plot(roll_mid, roll_k, color='#1f77b4', linewidth=1.5)
    ax[0,0].fill_between(roll_mid, roll_k - roll_sk, roll_k + roll_sk, alpha=0.25, color='#1f77b4')
    ax[0,0].set_ylabel(r"$\hat{\kappa}$", fontsize=12)
    ax[0,0].set_title(f"Rolling {roll_years}-yr (window centre)", fontsize=11)
    ax[0,0].grid(True, linestyle='--', alpha=0.4)

    ax[1,0].plot(roll_mid, roll_s, color='#d62728', linewidth=1.5)
    ax[1,0].fill_between(roll_mid, roll_s - roll_ss, roll_s + roll_ss, alpha=0.25, color='#d62728')
    ax[1,0].set_ylabel(r"$\hat{\sigma}$", fontsize=12)
    ax[1,0].set_xlabel("Date", fontsize=11)
    ax[1,0].grid(True, linestyle='--', alpha=0.4)

    ax[0,1].plot(exp_start, exp_k, color='#1f77b4', linewidth=1.5)
    ax[0,1].fill_between(exp_start, exp_k - exp_sk, exp_k + exp_sk, alpha=0.25, color='#1f77b4')
    ax[0,1].set_title(r"Expanding to $t_0$ (start date)", fontsize=11)
    ax[0,1].grid(True, linestyle='--', alpha=0.4)

    ax[1,1].plot(exp_start, exp_s, color='#d62728', linewidth=1.5)
    ax[1,1].fill_between(exp_start, exp_s - exp_ss, exp_s + exp_ss, alpha=0.25, color='#d62728')
    ax[1,1].set_xlabel("Date", fontsize=11)
    ax[1,1].grid(True, linestyle='--', alpha=0.4)

    for a in ax.flat:
        a.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.setp(a.xaxis.get_majorticklabels(), rotation=45, ha='right')

    fig.tight_layout()
    out_path = f"data/output/window_stability{label}.png"
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved figure to {out_path}")
    plt.show()

def run_window_stability_analyses(df, t0):
    window_stability(df, t0)

    df_excl = _exclude_storm_months(df)
    window_stability(df_excl, t0, max_gap_days=5, label="_excl")

def spike_excluded_robustness(path):
    df = load_historic(path, years=10)

    dt = 1 / 252

    t_f = (df["date"] - df["date"].iloc[0]).dt.days.to_numpy() / 365.25
    S_f = df["spot"].to_numpy()
    X_f = S_f - _design_matrix(t_f) @ fit_seasonal(t_f, S_f)
    Xt_f, Xn_f = X_f[:-1], X_f[1:]
    beta_f = np.sum(Xt_f * Xn_f) / np.sum(Xt_f**2)
    resid_f = Xn_f - beta_f * Xt_f
    kf = -np.log(beta_f) / dt
    sf = np.sqrt(2 * kf * np.var(resid_f, ddof=1) / (1 - np.exp(-2 * kf * dt)))

    df_ex = _exclude_storm_months(df)

    t_e = (df_ex["date"] - df_ex["date"].iloc[0]).dt.days.to_numpy() / 365.25
    S_e = df_ex["spot"].to_numpy()
    X_e = S_e - _design_matrix(t_e) @ fit_seasonal(t_e, S_e)

    dates_e = df_ex["date"].to_numpy()
    gaps = (dates_e[1:] - dates_e[:-1]) / np.timedelta64(1, "D")
    keep = gaps <= 5
    Xt_e, Xn_e = X_e[:-1][keep], X_e[1:][keep]

    beta_e = np.sum(Xt_e * Xn_e) / np.sum(Xt_e**2)
    resid_e = Xn_e - beta_e * Xt_e
    ke = -np.log(beta_e) / dt
    se = np.sqrt(2 * ke * np.var(resid_e, ddof=1) / (1 - np.exp(-2 * ke * dt)))

    # print(f"kappa: full={kf:.2f} excl={ke:.2f} change={ke-kf:+.2f}")
    # print(f"sigma: full={sf:.2f} excl={se:.2f} change={se-sf:+.2f}")

def calibrate(historic_data_path, future_data_path, output_path):
    kappa, sigma, coeffs_P = estimate_dynamics(historic_data_path)
    coeffs_Q = estimate_seasonal_curve(future_data_path)

    params = {
        "alpha_Q": coeffs_Q.tolist(),
        "alpha_P": coeffs_P.tolist(),
        "kappa": kappa,
        "sigma": sigma,
        "S_0": S_0,
    }

    with open(output_path, "w") as f:
        json.dump(params, f)

    # uncomment if you want to plot but most of them i didnt use. only last one (overlayed)
    # plot_fitted_seasonal_curve(future_data_path, coeffs_Q, "Risk-Neutral Measure (Q)")
    # plot_fitted_seasonal_curve(future_data_path, coeffs_P, "Physical Measure (P)")
    # plot_fitted_seasonal_curve_historic(historic_data_path, coeffs_Q, "Risk-Neutral Measure (Q)")
    # plot_fitted_seasonal_curve_historic(historic_data_path, coeffs_P, "Physical Measure (P)")
    # plot_fitted_seasonal_curves_overlayed(future_data_path, historic_data_path, coeffs_Q, coeffs_P, S_0, title="Overlayed Curves (P vs Q)")
