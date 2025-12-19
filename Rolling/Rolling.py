"""
==========================================================
Rolling-Window Desrosiers Observation Error Estimation
==========================================================

This script estimates a time-varying observation error
covariance matrix R_t using the Desrosiers innovation
diagnostic over a rolling window.

Key corrections:
----------------
- Uses daily data correctly (WINDOW_LENGTH = 252)
- Avoids truncating history via full-curve dropna
- Restricts to stable maturities available since 1990
- Associates each rolling estimate with a calendar date

Outputs:
--------
- rolling_R_all.csv (long format)
==========================================================
"""

import numpy as np
import pandas as pd


# ======================================================
# 1. Load yield curve data (CORRECTED)
# ======================================================
def load_yield_data(csv_path):
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    # Stable maturities available since early 1990s
    maturities = [
        "3 Mo", "6 Mo", "1 Yr",
        "2 Yr", "5 Yr", "10 Yr", "30 Yr"
    ]

    df = df[["Date"] + maturities].dropna()

    y = df[maturities].values
    dates = df["Date"].values

    return y, dates, maturities


# ======================================================
# 2. Kalman filter (random-walk state)
# ======================================================
def run_kalman_filter(y, Q_scale=1e-4, R_scale=1e-4):
    T, n = y.shape

    F = np.eye(n)
    H = np.eye(n)
    Q = Q_scale * np.eye(n)
    R = R_scale * np.eye(n)

    x = y[0].copy()
    P = np.eye(n)

    innovations = []
    analysis_residuals = []

    for t in range(1, T):
        # Forecast
        x_b = F @ x
        P_b = P + Q

        # Innovation
        d_b = y[t] - x_b
        S = P_b + R
        K = P_b @ np.linalg.inv(S)

        # Update
        x = x_b + K @ d_b
        P = (np.eye(n) - K) @ P_b

        d_a = y[t] - x

        innovations.append(d_b)
        analysis_residuals.append(d_a)

    return np.array(innovations), np.array(analysis_residuals)


# ======================================================
# 3. PSD enforcement
# ======================================================
def make_psd(matrix, eps=1e-6):
    eigvals, eigvecs = np.linalg.eigh(matrix)
    eigvals[eigvals < eps] = eps
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


# ======================================================
# 4. Rolling Desrosiers estimator (CORRECTED)
# ======================================================
def rolling_desrosiers_R(innovations, analysis_residuals, dates, window):
    T, n = innovations.shape
    R_list = []
    meta = []

    for t in range(window, T):
        R_t = np.zeros((n, n))

        for s in range(t - window, t):
            R_t += np.outer(analysis_residuals[s], innovations[s])

        R_t /= window
        R_t = 0.5 * (R_t + R_t.T)
        R_t = make_psd(R_t)

        R_list.append(R_t)

        end_date = pd.Timestamp(dates[t])
        meta.append({
            "window_index": t - window,
            "window_end_date": end_date,
            "window_end_year": end_date.year + end_date.dayofyear / 365.25
        })

    return R_list, meta


# ======================================================
# 5. Main execution
# ======================================================
if __name__ == "__main__":

    WINDOW_LENGTH = 252  # ~1 trading year (daily data)

    y, dates, maturities = load_yield_data(
        "data/yield-curve-rates-1990-2024.csv"
    )

    innovations, analysis_residuals = run_kalman_filter(y)

    R_rolling, meta = rolling_desrosiers_R(
        innovations,
        analysis_residuals,
        dates,
        window=WINDOW_LENGTH
    )

    print(f"Estimated {len(R_rolling)} rolling R matrices")
    print("First window end date:", meta[0]["window_end_date"])
    print("Last  window end date:", meta[-1]["window_end_date"])

    # ---- Save long-format CSV ----
    rows = []

    for t, R_t in enumerate(R_rolling):
        info = meta[t]
        for i, mi in enumerate(maturities):
            for j, mj in enumerate(maturities):
                rows.append({
                    "window_index": info["window_index"],
                    "window_end_date": info["window_end_date"],
                    "window_end_year": info["window_end_year"],
                    "maturity_i": mi,
                    "maturity_j": mj,
                    "covariance": R_t[i, j]
                })

    pd.DataFrame(rows).to_csv("output/rolling/rolling_R_all.csv", index=False)

    print("Saved rolling_R_all.csv")
    print("Done.")