"""
==========================================================
Rolling-Window Desrosiers Observation Error Estimation
==========================================================

This script estimates a time-varying observation error
covariance matrix R_t using the Desrosiers innovation
diagnostic over a rolling window.

Purpose:
---------
Capture changes in observation noise structure over time,
instead of assuming a single static covariance.

Outputs:
--------
- A time series of R matrices (one per time step)
- Optional saving of rolling covariance snapshots

==========================================================
"""

import numpy as np
import pandas as pd


# ======================================================
# 1. Load yield curve data
# ======================================================
def load_yield_data(csv_path):
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    maturities = [c for c in df.columns if c != "Date"]
    y = df[maturities].dropna().values

    return y, maturities


# ======================================================
# 2. Run Kalman filter and store residuals
# ======================================================
def run_kalman_filter(y, Q_scale=1e-6, R_scale=1e-4):
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
# 4. Rolling Desrosiers estimator
# ======================================================
def rolling_desrosiers_R(innovations, analysis_residuals, window):
    T, n = innovations.shape
    R_list = []

    for t in range(window, T):
        R_t = np.zeros((n, n))

        for s in range(t - window, t):
            R_t += np.outer(
                analysis_residuals[s],
                innovations[s]
            )

        R_t /= window
        R_t = 0.5 * (R_t + R_t.T)
        R_t = make_psd(R_t)

        R_list.append(R_t)

    return R_list


# ======================================================
# 5. Main execution
# ======================================================
if __name__ == "__main__":

    # ---- Parameters ----
    WINDOW_LENGTH = 252   # ~1 year of daily data

    # ---- Load data ----
    y, maturities = load_yield_data(
        "data/yield-curve-rates-1990-2024.csv"
    )

    # ---- Run Kalman filter ----
    innovations, analysis_residuals = run_kalman_filter(y)

    # ---- Estimate rolling R ----
    R_rolling = rolling_desrosiers_R(
        innovations,
        analysis_residuals,
        window=WINDOW_LENGTH
    )

    print(f"Estimated {len(R_rolling)} rolling R matrices.")
    print("Example R (last window):")
    print(pd.DataFrame(R_rolling[-1], index=maturities, columns=maturities))

    print("\nDone.")

    rows = []

    for t, R_t in enumerate(R_rolling):
        for i, mi in enumerate(maturities):
            for j, mj in enumerate(maturities):
                rows.append({
                    "time_index": t,
                    "maturity_i": mi,
                    "maturity_j": mj,
                    "covariance": R_t[i, j]
                })

    rolling_R_df = pd.DataFrame(rows)
    rolling_R_df.to_csv("rolling_R_all.csv", index=False)
