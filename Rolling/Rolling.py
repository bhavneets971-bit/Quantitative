"""
==========================================================
Rolling-Window Desroziers Observation Error Estimation
==========================================================
"""

import numpy as np
import pandas as pd
import os


# ======================================================
# 1. Load yield curve data
# ======================================================
def load_yield_data(csv_path):
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%y")
    df = df.sort_values("Date")

    maturities = [
        "3 Mo", "6 Mo", "1 Yr",
        "2 Yr", "5 Yr", "10 Yr"
    ]

    df = df[["Date"] + maturities].dropna()

    y = df[maturities].values
    dates = df["Date"].values

    return y, dates, maturities


# ======================================================
# 2. Kalman filter
# ======================================================
def run_kalman_filter(
    y,
    Q_scale=0.25,
    R_fraction=0.1,
    burn_in=50
):
    T, n = y.shape

    F = np.eye(n)
    H = np.eye(n)

    dy = np.diff(y, axis=0)
    Q_base = np.cov(dy.T)
    Q = Q_scale * Q_base

    R = np.diag(R_fraction * np.diag(Q_base))

    x = y[0].copy()
    P = np.eye(n)

    innovations = []
    analysis_residuals = []

    for t in range(1, T):
        x_b = F @ x
        P_b = P + Q

        d_b = y[t] - x_b
        S = P_b + R
        K = P_b @ np.linalg.inv(S)

        x = x_b + K @ d_b
        P = (np.eye(n) - K) @ P_b

        d_a = y[t] - x

        if t > burn_in:
            innovations.append(d_b)
            analysis_residuals.append(d_a)

    return (
        np.asarray(innovations),
        np.asarray(analysis_residuals),
        Q,
        R
    )


# ======================================================
# 3. PSD enforcement
# ======================================================
def enforce_psd(matrix, eps=1e-6):
    eigvals, eigvecs = np.linalg.eigh(matrix)
    eigvals[eigvals < eps] = eps
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


# ======================================================
# 4. Rolling Desroziers estimator
# ======================================================
def estimate_rolling_R(
    innovations,
    analysis_residuals,
    dates,
    window_length,
    burn_in=50      
):
    T_res, n = innovations.shape

    R_rolling = []
    meta = []

    half_window = window_length // 2

    for t in range(window_length, T_res):

        R_t = np.zeros((n, n))

        for s in range(t - window_length, t):
            R_t += np.outer(
                analysis_residuals[s],
                innovations[s]
            )

        R_t /= window_length
        R_t = 0.5 * (R_t + R_t.T)
        R_t = enforce_psd(R_t)

        R_rolling.append(R_t)

        center_index = t - half_window
        center_date = pd.Timestamp(dates[center_index + burn_in + 1])

        meta.append({
            "window_index": t - window_length,
            "window_start_date": pd.Timestamp(
                dates[t - window_length + burn_in + 1]
            ),
            "window_center_date": center_date,
            "window_end_date": pd.Timestamp(
                dates[t + burn_in + 1]
            ),
            "window_center_year": (
                center_date.year
                + center_date.dayofyear / 365.25
            )
        })

    return R_rolling, pd.DataFrame(meta)


# ======================================================
# 5. Save utilities
# ======================================================
def save_rolling_R_long(
    R_rolling,
    meta,
    maturities,
    output_path
):
    rows = []

    for t, R_t in enumerate(R_rolling):
        info = meta.iloc[t]
        for i, mi in enumerate(maturities):
            for j, mj in enumerate(maturities):
                rows.append({
                    "window_index": info["window_index"],
                    "window_start_date": info["window_start_date"],
                    "window_center_date": info["window_center_date"],
                    "window_end_date": info["window_end_date"],
                    "window_center_year": info["window_center_year"],
                    "maturity_i": mi,
                    "maturity_j": mj,
                    "covariance": R_t[i, j]
                })

    pd.DataFrame(rows).to_csv(output_path, index=False)


def save_metadata(
    meta,
    window_length,
    Q_scale,
    R_fraction,
    maturities,
    output_path
):
    summary = pd.DataFrame([{
        "window_length": window_length,
        "Q_scale": Q_scale,
        "R_fraction_init": R_fraction,
        "n_maturities": len(maturities),
        "n_windows": len(meta),
        "start_date": meta.iloc[0]["window_center_date"],
        "end_date": meta.iloc[-1]["window_center_date"]
    }])

    summary.to_csv(output_path, index=False)


# ======================================================
# 6. CLI execution
# ======================================================
if __name__ == "__main__":

    os.makedirs("output/rolling", exist_ok=True)

    WINDOW_LENGTH = 378 # Based on Tuning Results
    Q_SCALE = 0.25
    R_FRACTION = 0.1
    BURN_IN = 50

    y, dates, maturities = load_yield_data(
        "data/yield-curve-rates-1990-2024.csv"
    )

    innovations, analysis_residuals, Q, R0 = run_kalman_filter(
        y,
        Q_scale=Q_SCALE,
        R_fraction=R_FRACTION,
        burn_in=BURN_IN
    )

    R_rolling, meta = estimate_rolling_R(
        innovations,
        analysis_residuals,
        dates,
        window_length=WINDOW_LENGTH,
        burn_in=BURN_IN
    )

    save_rolling_R_long(
        R_rolling,
        meta,
        maturities,
        "output/rolling/rolling_R_all.csv"
    )

    save_metadata(
        meta,
        WINDOW_LENGTH,
        Q_SCALE,
        R_FRACTION,
        maturities,
        "output/rolling/rolling_R_metadata.csv"
    )

    print(f"Saved {len(R_rolling)} rolling R matrices")
    print("Done.")
