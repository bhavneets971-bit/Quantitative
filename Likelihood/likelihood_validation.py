"""
==========================================================
LIKELIHOOD-BASED VALIDATION OF OBSERVATION ERROR MODELS
==========================================================

Diagonal, static full, and rolling full observation
error covariance models compared via Kalman filter
innovation log-likelihood.

All models are evaluated on the identical sample.
"""

import numpy as np
import pandas as pd
import os
import json


# ======================================================
# Ensure output directories exist
# ======================================================
os.makedirs("output/likelihood", exist_ok=True)


# ======================================================
# Gaussian log-likelihood
# ======================================================
def loglik_gaussian(d, S):
    L = np.linalg.cholesky(S)
    v = np.linalg.solve(L, d)
    logdet = 2 * np.sum(np.log(np.diag(L)))
    n = len(d)
    return -0.5 * (v @ v + logdet + n * np.log(2 * np.pi))


# ======================================================
# Kalman filter likelihood
# ======================================================
def kalman_loglik(y, F, H, Q, R_model, x0, P0):
    T, n = y.shape
    x = x0.copy()
    P = P0.copy()
    ll = 0.0

    for t in range(1, T):
        x_b = F @ x
        P_b = F @ P @ F.T + Q

        d = y[t] - H @ x_b
        R = R_model(t)
        S = H @ P_b @ H.T + R

        ll += loglik_gaussian(d, S)

        K = P_b @ H.T @ np.linalg.solve(S, np.eye(n))
        x = x_b + K @ d
        P = (np.eye(n) - K @ H) @ P_b

    return float(ll)


# ======================================================
# R model factories
# ======================================================
def diagonal_R(R_full):
    R_diag = np.diag(np.diag(R_full))
    return lambda _: R_diag


def static_R(R_full):
    return lambda _: R_full


def rolling_R_model(dates, R_by_date):
    def R_t(t):
        return R_by_date[pd.Timestamp(dates[t])]
    return R_t


# ======================================================
# Load rolling R matrices
# ======================================================
def load_rolling_R(csv_path, maturities):
    df = pd.read_csv(csv_path, parse_dates=["window_end_date"])
    R_by_date = {}

    for date, g in df.groupby("window_end_date"):
        mat = (
            g.pivot(
                index="maturity_i",
                columns="maturity_j",
                values="covariance"
            )
            .loc[maturities, maturities]
            .values
        )
        R_by_date[pd.Timestamp(date)] = mat

    return R_by_date


# ======================================================
# Main
# ======================================================
def main():

    # ---- Load yield data ----
    df = pd.read_csv("data/yield-curve-rates-1990-2024.csv")
    df["Date"] = pd.to_datetime(df["Date"], format="mixed")
    df = df.sort_values("Date")

    maturities = [
        "3 Mo", "6 Mo", "1 Yr",
        "2 Yr", "5 Yr", "10 Yr", "30 Yr"
    ]

    df = df[["Date"] + maturities].dropna(subset=maturities)

    # ---- Compute empirical Q anchor ----
    y_full = df[maturities].values
    dy = np.diff(y_full, axis=0)
    Q_base = np.cov(dy.T)

    # ---- Use validated Q scale ----
    Q_SCALE = 0.2
    Q = Q_SCALE * Q_base

    # ---- Load rolling R ----
    R_rolling = load_rolling_R(
        "output/rolling/rolling_R_all.csv",
        maturities
    )

    # ---- Force identical evaluation dates ----
    df = df[df["Date"].isin(R_rolling.keys())].copy()
    df = df.sort_values("Date")

    y = df[maturities].values
    dates = df["Date"].values
    n = y.shape[1]

    # ---- Model matrices ----
    F = np.eye(n)
    H = np.eye(n)

    x0 = y[0]
    P0 = np.eye(n)

    # ---- Load static R ----
    R_static = pd.read_csv(
        "output/static/observation_error_covariance.csv",
        index_col=0
    ).loc[maturities, maturities].values

    assert np.all(np.linalg.eigvalsh(R_static) > 0)

    # ==================================================
    # Likelihood comparison
    # ==================================================
    ll_diag = kalman_loglik(
        y, F, H, Q,
        diagonal_R(R_static),
        x0, P0
    )

    ll_static = kalman_loglik(
        y, F, H, Q,
        static_R(R_static),
        x0, P0
    )

    ll_rolling = kalman_loglik(
        y, F, H, Q,
        rolling_R_model(dates, R_rolling),
        x0, P0
    )

    print("\nLikelihood comparison (common sample):\n")
    print(f"Diagonal R      loglik = {ll_diag:.2f}")
    print(f"Static full R   loglik = {ll_static:.2f}")
    print(f"Rolling full R  loglik = {ll_rolling:.2f}")

    # ==================================================
    # >>> HEALTH REPORT SUPPORT: save results
    # ==================================================
    summary_df = pd.DataFrame([
        {"model": "diagonal", "loglik": ll_diag},
        {"model": "static_full", "loglik": ll_static},
        {"model": "rolling_full", "loglik": ll_rolling},
    ])

    summary_df.to_csv(
        "output/likelihood/likelihood_summary.csv",
        index=False
    )

    metadata = {
        "Q_scale": Q_SCALE,
        "n_obs": int(len(y)),
        "n_maturities": int(n),
        "sample_start": str(dates[0]),
        "sample_end": str(dates[-1]),
        "models_compared": [
            "diagonal",
            "static_full",
            "rolling_full"
        ]
    }

    with open("output/likelihood/likelihood_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nSaved likelihood_summary.csv and likelihood_metadata.json")
    print("Done.")


if __name__ == "__main__":
    main()
