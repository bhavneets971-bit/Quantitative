"""
==========================================================
OUT-OF-SAMPLE LIKELIHOOD COMPARISON OF OBSERVATION ERROR MODELS
==========================================================

Diagonal, static full, and rolling full observation
error covariance models compared via OUT-OF-SAMPLE
Kalman filter log-likelihood.

Training period:
----------------
Used to estimate Q, static R, and rolling R

Test period:
------------
Used ONLY for likelihood evaluation

This avoids overfitting and allows fair comparison
between models of different flexibility.
==========================================================
"""

import numpy as np
import pandas as pd
import os
import json

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
# Kalman filter likelihood (fixed R)
# ======================================================
def kalman_loglik(y, F, H, Q, R, x0, P0):
    T, n = y.shape
    x = x0.copy()
    P = P0.copy()
    ll = 0.0

    for t in range(1, T):
        x_b = F @ x
        P_b = F @ P @ F.T + Q

        d = y[t] - H @ x_b
        S = H @ P_b @ H.T + R

        ll += loglik_gaussian(d, S)

        K = P_b @ H.T @ np.linalg.solve(S, np.eye(n))
        x = x_b + K @ d
        P = (np.eye(n) - K @ H) @ P_b

    return float(ll)


# ======================================================
# Kalman filter likelihood (rolling R)
# ======================================================
def kalman_loglik_rolling(y, dates, F, H, Q, R_by_date, x0, P0):
    T, n = y.shape
    x = x0.copy()
    P = P0.copy()
    ll = 0.0

    available_dates = sorted(R_by_date.keys())
    last_R = R_by_date[available_dates[-1]]

    for t in range(1, T):
        x_b = F @ x
        P_b = F @ P @ F.T + Q

        date = pd.Timestamp(dates[t])
        R = R_by_date.get(date, last_R)

        d = y[t] - H @ x_b
        S = H @ P_b @ H.T + R

        ll += loglik_gaussian(d, S)

        K = P_b @ H.T @ np.linalg.solve(S, np.eye(n))
        x = x_b + K @ d
        P = (np.eye(n) - K @ H) @ P_b

    return float(ll)


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

    TRAIN_END_DATE = "2010-12-31"

    # ---- Load yield data ----
    df = pd.read_csv("data/yield-curve-rates-1990-2024.csv")
    df["Date"] = pd.to_datetime(df["Date"], format="mixed")
    df = df.sort_values("Date")

    maturities = [
        "3 Mo", "6 Mo", "1 Yr",
        "2 Yr", "5 Yr", "10 Yr", "30 Yr"
    ]

    df = df[["Date"] + maturities].dropna(subset=maturities)

    # ---- Train / test split ----
    train_df = df[df["Date"] <= TRAIN_END_DATE].copy()
    test_df = df[df["Date"] > TRAIN_END_DATE].copy()

    y_train = train_df[maturities].values
    y_test = test_df[maturities].values
    dates_test = test_df["Date"].values

    n = y_train.shape[1]

    # ---- Estimate Q on TRAIN ----
    dy = np.diff(y_train, axis=0)
    Q_base = np.cov(dy.T)
    Q_SCALE = 0.2
    Q = Q_SCALE * Q_base

    # ---- Load static R (estimated on TRAIN elsewhere) ----
    R_static = pd.read_csv(
        "output/static/observation_error_covariance.csv",
        index_col=0
    ).loc[maturities, maturities].values

    # ---- Load rolling R (estimated on TRAIN) ----
    R_rolling_all = load_rolling_R(
        "output/rolling/rolling_R_all.csv",
        maturities
    )

    # Keep only rolling R estimated BEFORE test period
    R_rolling = {
        d: R for d, R in R_rolling_all.items()
        if d <= pd.to_datetime(TRAIN_END_DATE)
    }

    # Use last TRAIN R as fixed regime for test
    last_R_rolling = R_rolling[max(R_rolling.keys())]

    # ---- Model matrices ----
    F = np.eye(n)
    H = np.eye(n)
    x0 = y_test[0]
    P0 = np.eye(n)

    # ==================================================
    # OOS likelihood comparison
    # ==================================================
    ll_diag = kalman_loglik(
        y_test, F, H, Q,
        np.diag(np.diag(R_static)),
        x0, P0
    )

    ll_static = kalman_loglik(
        y_test, F, H, Q,
        R_static,
        x0, P0
    )

    ll_rolling = kalman_loglik(
        y_test, F, H, Q,
        last_R_rolling,
        x0, P0
    )

    print("\nOUT-OF-SAMPLE LIKELIHOOD COMPARISON:\n")
    print(f"Diagonal R      loglik = {ll_diag:.2f}")
    print(f"Static full R   loglik = {ll_static:.2f}")
    print(f"Rolling full R  loglik = {ll_rolling:.2f}")

    # ==================================================
    # Save results
    # ==================================================
    summary_df = pd.DataFrame([
        {"model": "diagonal", "loglik": ll_diag},
        {"model": "static_full", "loglik": ll_static},
        {"model": "rolling_full", "loglik": ll_rolling},
    ])

    summary_df.to_csv(
        "output/likelihood/likelihood_oos_summary.csv",
        index=False
    )

    metadata = {
        "evaluation": "out_of_sample",
        "train_end_date": TRAIN_END_DATE,
        "Q_scale": Q_SCALE,
        "n_test_obs": int(len(y_test)),
        "models_compared": [
            "diagonal",
            "static_full",
            "rolling_full"
        ]
    }

    with open("output/likelihood/likelihood_oos_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nSaved likelihood_oos_summary.csv and likelihood_oos_metadata.json")
    print("Done.")


if __name__ == "__main__":
    main()
