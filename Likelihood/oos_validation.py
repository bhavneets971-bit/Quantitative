"""
==========================================================
OUT-OF-SAMPLE LIKELIHOOD-BASED VALIDATION OF OBSERVATION ERROR MODELS
==========================================================

Diagonal, static full, and rolling full observation
error covariance models compared via Kalman filter
innovation log-likelihood.

This script is AUTHORITATIVE for out-of-sample likelihood
comparison. No logic is removed — only metadata is added
to support downstream health monitoring.
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

    # ---- Configuration ----
    Q_SCALE = 0.2
    TRAIN_END_DATE = "2014-12-31"   # explicit, auditable split

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
    df_train = df[df["Date"] <= TRAIN_END_DATE].copy()
    df_test = df[df["Date"] > TRAIN_END_DATE].copy()

    # ---- Compute Q from TRAIN ONLY ----
    y_train = df_train[maturities].values
    dy = np.diff(y_train, axis=0)
    Q_base = np.cov(dy.T)
    Q = Q_SCALE * Q_base

    # ---- Load rolling R (estimated on training sample) ----
    R_rolling = load_rolling_R(
        "output/rolling/rolling_R_all.csv",
        maturities
    )

    roll_meta = pd.read_csv(
        "output/rolling/rolling_R_metadata.csv"
    ).iloc[0]

    # ---- Restrict test sample to dates with R available ----
    df_test = df_test[df_test["Date"].isin(R_rolling.keys())].copy()
    df_test = df_test.sort_values("Date")

    y = df_test[maturities].values
    dates = df_test["Date"].values
    n = y.shape[1]

    # ---- Model matrices ----
    F = np.eye(n)
    H = np.eye(n)

    x0 = y[0]
    P0 = np.eye(n)

    # ---- Load static R (estimated on training sample) ----
    R_static = pd.read_csv(
        "output/static/observation_error_covariance.csv",
        index_col=0
    ).loc[maturities, maturities].values

    assert np.all(np.linalg.eigvalsh(R_static) > 0)

    # ==================================================
    # Likelihood comparison (AUTHORITATIVE, OOS)
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

    print("\nOut-of-sample likelihood comparison:\n")
    print(f"Diagonal R      loglik = {ll_diag:.2f}")
    print(f"Static full R   loglik = {ll_static:.2f}")
    print(f"Rolling full R  loglik = {ll_rolling:.2f}")

    # ==================================================
    # Save primary OOS likelihood outputs
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

    # ==================================================
    # Extended metadata for health monitoring
    # ==================================================
    deltas = {
        "static_minus_diagonal": ll_static - ll_diag,
        "rolling_minus_static": ll_rolling - ll_static
    }

    extras = {
        "sample_type": "out_of_sample",
        "train_end_date": TRAIN_END_DATE,
        "Q_scale": Q_SCALE,
        "window_length": int(roll_meta["window_length"]),
        "n_test_obs": int(len(y)),
        "n_maturities": int(n),
        "test_start": str(dates[0]),
        "test_end": str(dates[-1]),
        "models_compared": [
            "diagonal",
            "static_full",
            "rolling_full"
        ],
        "likelihoods": {
            "diagonal": ll_diag,
            "static_full": ll_static,
            "rolling_full": ll_rolling
        },
        "deltas": deltas,
        "ordering_ok": ll_diag < ll_static < ll_rolling,
        "rolling_improvement_fraction": (
            (ll_rolling - ll_static) / abs(ll_static)
        )
    }

    with open("output/likelihood/likelihood_oos_extras.json", "w") as f:
        json.dump(extras, f, indent=2)

    metadata = {
        "sample_type": "out_of_sample",
        "train_end_date": TRAIN_END_DATE,
        "Q_scale": Q_SCALE,
        "n_test_obs": int(len(y)),
        "n_maturities": int(n),
        "models_compared": extras["models_compared"]
    }

    with open("output/likelihood/likelihood_oos_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nSaved out-of-sample likelihood outputs and metadata")
    print("Done.")


if __name__ == "__main__":
    main()
