"""
==========================================================
Out-of-Sample Likelihood Tuning for Rolling Desroziers R
==========================================================

Method:
-------
1. Split data into TRAIN / TEST by date
2. Estimate rolling R_t on TRAIN only
3. Run Kalman filter on TEST using fixed R_t
4. Compute one-step-ahead Gaussian log-likelihood
5. Select window with highest OOS likelihood

This prevents overfitting and correctly selects
the smoothing window.

Output:
-------
- output/rolling/window_oos_likelihood_results.csv
==========================================================
"""

import numpy as np
import pandas as pd
import os

from Rolling import (
    load_yield_data,
    run_kalman_filter,
    estimate_rolling_R
)

# ======================================================
# Config
# ======================================================

WINDOW_GRID = [
    63,    # 3m
    252,   # 1.0y
    273,   # ~1.1y
    294,   # ~1.17y
    315,   # ~1.25y
    336,   # ~1.33y
    357,   # ~1.42y
    378,   # 1.5y
    504    #2y     
]

Q_SCALE = 0.2
R_FRACTION = 0.1

TRAIN_END_DATE = "2010-12-31"

os.makedirs("output/rolling", exist_ok=True)


# ======================================================
# Kalman filter OOS likelihood (FIXED R)
# ======================================================
def kalman_oos_loglikelihood(y_test, R_fixed, Q):
    """
    Computes one-step-ahead log-likelihood on test data
    using a fixed observation error covariance.
    """

    T, n = y_test.shape

    x = y_test[0].copy()
    P = np.eye(n)

    loglik = 0.0

    for t in range(1, T):
        # Forecast
        x_b = x
        P_b = P + Q

        S = P_b + R_fixed
        e = y_test[t] - x_b

        sign, logdet = np.linalg.slogdet(S)
        if sign <= 0:
            return -np.inf

        loglik += -0.5 * (
            logdet + e.T @ np.linalg.inv(S) @ e
        )

        # Analysis update
        K = P_b @ np.linalg.inv(S)
        x = x_b + K @ e
        P = (np.eye(n) - K) @ P_b

    return loglik


# ======================================================
# Main
# ======================================================
if __name__ == "__main__":

    # --------------------------------------------------
    # Load full data
    # --------------------------------------------------
    y, dates, maturities = load_yield_data(
        "data/yield-curve-rates-1990-2024.csv"
    )

    dates = pd.to_datetime(dates)

    # --------------------------------------------------
    # Train / test split
    # --------------------------------------------------
    train_mask = dates <= TRAIN_END_DATE
    test_mask = dates > TRAIN_END_DATE

    y_train = y[train_mask]
    y_test = y[test_mask]
    dates_train = dates[train_mask]

    print(f"Training samples: {len(y_train)}")
    print(f"Testing samples : {len(y_test)}")

    # --------------------------------------------------
    # Run Kalman filter on TRAIN only
    # --------------------------------------------------
    innovations, analysis_residuals, Q, R0 = run_kalman_filter(
        y_train,
        Q_scale=Q_SCALE,
        R_fraction=R_FRACTION
    )

    results = []

    # --------------------------------------------------
    # Window loop
    # --------------------------------------------------
    for window in WINDOW_GRID:
        print(f"Evaluating window length = {window}")

        # Must have enough data to estimate rolling R
        if len(innovations) <= window:
            print("  Skipped (window too large for training set)")
            continue

        # Estimate rolling R_t on TRAIN
        R_rolling, _ = estimate_rolling_R(
            innovations,
            analysis_residuals,
            dates_train,
            window_length=window
        )

        # Use LAST available R as fixed test covariance
        R_fixed = R_rolling[-1]

        # OOS likelihood
        loglik = kalman_oos_loglikelihood(
            y_test=y_test,
            R_fixed=R_fixed,
            Q=Q
        )

        results.append({
            "window_length": window,
            "oos_log_likelihood": loglik
        })

    # --------------------------------------------------
    # Results
    # --------------------------------------------------
    df = (
        pd.DataFrame(results)
        .sort_values("oos_log_likelihood", ascending=False)
        .reset_index(drop=True)
    )

    df.to_csv(
        "output/rolling/window_oos_likelihood_results.csv",
        index=False
    )

    print("\n===== OUT-OF-SAMPLE WINDOW RESULTS =====")
    print(df)
    print("\nBest window length:", df.loc[0, "window_length"])
