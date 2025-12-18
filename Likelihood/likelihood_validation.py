"""
==========================================================
LIKELIHOOD-BASED VALIDATION OF OBSERVATION ERROR MODELS
==========================================================

This script compares different observation error covariance
models (R) using innovation log-likelihood from a Kalman
filter.

Purpose:
---------
To objectively validate whether a structured observation
error covariance (e.g. Desrosiers-estimated R) improves
model performance relative to a diagonal baseline.

Design:
--------
- R is treated as a callable: R_t = R_model(t)
- Supports static, rolling-window, and future ANN-based R
- Uses innovation likelihood (Gaussian assumption)

This script is intentionally separate from covariance
estimation code to enforce clean model validation.
"""

import numpy as np
import pandas as pd

# ======================================================
# Simple Gaussian log-likelihood
# ======================================================
def loglik_gaussian(d, S):
    L = np.linalg.cholesky(S)          # ensures PD
    v = np.linalg.solve(L, d)
    logdet = 2 * np.sum(np.log(np.diag(L)))
    n = len(d)
    return -0.5 * (v @ v + logdet + n * np.log(2 * np.pi))


# ======================================================
# Kalman filter with likelihood only
# ======================================================
def kalman_loglik(y, F, H, Q, R_model, x0, P0):
    T, n = y.shape
    x = x0.copy()
    P = P0.copy()
    loglik = 0.0

    for t in range(T):
        # Forecast
        x_b = F @ x
        P_b = F @ P @ F.T + Q

        # Innovation
        d = y[t] - H @ x_b
        R = R_model(t)
        S = H @ P_b @ H.T + R

        # Likelihood
        loglik += loglik_gaussian(d, S)

        # Update
        K = P_b @ H.T @ np.linalg.solve(S, np.eye(n))
        x = x_b + K @ d
        P = (np.eye(n) - K @ H) @ P_b

    return loglik


# ======================================================
# Nested R model (IMPORTANT)
# ======================================================
def nested_R_model(R_full, alpha):
    R_diag = np.diag(np.diag(R_full))
    return lambda _: alpha * R_full + (1 - alpha) * R_diag


# ======================================================
# Main
# ======================================================
def main():

    # ---- Load data ----
    df = pd.read_csv("data/yield-curve-rates-1990-2024.csv", index_col=0)
    df = df.dropna()
    y = df.values
    T, n = y.shape

    # ---- Model setup ----
    F = np.eye(n)
    H = np.eye(n)
    Q = 1e-6 * np.eye(n)

    x0 = y[0]
    P0 = np.eye(n)

    # ---- Load Desrosiers R ----
    R_des = pd.read_csv(
        "output/static/observation_error_covariance.csv",
        index_col=0
    ).values

    # ---- Simple safety check ----
    assert np.all(np.linalg.eigvalsh(R_des) > 0), "R_des not PD!"

    # ---- Likelihood comparison ----
    print("\nLikelihood comparison:\n")

    alphas = np.linspace(0, 1, 11)
    results = {}

    for a in alphas:
        ll = kalman_loglik(
            y, F, H, Q,
            nested_R_model(R_des, a),
            x0, P0
        )
        results[a] = ll
        print(f"alpha = {a:.1f}   loglik = {ll:.2f}")

    best_alpha = max(results, key=results.get)

    print("\nBest alpha:")
    print(f"alpha = {best_alpha:.1f}")

    if best_alpha > 0:
        print("✔ Correlated observation errors improve the model.")
    else:
        print("✔ Diagonal observation errors are sufficient.")

    print("\nDone.")


if __name__ == "__main__":
    main()

