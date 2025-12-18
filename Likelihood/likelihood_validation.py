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

# ---------------------------------------------------------
# Utility functions
# ---------------------------------------------------------

def log_likelihood_gaussian(innovation, S):
    """
    Compute log-likelihood of a Gaussian innovation.

    Parameters
    ----------
    innovation : ndarray (n,)
        Innovation vector d_t^b
    S : ndarray (n x n)
        Innovation covariance matrix

    Returns
    -------
    float
        Log-likelihood contribution
    """
    n = innovation.shape[0]
    sign, logdet = np.linalg.slogdet(S)

    if sign <= 0:
        raise ValueError("Innovation covariance not positive definite.")

    quad = innovation.T @ np.linalg.solve(S, innovation)

    return -0.5 * (logdet + quad + n * np.log(2 * np.pi))


# ---------------------------------------------------------
# Kalman filter with likelihood accumulation
# ---------------------------------------------------------

def run_kalman_filter_with_likelihood(
    observations,
    F,
    H,
    Q,
    R_model,
    x0,
    P0
):
    """
    Run Kalman filter and accumulate innovation log-likelihood.

    Parameters
    ----------
    observations : ndarray (T x n)
    F : ndarray (n x n)
    H : ndarray (n x n)
    Q : ndarray (n x n)
    R_model : callable
        Function returning R_t given time index t
    x0 : ndarray (n,)
        Initial state
    P0 : ndarray (n x n)
        Initial covariance

    Returns
    -------
    float
        Total log-likelihood
    """
    T, n = observations.shape

    x = x0.copy()
    P = P0.copy()

    loglik = 0.0

    for t in range(T):

        # ---- Forecast step ----
        x_b = F @ x
        P_b = F @ P @ F.T + Q

        # ---- Innovation ----
        y = observations[t]
        d_b = y - H @ x_b

        R_t = R_model(t)
        S = H @ P_b @ H.T + R_t

        # ---- Likelihood contribution ----
        loglik += log_likelihood_gaussian(d_b, S)

        # ---- Kalman update ----
        K = P_b @ H.T @ np.linalg.inv(S)
        x = x_b + K @ d_b
        P = (np.eye(n) - K @ H) @ P_b

    return loglik


# ---------------------------------------------------------
# R model builders
# ---------------------------------------------------------

def static_R_model(R):
    """
    Return a callable static R model.
    """
    def R_t(_):
        return R
    return R_t


# ---------------------------------------------------------
# Main execution
# ---------------------------------------------------------

def main():

    # ---- Load data ----
    df = pd.read_csv("data/yield-curve-rates-1990-2024.csv", index_col=0)
    df = df.dropna()
    
    observations = df.values

    T, n = observations.shape

    # ---- Model setup ----
    F = np.eye(n)              # Random walk
    H = np.eye(n)
    Q = 1e-6 * np.eye(n)       # Small process noise

    x0 = observations[0]
    P0 = np.eye(n)

    # ---- Load R models ----
    R_diag = np.diag(np.var(observations, axis=0))
    R_desrosiers = pd.read_csv(
        "output/static/observation_error_covariance.csv",
        index_col=0
    ).values

    R_models = {
        "Diagonal R": static_R_model(R_diag),
        "Static Desrosiers R": static_R_model(R_desrosiers)
    }

    # ---- Run comparison ----
    print("\nModel comparison (innovation log-likelihood):\n")

    results = {}

    for name, R_model in R_models.items():
        ll = run_kalman_filter_with_likelihood(
            observations,
            F,
            H,
            Q,
            R_model,
            x0,
            P0
        )
        results[name] = ll
        print(f"{name:<25s}: {ll: .2f}")

    # ---- Relative improvement ----
    if len(results) >= 2:
        keys = list(results.keys())
        improvement = results[keys[1]] - results[keys[0]]

        print("\nImprovement:")
        print(f"{keys[1]} vs {keys[0]}: {improvement: .2f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
