"""
==========================================================
Q DIAGNOSTICS FOR KALMAN FILTER
==========================================================

Purpose
-------
Diagnose the process noise covariance Q by checking:
- Innovation whitening
- State covariance stability

This script does NOT perform likelihood comparison.
It is intended to calibrate Q scale only.

==========================================================
"""

import numpy as np
import pandas as pd


# ======================================================
# Kalman filter diagnostics
# ======================================================
def kalman_q_diagnostics(y, Q, R):
    T, n = y.shape
    F = np.eye(n)
    H = np.eye(n)

    x = y[0].copy()
    P = np.eye(n)

    whitened_norms = []
    P_traces = []

    for t in range(1, T):
        # Forecast
        x_b = x
        P_b = P + Q

        # Innovation
        d = y[t] - x_b
        S = P_b + R

        # Whitening
        L = np.linalg.cholesky(S)
        z = np.linalg.solve(L, d)
        whitened_norms.append(np.linalg.norm(z))

        # Update
        K = P_b @ np.linalg.inv(S)
        x = x_b + K @ d
        P = (np.eye(n) - K) @ P_b

        P_traces.append(np.trace(P))

    return (
        np.mean(whitened_norms),
        np.std(whitened_norms),
        np.mean(P_traces),
    )


# ======================================================
# Main
# ======================================================
def main():

    # ---- Load data ----
    df = pd.read_csv("data/yield-curve-rates-1990-2024.csv")
    df["Date"] = pd.to_datetime(df["Date"], format="mixed")
    df = df.sort_values("Date")

    maturities = ["3 Mo", "6 Mo", "1 Yr", "2 Yr", "5 Yr", "10 Yr", "30 Yr"]
    df = df[maturities].dropna(subset=maturities)
    y = df.values
    _, n = y.shape

    # ---- Empirical Q anchor ----
    dy = np.diff(y, axis=0)
    Q_base = np.cov(dy.T)

    # ==================================================
    # PRINCIPLED INITIAL R
    # ==================================================
    # R explains only obvious measurement noise
    # Set as a small fraction of one-step yield variance
    alpha_R = 0.1  # 10% is a good default
    R = np.diag(alpha_R * np.diag(Q_base))

    print(R)
    
    # ---- Q scale grid ----
    Q_scales = [0.1, 0.2, 0.25, 0.3]

    print("\nQ diagnostics:\n")
    print("Expected ||z|| ≈ sqrt(n) ≈", np.sqrt(n))
    print(f"Initial R scale = {alpha_R:.0%} of diag(Q_base)")
    print("")

    for scale in Q_scales:
        Q = scale * Q_base

        z_mean, z_std, P_trace = kalman_q_diagnostics(y, Q, R)

        print(f"Q scale = {scale:.2f}")
        print(f"  mean ||whitened z|| : {z_mean:.2f}")
        print(f"  std  ||whitened z|| : {z_std:.2f}")
        print(f"  mean trace(P)      : {P_trace:.4e}")
        print("")

    print("Done.")


if __name__ == "__main__":
    main()