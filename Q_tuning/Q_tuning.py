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
import os
import json


# ======================================================
# Ensure output directories exist
# ======================================================
os.makedirs("output/diagnostics", exist_ok=True)


# ======================================================
# Kalman filter diagnostics
# ======================================================
def kalman_q_diagnostics(y, Q, R):
    T, n = y.shape

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
        float(np.mean(whitened_norms)),
        float(np.std(whitened_norms)),
        float(np.mean(P_traces)),
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
    T, n = y.shape

    # ---- Empirical Q anchor ----
    dy = np.diff(y, axis=0)
    Q_base = np.cov(dy.T)

    # ==================================================
    # Principled initial R
    # ==================================================
    ALPHA_R = 0.1
    R = np.diag(ALPHA_R * np.diag(Q_base))

    # ---- Q scale grid ----
    Q_scales = [0.1, 0.2, 0.25, 0.3]

    print("\nQ diagnostics:\n")
    print("Expected ||z|| ≈ sqrt(n) ≈", np.sqrt(n))
    print(f"Initial R scale = {ALPHA_R:.0%} of diag(Q_base)\n")

    rows = []

    for scale in Q_scales:
        Q = scale * Q_base

        z_mean, z_std, P_trace = kalman_q_diagnostics(y, Q, R)

        rows.append({
            "Q_scale": scale,
            "z_mean": z_mean,
            "z_std": z_std,
            "mean_trace_P": P_trace
        })

        print(f"Q scale = {scale:.2f}")
        print(f"  mean ||whitened z|| : {z_mean:.2f}")
        print(f"  std  ||whitened z|| : {z_std:.2f}")
        print(f"  mean trace(P)      : {P_trace:.4e}\n")

    # ==================================================
    # >>> HEALTH REPORT SUPPORT: save diagnostics
    # ==================================================
    diagnostics_df = pd.DataFrame(rows)
    diagnostics_df.to_csv(
        "output/diagnostics/q_diagnostics.csv",
        index=False
    )

    metadata = {
        "alpha_R": ALPHA_R,
        "n_obs": int(T),
        "n_maturities": int(n),
        "Q_scales_tested": Q_scales,
        "expected_whitened_norm": float(np.sqrt(n))
    }

    with open("output/diagnostics/q_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("Saved q_diagnostics.csv and q_metadata.json")
    print("Done.")


if __name__ == "__main__":
    main()
