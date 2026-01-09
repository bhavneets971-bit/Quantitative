"""
==========================================================
MODEL HEALTH MONITOR
==========================================================

This script validates the internal consistency and
statistical health of the Kalman filter pipeline by
consuming saved outputs from:

- Q diagnostics
- Static Desroziers R estimation
- Rolling Desroziers R estimation
- Rolling R eigenvalue diagnostics
- In-sample likelihood comparison (IS)
- Out-of-sample likelihood comparison (OOS)

It performs sanity checks and produces a final
PASS / WARN / FAIL verdict, plus diagnostic plots.

NO estimation is performed here.
==========================================================
"""

import numpy as np
import pandas as pd
import json
import os
import matplotlib.pyplot as plt


# ======================================================
# Setup
# ======================================================
PLOT_DIR = "health/plots"
os.makedirs(PLOT_DIR, exist_ok=True)


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def check(name, condition, level="FAIL"):
    status = "PASS" if condition else level
    print(f"[{status:<4}] {name}")
    return status


statuses = []


# ======================================================
# Load artifacts
# ======================================================
section("Loading artifacts")

# ---- Q diagnostics ----
q_diag = pd.read_csv("output/diagnostics/q_diagnostics.csv")
with open("output/diagnostics/q_metadata.json") as f:
    q_meta = json.load(f)

# ---- Static R ----
R_static = pd.read_csv(
    "output/static/observation_error_covariance.csv",
    index_col=0
).values

with open("output/static/R_metadata.json") as f:
    R_static_meta = json.load(f)

# ---- Rolling R ----
R_roll_df = pd.read_csv(
    "output/rolling/rolling_R_all.csv",
    parse_dates=["window_end_date"]
)

roll_meta = pd.read_csv(
    "output/rolling/rolling_R_metadata.csv"
).iloc[0]

# ---- Rolling R eigenvalues ----
eig_path = "output/eigen/rolling_R_eigenvalues.csv"
eig_df = None
if os.path.exists(eig_path):
    eig_df = pd.read_csv(eig_path, parse_dates=["window_end_date"])

# ---- Likelihood (IS + OOS) ----
ll_is_df = pd.read_csv("output/likelihood/likelihood_is_summary.csv")
with open("output/likelihood/likelihood_is_extras.json") as f:
    ll_is_extra = json.load(f)

ll_oos_df = pd.read_csv("output/likelihood/likelihood_oos_summary.csv")
with open("output/likelihood/likelihood_oos_extras.json") as f:
    ll_oos_extra = json.load(f)


# ======================================================
# 1. Q diagnostics sanity
# ======================================================
section("Q diagnostics")

n = q_meta["n_maturities"]
expected_norm = np.sqrt(n)

best_row = q_diag.iloc[
    (q_diag["z_mean"] - expected_norm).abs().idxmin()
]

statuses.append(
    check(
        "Whitened innovation norm close to sqrt(n)",
        abs(best_row["z_mean"] - expected_norm) / expected_norm < 0.15
    )
)

statuses.append(
    check(
        "State covariance trace is finite",
        np.isfinite(best_row["mean_trace_P"])
    )
)


# ======================================================
# 2. Static R sanity
# ======================================================
section("Static R")

eigvals = np.linalg.eigvalsh(R_static)

statuses.append(
    check(
        "Static R is positive definite",
        np.all(eigvals > 0)
    )
)

diag_R = np.sqrt(np.diag(R_static))

statuses.append(
    check(
        "Short-end noise larger than long-end noise",
        diag_R[0] > diag_R[-1],
        level="WARN"
    )
)

corr = R_static / np.outer(diag_R, diag_R)

statuses.append(
    check(
        "Nearby maturities more correlated than distant ones",
        corr[0, 1] > corr[0, -1],
        level="WARN"
    )
)


# ======================================================
# 3. Rolling R sanity
# ======================================================
section("Rolling R")

dates, traces = [], []

for date, g in R_roll_df.groupby("window_end_date"):
    mat = g.pivot(
        index="maturity_i",
        columns="maturity_j",
        values="covariance"
    ).values
    dates.append(date)
    traces.append(np.trace(mat))

dates = pd.to_datetime(dates)
traces = np.asarray(traces)

statuses.append(
    check(
        "Rolling R trace strictly positive",
        np.all(traces > 0)
    )
)

statuses.append(
    check(
        "Rolling R trace stability (relative variability)",
        np.std(traces) / np.mean(traces) < 0.75,
        level="WARN"
    )
)

statuses.append(
    check(
        "Sufficient number of rolling windows",
        roll_meta["n_windows"] > 100
    )
)


# ======================================================
# 3b. Rolling R eigenvalue / PSD diagnostics
# ======================================================
section("Rolling R eigenvalue diagnostics")

EIG_DIR = "Rolling/diagnostics"

eigvals_path = os.path.join(EIG_DIR, "R_eigenvalues.csv")
eigsum_path = os.path.join(EIG_DIR, "R_eigen_summary.csv")

if not (os.path.exists(eigvals_path) and os.path.exists(eigsum_path)):
    statuses.append(
        check(
            "Eigenvalue diagnostics available",
            False
        )
    )
else:
    eigvals_df = pd.read_csv(eigvals_path, parse_dates=["date"])
    eigsum_df = pd.read_csv(eigsum_path, parse_dates=["date"])

    # ---- PSD check ----
    lambda_cols = [c for c in eigvals_df.columns if c.startswith("lambda_")]
    min_eig = eigvals_df[lambda_cols].min(axis=1)

    statuses.append(
        check(
            "No negative eigenvalues in rolling R",
            (min_eig > 0).all()
        )
    )

    # ---- Conditioning / rank diagnostics ----
    statuses.append(
        check(
            "Rolling R effective rank reasonable (median)",
            eigsum_df["effective_rank"].median() > 0.5 * len(lambda_cols),
            level="WARN"
        )
    )

    statuses.append(
        check(
            "No extreme eigenvalue dominance",
            eigsum_df["lambda1_fraction"].median() < 0.9,
            level="WARN"
        )
    )

    # ---- Cross-check: trace consistency ----
    merged = pd.merge(
        eigsum_df[["date", "trace_R"]],
        pd.DataFrame({
            "date": dates,
            "trace_from_R": traces
        }),
        on="date",
        how="inner"
    )

    statuses.append(
        check(
            "Eigenvalue trace matches rolling R trace",
            np.allclose(
                merged["trace_R"],
                merged["trace_from_R"],
                rtol=1e-6
            ),
            level="WARN"
        )
    )



# ======================================================
# 4. Rolling R diagnostic plot
# ======================================================
section("Rolling R diagnostic plot")

plt.figure(figsize=(12, 4))
plt.plot(dates, traces, lw=1)
plt.title("Rolling Observation Error Variance (trace(Rₜ))")
plt.xlabel("Date")
plt.ylabel("trace(Rₜ)")
plt.grid(alpha=0.3)
plt.tight_layout()

plot_path = os.path.join(PLOT_DIR, "rolling_R_trace.png")
plt.savefig(plot_path)
plt.close()

print(f"Saved rolling R trace plot → {plot_path}")


# ======================================================
# 5. Likelihood validation — IN-SAMPLE
# ======================================================
section("Likelihood validation (IN-SAMPLE)")

statuses.append(
    check(
        "IS likelihood ordering: diagonal < static < rolling",
        ll_is_extra["ordering_ok"]
    )
)

statuses.append(
    check(
        "IS rolling likelihood improvement is material",
        ll_is_extra["rolling_improvement_fraction"] > 0.005,
        level="WARN"
    )
)


# ======================================================
# 6. Likelihood validation — OUT-OF-SAMPLE
# ======================================================
section("Likelihood validation (OUT-OF-SAMPLE)")

statuses.append(
    check(
        "OOS likelihood ordering: diagonal < static < rolling",
        ll_oos_extra["ordering_ok"]
    )
)

statuses.append(
    check(
        "OOS rolling improves generalization",
        ll_oos_extra["deltas"]["rolling_minus_static"] > 0
    )
)

statuses.append(
    check(
        "OOS rolling gain smaller than IS gain (overfitting guard)",
        ll_oos_extra["rolling_improvement_fraction"]
        <= ll_is_extra["rolling_improvement_fraction"],
        level="WARN"
    )
)


# ======================================================
# 7. Cross-consistency checks
# ======================================================
section("Cross-consistency")

statuses.append(
    check(
        "Q scale consistent across scripts",
        R_static_meta["Q_scale"] == ll_is_extra["Q_scale"]
    )
)

statuses.append(
    check(
        "Number of maturities consistent",
        q_meta["n_maturities"] == ll_is_extra["n_maturities"]
    )
)


# ======================================================
# Final verdict
# ======================================================
section("FINAL VERDICT")

if "FAIL" in statuses:
    verdict = "FAIL"
elif "WARN" in statuses:
    verdict = "WARN"
else:
    verdict = "PASS"

print(f"\nMODEL HEALTH VERDICT: {verdict}\n")

if verdict == "PASS":
    print("Model is internally consistent and generalizes out-of-sample.")
elif verdict == "WARN":
    print(
        "Model is statistically sound but exhibits warnings. "
        "Review diagnostics before production use."
    )
else:
    print(
        "Model shows structural inconsistencies. "
        "Do NOT use in production without investigation."
    )
