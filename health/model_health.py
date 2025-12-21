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
- Likelihood-based model comparison

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
os.makedirs("health/plots", exist_ok=True)


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def check(name, condition, level="FAIL"):
    """
    level: FAIL or WARN
    """
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

# ---- Likelihood ----
ll_df = pd.read_csv("output/likelihood/likelihood_summary.csv")

with open("output/likelihood/likelihood_metadata.json") as f:
    ll_meta = json.load(f)


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

dates = []
traces = []

for date, g in R_roll_df.groupby("window_end_date"):
    mat = g.pivot(
        index="maturity_i",
        columns="maturity_j",
        values="covariance"
    ).values
    dates.append(date)
    traces.append(np.trace(mat))

dates = pd.to_datetime(dates)
traces = np.array(traces)

statuses.append(
    check(
        "Rolling R trace is stable (relative variability)",
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

plot_path = "health/plots/rolling_R_trace.png"
plt.savefig(plot_path)
plt.close()

print(f"Saved rolling R trace plot → {plot_path}")


# ======================================================
# 5. Likelihood sanity
# ======================================================
section("Likelihood")

ll = ll_df.set_index("model")["loglik"]

statuses.append(
    check(
        "Likelihood ordering: diagonal < static < rolling",
        ll["diagonal"] < ll["static_full"] < ll["rolling_full"]
    )
)

statuses.append(
    check(
        "Rolling likelihood improvement is material",
        (ll["rolling_full"] - ll["static_full"])
        > 0.005 * abs(ll["static_full"]),
        level="WARN"
    )
)


# ======================================================
# 6. Cross-consistency checks
# ======================================================
section("Cross-consistency")

statuses.append(
    check(
        "Q scale consistent across scripts",
        R_static_meta["Q_scale"] in q_meta["Q_scales_tested"]
    )
)

statuses.append(
    check(
        "Number of maturities consistent",
        q_meta["n_maturities"] == ll_meta["n_maturities"]
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
    print("Model is internally consistent and behaving as intended.")
elif verdict == "WARN":
    print(
        "Model is statistically sound but exhibits regime-dependent "
        "behavior. Review diagnostics before production use."
    )
else:
    print(
        "Model shows structural inconsistencies. "
        "Do NOT use in production without investigation."
    )
