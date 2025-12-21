"""
==========================================================
MODEL HEALTH REPORT
==========================================================

Consumes outputs from:
- Q diagnostics
- Static Desroziers R
- Rolling Desroziers R
- Likelihood validation

Produces a unified PASS / WARN / FAIL verdict.
"""

import numpy as np
import pandas as pd
import json
import os


# ======================================================
# Utilities
# ======================================================
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
# Track overall status
# ======================================================
statuses = []


# ======================================================
# 1. Q diagnostics sanity
# ======================================================
section("Q diagnostics")

n = q_meta["n_maturities"]
expected_norm = np.sqrt(n)

best = q_diag.iloc[
    (q_diag["z_mean"] - expected_norm).abs().idxmin()
]

statuses.append(
    check(
        "Whitened innovation norm close to sqrt(n)",
        abs(best["z_mean"] - expected_norm) / expected_norm < 0.15
    )
)

statuses.append(
    check(
        "State covariance trace is finite",
        np.isfinite(best["mean_trace_P"])
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

traces = []

for date, g in R_roll_df.groupby("window_end_date"):
    mat = g.pivot(
        index="maturity_i",
        columns="maturity_j",
        values="covariance"
    ).values
    traces.append(np.trace(mat))

traces = np.array(traces)

statuses.append(
    check(
        "Rolling R trace is stable",
        np.std(traces) / np.mean(traces) < 0.75,
        level="WARN"
    )
)

statuses.append(
    check(
        "Sufficient rolling windows",
        roll_meta["n_windows"] > 100
    )
)


# ======================================================
# 4. Likelihood sanity
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
# 5. Cross-consistency
# ======================================================
section("Cross-consistency")

statuses.append(
    check(
        "Q scale consistent across scripts",
        q_meta["Q_scales_tested"].count(R_static_meta["Q_scale"]) > 0
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

if verdict != "PASS":
    print("Review warnings/failures before using model in production.")
else:
    print("Model is internally consistent and behaving as intended.")
