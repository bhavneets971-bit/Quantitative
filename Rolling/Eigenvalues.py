import numpy as np
import pandas as pd
import os

# ==================================================
# Paths
# ==================================================

INPUT_CSV = "output/rolling/rolling_R_all.csv"
OUTPUT_DIR = "Rolling/diagnostics"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================================================
# Load rolling R data (explicit date parsing)
# ==================================================

df = pd.read_csv(
    INPUT_CSV,
    parse_dates=[
        "window_start_date",
        "window_center_date",
        "window_end_date"
    ]
)

# Use CENTER date for diagnostics
DATE_COL = "window_center_date"

# ==================================================
# Stable maturity ordering
# ==================================================

maturities = sorted(df["maturity_i"].unique())
n = len(maturities)

windows = sorted(df["window_index"].unique())

R_series = []
dates = []

# ==================================================
# Reconstruct R_t matrices
# ==================================================

for w in windows:
    sub = df[df["window_index"] == w]

    R = np.zeros((n, n))

    for i, mi in enumerate(maturities):
        for j, mj in enumerate(maturities):
            vals = sub.loc[
                (sub["maturity_i"] == mi) &
                (sub["maturity_j"] == mj),
                "covariance"
            ].values

            if len(vals) != 1:
                raise ValueError(
                    f"Non-unique covariance for window {w}, ({mi}, {mj})"
                )

            R[i, j] = vals[0]

    R_series.append(R)
    dates.append(sub[DATE_COL].iloc[0])

R_series = np.asarray(R_series)
dates = np.asarray(dates)

T = len(R_series)

# ==================================================
# Eigenvalues (core diagnostic)
# ==================================================

eigvals = np.array([
    np.sort(np.linalg.eigvalsh(R))[::-1]
    for R in R_series
])

eig_df = pd.DataFrame(
    eigvals,
    columns=[f"lambda_{i+1}" for i in range(n)]
)
eig_df.insert(0, "date", dates)

eig_df.to_csv(
    os.path.join(OUTPUT_DIR, "R_eigenvalues.csv"),
    index=False
)

# ==================================================
# Eigenvalue summary metrics
# ==================================================

trace_R = eigvals.sum(axis=1)
effective_rank = (trace_R ** 2) / (eigvals ** 2).sum(axis=1)
lambda1_fraction = eigvals[:, 0] / trace_R

summary_df = pd.DataFrame({
    "date": dates,
    "trace_R": trace_R,
    "effective_rank": effective_rank,
    "lambda1_fraction": lambda1_fraction
})

summary_df.to_csv(
    os.path.join(OUTPUT_DIR, "R_eigen_summary.csv"),
    index=False
)

# ==================================================
# Diagonal of R (variances)
# ==================================================

diag_df = pd.DataFrame(
    np.diagonal(R_series, axis1=1, axis2=2),
    columns=[f"var_{m}" for m in maturities]
)
diag_df.insert(0, "date", dates)

diag_df.to_csv(
    os.path.join(OUTPUT_DIR, "R_cov_diag.csv"),
    index=False
)

# ==================================================
# Mean off-diagonal correlation
# ==================================================

mean_corr = []

for R in R_series:
    std = np.sqrt(np.diag(R))
    Corr = R / np.outer(std, std)
    off_diag = Corr[~np.eye(n, dtype=bool)]
    mean_corr.append(off_diag.mean())

corr_df = pd.DataFrame({
    "date": dates,
    "mean_offdiag_correlation": mean_corr
})

corr_df.to_csv(
    os.path.join(OUTPUT_DIR, "R_corr_offdiag_mean.csv"),
    index=False
)

print("Diagnostics exported to CSV:")
print(" - R_eigenvalues.csv")
print(" - R_eigen_summary.csv")
print(" - R_cov_diag.csv")
print(" - R_corr_offdiag_mean.csv")
