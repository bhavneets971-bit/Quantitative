import os
import pandas as pd
import matplotlib.pyplot as plt

# ======================================================
# Configuration
# ======================================================

CSV_PATH = "output/rolling/rolling_R_all.csv"
DIAGNOSTICS_PATH = os.path.join("Rolling", "diagnostics", "R_eigen_summary.csv")

OUTPUT_DIR = "Rolling/plots"
OUTPUT_FILE = "rolling_variances_by_year.png"
DIAGNOSTICS_FILE = "rolling_R_eigen_diagnostics.png"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Apply global plotting style
plt.style.use("default")

# ======================================================
# Load rolling covariance data
# ======================================================

df = pd.read_csv(
    CSV_PATH,
    parse_dates=["window_center_date", "window_end_date"]
)

# Keep diagonal entries only (variances)
diag = df[df["maturity_i"] == df["maturity_j"]].copy()

# Sort properly by CENTER date
diag = diag.sort_values("window_center_date")

# Stable maturity ordering
maturities = sorted(diag["maturity_i"].unique())

# ======================================================
# Plot rolling variances vs calendar time
# ======================================================

plt.figure(figsize=(12, 7))

for m in maturities:
    subset = diag[diag["maturity_i"] == m]
    plt.plot(
        subset["window_center_date"],
        subset["covariance"],
        label=m,
        linewidth=1
    )

plt.xlabel("Year")
plt.ylabel("Observation error variance")
plt.title("Rolling Observation Error Variance by Maturity")
plt.legend(ncol=2, fontsize=9)
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(os.path.join(OUTPUT_DIR, OUTPUT_FILE), dpi=300)
plt.close()

print("Saved plot to:", os.path.join(OUTPUT_DIR, OUTPUT_FILE))

# ======================================================
# Load R eigen diagnostics
# ======================================================

eig = pd.read_csv(
    DIAGNOSTICS_PATH,
    parse_dates=["date"]
).set_index("date").sort_index()

required_cols = {"trace_R", "effective_rank", "lambda1_fraction"}
missing = required_cols - set(eig.columns)
if missing:
    raise ValueError(f"R_eigen_summary.csv missing columns: {missing}")

# ======================================================
# Plot R eigen diagnostics (subplots)
# ======================================================

fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

axes[0].plot(eig.index, eig["trace_R"], linewidth=1.5)
axes[0].set_title("trace(R)")
axes[0].grid(alpha=0.3)

axes[1].plot(eig.index, eig["effective_rank"], linewidth=1.5)
axes[1].set_title("Effective Rank")
axes[1].grid(alpha=0.3)

axes[2].plot(eig.index, eig["lambda1_fraction"], linewidth=1.5)
axes[2].set_title("Leading Eigenvalue Fraction")
axes[2].set_xlabel("Date")
axes[2].grid(alpha=0.3)

fig.suptitle(
    "Rolling Observation Noise Eigen Diagnostics",
    fontsize=14,
    y=0.995
)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, DIAGNOSTICS_FILE), dpi=300)
plt.close()

print("Saved diagnostics plot to:", os.path.join(OUTPUT_DIR, DIAGNOSTICS_FILE))
