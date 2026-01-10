import os
import pandas as pd
import matplotlib.pyplot as plt

# ======================================================
# Configuration
# ======================================================

CSV_PATH = "output/rolling/rolling_R_all.csv"
OUTPUT_DIR = "Rolling/plots"
OUTPUT_FILE = "rolling_variances_by_year.png"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Apply global plotting style (optional but clean)
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
