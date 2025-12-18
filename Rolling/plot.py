import os
import pandas as pd
import matplotlib.pyplot as plt

# ======================================================
# Configuration
# ======================================================
CSV_PATH = "rolling_R_all.csv"   # adjust path if needed
OUTPUT_DIR = "Rolling/plots"
OUTPUT_FILE = "rolling_variances_all_maturities.png"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ======================================================
# Load data
# ======================================================
df = pd.read_csv(CSV_PATH)

# Keep only diagonal entries (variances)
diag = df[df["maturity_i"] == df["maturity_j"]]

# Get maturities in consistent order
maturities = diag["maturity_i"].unique()

# ======================================================
# Plot rolling variances
# ======================================================
plt.figure(figsize=(10, 6))

for m in maturities:
    subset = diag[diag["maturity_i"] == m]
    plt.plot(
        subset["time_index"],
        subset["covariance"],
        label=m,
        linewidth=1
    )

plt.xlabel("Rolling window index")
plt.ylabel("Observation error variance")
plt.title("Rolling Observation Error Variance (All Maturities)")
plt.legend(ncol=2, fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, OUTPUT_FILE))
plt.close()

print("Rolling variance plot saved to:", os.path.join(OUTPUT_DIR, OUTPUT_FILE))
