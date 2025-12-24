import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import FFMpegWriter
import os

INPUT_CSV = "output/rolling/rolling_R_all.csv"
OUTPUT_DIR = "Rolling/plots"

STEP = 20
FPS = 20
DPI = 120

CMAP_COV = "inferno"
CMAP_CORR = "coolwarm"

mpl.rcParams["animation.writer"] = "ffmpeg"
mpl.rcParams["animation.ffmpeg_path"] = (
    r"C:\Users\User\AppData\Local\Microsoft\WinGet\Packages"
    r"\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
    r"\ffmpeg-8.0.1-full_build\bin\ffmpeg.exe"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# LOAD DATA

df = pd.read_csv(INPUT_CSV)

maturities = df["maturity_i"].unique()
n = len(maturities)
windows = sorted(df["window_index"].unique())

R_series = []
dates = []

for w in windows:
    sub = df[df["window_index"] == w]
    R = np.zeros((n, n))
    for i, mi in enumerate(maturities):
        for j, mj in enumerate(maturities):
            R[i, j] = sub.loc[
                (sub["maturity_i"] == mi) &
                (sub["maturity_j"] == mj),
                "covariance"
            ].values[0]
    R_series.append(R)
    dates.append(sub["window_end_date"].iloc[0])

R_series = np.array(R_series)
dates = np.array(dates)

# DOWNSAMPLE

R_series = R_series[::STEP]
dates = dates[::STEP]
T = len(R_series)

print(f"Animating {T} frames")

# Correlation matrices
R_corr = np.zeros_like(R_series)
for t in range(T):
    std = np.sqrt(np.diag(R_series[t]))
    R_corr[t] = R_series[t] / np.outer(std, std)

R_corr = np.clip(R_corr, -1.0, 1.0)

# Eigenvalues
eigvals = np.array([
    np.sort(np.linalg.eigvalsh(R))[::-1]
    for R in R_series
])

writer = FFMpegWriter(fps=FPS, bitrate=1800)

# COVARIANCE HEATMAP ANIMATION

vmin_cov = np.min(R_series)
vmax_cov = np.max(R_series)

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(R_series[0], cmap=CMAP_COV, vmin=vmin_cov, vmax=vmax_cov)
ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels(maturities, rotation=45)
ax.set_yticklabels(maturities)
title = ax.set_title(f"Observation Error Covariance\n{dates[0]}")
plt.colorbar(im, ax=ax, fraction=0.046)

def update_cov(frame):
    im.set_array(R_series[frame])
    title.set_text(f"Observation Error Covariance\n{dates[frame]}")
    return im, title

ani_cov = animation.FuncAnimation(fig, update_cov, frames=T)
ani_cov.save(
    os.path.join(OUTPUT_DIR, "R_covariance_heatmap.mp4"),
    writer=writer,
    dpi=DPI
)
plt.close()

print("Saved R_covariance_heatmap.mp4")

# CORRELATION HEATMAP ANIMATION

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(R_corr[0], cmap=CMAP_CORR, vmin=-1, vmax=1)
ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels(maturities, rotation=45)
ax.set_yticklabels(maturities)
title = ax.set_title(f"Observation Error Correlation\n{dates[0]}")
plt.colorbar(im, ax=ax, fraction=0.046)

def update_corr(frame):
    im.set_array(R_corr[frame])
    title.set_text(f"Observation Error Correlation\n{dates[frame]}")
    return im, title

ani_corr = animation.FuncAnimation(fig, update_corr, frames=T)
ani_corr.save(
    os.path.join(OUTPUT_DIR, "R_correlation_heatmap.mp4"),
    writer=writer,
    dpi=DPI
)
plt.close()

print("Saved R_correlation_heatmap.mp4")

# EIGENVALUE ANIMATION

fig, ax = plt.subplots(figsize=(8, 5))
lines = []

for i in range(n):
    line, = ax.plot([], [], lw=2, label=f"$\\lambda_{i+1}$")
    lines.append(line)

ax.set_yscale("log")
ax.set_xlim(0, T)
ax.set_ylim(
    np.min(eigvals[eigvals > 0]),
    np.max(eigvals)
)
ax.set_xlabel("Rolling window index")
ax.set_ylabel("Eigenvalue (log scale)")
ax.legend()
title = ax.set_title("Observation Error Eigenvalue Spectrum")

def update_eig(frame):
    for i, line in enumerate(lines):
        line.set_data(
            range(frame + 1),
            eigvals[:frame + 1, i]
        )
    title.set_text(
        f"Observation Error Eigenvalue Spectrum\n{dates[frame]}"
    )
    return lines + [title]

ani_eig = animation.FuncAnimation(fig, update_eig, frames=T)
ani_eig.save(
    os.path.join(OUTPUT_DIR, "R_eigenvalues.mp4"),
    writer=writer,
    dpi=DPI
)
plt.close()

print("Saved R_eigenvalues.mp4")

print("All three animations generated successfully.")
