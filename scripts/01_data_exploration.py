"""
01 — Data Exploration
What the cylinder wake vorticity data looks like.

Generates:
  fig01_snapshot_panel.png       — 6 vorticity snapshots showing vortex shedding
  fig02_time_mean_vorticity.png  — period-averaged vorticity field
  fig03_time_mean_velocity.png   — period-averaged streamwise velocity
  fig04_fluctuation_field.png    — full vs. mean-subtracted vorticity
"""
import sys, numpy as np, matplotlib.pyplot as plt, matplotlib as mpl
import scipy.io as sio

sys.path.insert(0, ".")
from src.load import load_cylinder

mpl.rcParams.update({"font.family": "serif", "font.size": 11,
                      "savefig.dpi": 300, "savefig.bbox": "tight"})

X, dt, ny, nx = load_cylinder()
mat = sio.loadmat("data/CYLINDER_ALL.mat")
U = mat["UALL"].astype(np.float64)

print(f"Loaded: X {X.shape}, dt={dt}, grid {ny}×{nx}")

# ── fig01: snapshot panel ────────────────────────────────────────────────
indices = [0, 25, 50, 75, 100, 150]
fig, axes = plt.subplots(2, 3, figsize=(14, 6))
for ax, idx in zip(axes.flat, indices):
    field = X[:, idx].reshape(ny, nx, order="F")
    vmax = np.abs(field).max()
    ax.imshow(field, cmap="RdBu_r", origin="lower", vmin=-vmax, vmax=vmax)
    ax.set_title(f"Snapshot {idx}")
    ax.axis("off")
fig.suptitle("Cylinder wake vorticity at Re = 100", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig("figures/fig01_snapshot_panel.png")
plt.close()
print("  fig01_snapshot_panel.png")

# ── fig02: time-mean vorticity ───────────────────────────────────────────
X_mean = X.mean(axis=1)
field_mean = X_mean.reshape(ny, nx, order="F")

fig, ax = plt.subplots(figsize=(12, 3.5))
vmax = np.abs(field_mean).max()
im = ax.imshow(field_mean, cmap="RdBu_r", origin="lower",
               vmin=-vmax, vmax=vmax, aspect="equal")
ax.set_title("Time-mean vorticity field (Re = 100)")
ax.set_xlabel("x"); ax.set_ylabel("y")
plt.colorbar(im, ax=ax, label="Mean vorticity", shrink=0.8)
fig.tight_layout()
fig.savefig("figures/fig02_time_mean_vorticity.png")
plt.close()
print("  fig02_time_mean_vorticity.png")

# ── fig03: time-mean streamwise velocity ─────────────────────────────────
U_mean = U.mean(axis=1).reshape(ny, nx, order="F")

fig, ax = plt.subplots(figsize=(12, 3.5))
im = ax.imshow(U_mean, cmap="coolwarm", origin="lower", aspect="equal")
ax.set_title("Time-mean streamwise velocity (Re = 100)")
ax.set_xlabel("x"); ax.set_ylabel("y")
plt.colorbar(im, ax=ax, label=r"$\bar{u}$", shrink=0.8)
fig.tight_layout()
fig.savefig("figures/fig03_time_mean_velocity.png")
plt.close()
print("  fig03_time_mean_velocity.png")

# ── fig04: fluctuation field ─────────────────────────────────────────────
X_fluct = X - X_mean[:, None]

fig, axes = plt.subplots(2, 1, figsize=(12, 6))
for ax, (data, title) in zip(axes, [
    (X[:, 75], "Full vorticity (snapshot 75)"),
    (X_fluct[:, 75], "Fluctuation vorticity (mean removed)")]):
    field = data.reshape(ny, nx, order="F")
    vmax = np.abs(field).max()
    im = ax.imshow(field, cmap="RdBu_r", origin="lower",
                   vmin=-vmax, vmax=vmax, aspect="equal")
    ax.set_title(title); ax.axis("off")
    plt.colorbar(im, ax=ax, shrink=0.8)
fig.tight_layout()
fig.savefig("figures/fig04_fluctuation_field.png")
plt.close()
print("  fig04_fluctuation_field.png")

print("Done: 01_data_exploration")
