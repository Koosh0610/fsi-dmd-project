"""
06 — POD Comparison
Compare POD (energy-ranked) modes with DMD (frequency-ranked) modes.

Generates:
  fig22_pod_modes.png            — first 6 POD modes with energy percentages
  fig23_dmd_vs_pod_comparison.png — side-by-side dominant oscillatory mode
"""
import sys, numpy as np, matplotlib.pyplot as plt, matplotlib as mpl

sys.path.insert(0, ".")
from src.load import load_cylinder
from src.dmd_runner import run_dmd
from src.pod import compute_pod

mpl.rcParams.update({"font.family": "serif", "font.size": 11,
                      "savefig.dpi": 300, "savefig.bbox": "tight"})

X, dt, ny, nx = load_cylinder()

# ── fig22: POD modes ─────────────────────────────────────────────────────
U_modes, S, Vt, X_mean = compute_pod(X, n_modes=6)

fig, axes = plt.subplots(2, 3, figsize=(14, 6))
for k, ax in enumerate(axes.flat):
    mode = U_modes[:, k].reshape(ny, nx, order="F")
    vmax = np.abs(mode).max()
    ax.imshow(mode, cmap="RdBu_r", origin="lower", vmin=-vmax, vmax=vmax)
    energy = S[k] ** 2 / (S ** 2).sum() * 100
    ax.set_title(f"POD mode {k}  ({energy:.1f}% energy)", fontsize=10)
    ax.axis("off")
fig.suptitle("First 6 POD modes (Re = 100)", fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig("figures/fig22_pod_modes.png")
plt.close()
print("  fig22_pod_modes.png")

# ── fig23: DMD vs. POD side-by-side ──────────────────────────────────────
dmd = run_dmd(X, dt, svd_rank=21)
freqs = dmd.frequency
amps = np.abs(dmd.amplitudes)
keep = np.abs(freqs) > 1e-3
k_dom = np.where(keep)[0][np.argmax(amps[keep])]

fig, axes = plt.subplots(1, 2, figsize=(13, 3.5))

pod_mode = U_modes[:, 0].reshape(ny, nx, order="F")
vmax = np.abs(pod_mode).max()
axes[0].imshow(pod_mode, cmap="RdBu_r", origin="lower", vmin=-vmax, vmax=vmax)
axes[0].set_title("POD mode 0 (dominant)", fontsize=11)
axes[0].axis("off")

dmd_mode = dmd.modes[:, k_dom].real.reshape(ny, nx, order="F")
vmax = np.abs(dmd_mode).max()
axes[1].imshow(dmd_mode, cmap="RdBu_r", origin="lower", vmin=-vmax, vmax=vmax)
axes[1].set_title(f"DMD mode (f = {abs(freqs[k_dom]):.3f})", fontsize=11)
axes[1].axis("off")

fig.suptitle("POD vs. DMD: dominant oscillatory mode", fontsize=13, y=1.04)
fig.tight_layout()
fig.savefig("figures/fig23_dmd_vs_pod_comparison.png")
plt.close()
print("  fig23_dmd_vs_pod_comparison.png")

print("Done: 06_pod_comparison")
