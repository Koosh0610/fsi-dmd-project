"""
08 — Animation
Side-by-side comparison of true vs. DMD-reconstructed vorticity fields.

Generates:
  fig28_comparison_animation.mp4
"""
import sys, numpy as np, matplotlib.pyplot as plt, matplotlib as mpl
from matplotlib.animation import FFMpegWriter

sys.path.insert(0, ".")
from src.load import load_cylinder
from src.dmd_runner import run_dmd

mpl.rcParams.update({"font.family": "serif", "font.size": 11})

X, dt, ny, nx = load_cylinder()
dmd = run_dmd(X, dt, svd_rank=21)
X_dmd = dmd.reconstructed_data.real
n_snaps = X.shape[1]

vmax = np.abs(X).max() * 0.8

fig, axes = plt.subplots(1, 2, figsize=(13, 3.5))
im_true = axes[0].imshow(np.zeros((ny, nx)), cmap="RdBu_r", origin="lower",
                          vmin=-vmax, vmax=vmax, animated=True)
im_dmd = axes[1].imshow(np.zeros((ny, nx)), cmap="RdBu_r", origin="lower",
                         vmin=-vmax, vmax=vmax, animated=True)
axes[0].set_title("Truth"); axes[0].axis("off")
axes[1].set_title("DMD (rank 21)"); axes[1].axis("off")
time_text = fig.suptitle("", fontsize=12)
fig.tight_layout()

writer = FFMpegWriter(fps=12, metadata={"title": "DMD reconstruction"})
out = "figures/fig28_comparison_animation.mp4"

print(f"  Rendering {n_snaps} frames ...")
with writer.saving(fig, out, dpi=150):
    for j in range(n_snaps):
        im_true.set_data(X[:, j].reshape(ny, nx, order="F"))
        im_dmd.set_data(X_dmd[:, j].reshape(ny, nx, order="F"))
        time_text.set_text(f"Snapshot {j}  (t = {j * dt:.1f})")
        writer.grab_frame()
        if (j + 1) % 30 == 0:
            print(f"    {j + 1}/{n_snaps}")

plt.close()
print(f"  {out}")
print("Done: 08_animation")
