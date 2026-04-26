"""
02 — SVD & Rank Selection
Singular value spectrum guides the truncation rank choice.

Generates:
  fig05_singular_spectrum.png — log-scale singular values + cumulative energy
"""
import sys, numpy as np, matplotlib.pyplot as plt, matplotlib as mpl

sys.path.insert(0, ".")
from src.load import load_cylinder

mpl.rcParams.update({"font.family": "serif", "font.size": 11,
                      "savefig.dpi": 300, "savefig.bbox": "tight"})

X, dt, ny, nx = load_cylinder()

# ── fig05: singular spectrum ─────────────────────────────────────────────
s = np.linalg.svd(X[:, :-1], compute_uv=False)
k = np.arange(1, len(s) + 1)
max_show = 50

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

ax = axes[0]
ax.semilogy(k[:max_show], s[:max_show], "o-", markersize=4)
ax.axvline(21, color="red", ls="--", lw=1, alpha=0.7, label="r = 21")
ax.set(xlabel="Index k", ylabel=r"$\sigma_k$",
       title="Singular spectrum (log scale)")
ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1]
cum = np.cumsum(s[:max_show]) / s.sum()
ax.plot(k[:max_show], cum, "o-", markersize=4)
ax.axvline(21, color="red", ls="--", lw=1, alpha=0.7, label="r = 21")
ax.axhline(cum[20], color="red", ls=":", lw=0.8, alpha=0.5)
ax.text(25, cum[20] - 0.02, f"{cum[20]:.3f}", fontsize=9, color="red")
ax.set(xlabel="Index k", ylabel="Cumulative energy fraction",
       title="Energy capture", ylim=(0, 1.02))
ax.legend(); ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig("figures/fig05_singular_spectrum.png")
plt.close()
print("  fig05_singular_spectrum.png")

print(f"  Top 10 singular values: {s[:10].round(1)}")
print(f"  Energy at r=21: {cum[20]:.4f}")
print("Done: 02_svd_rank")
