"""
07 — Sensitivity Analysis & BOPDMD Comparison
Robustness of DMD results, and comparison with Optimized DMD.

Generates:
  fig24_strouhal_vs_rank.png         — St stability across ranks
  fig25_rank_sensitivity_heatmap.png — error vs. (rank, snapshot count)
  fig26_dmd_vs_bopdmd_eigenvalues.png— eigenvalue comparison
  fig27_dmd_vs_bopdmd_spectra.png    — amplitude spectra comparison
"""
import sys, numpy as np, matplotlib.pyplot as plt, matplotlib as mpl

sys.path.insert(0, ".")
from src.load import load_cylinder
from src.dmd_runner import run_dmd

mpl.rcParams.update({"font.family": "serif", "font.size": 11,
                      "savefig.dpi": 300, "savefig.bbox": "tight"})

X, dt, ny, nx = load_cylinder()

# ── fig24: Strouhal vs. rank ─────────────────────────────────────────────
test_ranks = [3, 5, 7, 9, 11, 15, 21, 31, 41, 51, 61, 81]
st_vals = []
for r in test_ranks:
    d = run_dmd(X, dt, svd_rank=r)
    f = d.frequency
    a = np.abs(d.amplitudes)
    k = np.abs(f) > 1e-3
    st_vals.append(abs(f[k][np.argmax(a[k])]) if k.any() else np.nan)

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(test_ranks, st_vals, "o-", lw=2, markersize=8, color="steelblue")
ax.axhline(0.166, color="red", ls="--", lw=1.5, label="St = 0.166 (Williamson 1996)")
ax.fill_between([test_ranks[0], test_ranks[-1]], 0.158, 0.174,
                alpha=0.15, color="red", label=r"$\pm 5\%$ tolerance")
ax.set(xlabel="SVD rank $r$", ylabel="Recovered Strouhal number",
       title="Strouhal number stability vs. truncation rank")
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("figures/fig24_strouhal_vs_rank.png")
plt.close()
print("  fig24_strouhal_vs_rank.png")
for r, s in zip(test_ranks, st_vals):
    print(f"    r={r:3d}  St={s:.4f}  {'OK' if 0.158 < s < 0.174 else 'FAIL'}")

# ── fig25: rank-sensitivity heatmap ──────────────────────────────────────
ranks = [3, 5, 7, 11, 15, 21, 31, 41]
n_snaps = [20, 40, 60, 80, 100, 120, 151]
err_mat = np.full((len(ranks), len(n_snaps)), np.nan)

for i, r in enumerate(ranks):
    for j, ns in enumerate(n_snaps):
        try:
            d = run_dmd(X[:, :ns], dt, svd_rank=min(r, ns - 2))
            Xr = d.reconstructed_data.real
            if Xr.shape == X[:, :ns].shape:
                err_mat[i, j] = np.linalg.norm(X[:, :ns] - Xr) / np.linalg.norm(X[:, :ns])
        except Exception:
            pass

fig, ax = plt.subplots(figsize=(9, 6))
im = ax.imshow(np.log10(err_mat + 1e-10), cmap="YlOrRd_r",
               aspect="auto", origin="lower")
ax.set_xticks(range(len(n_snaps))); ax.set_xticklabels(n_snaps)
ax.set_yticks(range(len(ranks))); ax.set_yticklabels(ranks)
ax.set(xlabel="Number of snapshots", ylabel="SVD rank $r$",
       title=r"Reconstruction error: $\log_{10}(\epsilon)$")
for i in range(len(ranks)):
    for j in range(len(n_snaps)):
        v = err_mat[i, j]
        if not np.isnan(v):
            txt = f"{v:.1%}" if v > 0.001 else f"{v:.0e}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                    color="white" if v > 0.05 else "black")
plt.colorbar(im, ax=ax, label=r"$\log_{10}(\epsilon)$", shrink=0.8)
fig.tight_layout()
fig.savefig("figures/fig25_rank_sensitivity_heatmap.png")
plt.close()
print("  fig25_rank_sensitivity_heatmap.png")

# ── fig26 & fig27: BOPDMD comparison ─────────────────────────────────────
from pydmd import BOPDMD
from pydmd.preprocessing import zero_mean_preprocessing

t = np.arange(X.shape[1]) * dt
bop = BOPDMD(svd_rank=11, num_trials=100, trial_size=0.8,
             eig_constraints={"imag", "conjugate_pairs"})
bop = zero_mean_preprocessing(bop)
bop.fit(X, t=t)

bop_omega = bop.eigs
bop_freqs = bop_omega.imag / (2 * np.pi)
bop_amps = np.abs(bop.amplitudes)

dmd_std = run_dmd(X, dt, svd_rank=21)
theta = np.linspace(0, 2 * np.pi, 400)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ax = axes[0]
ax.plot(np.cos(theta), np.sin(theta), "k--", lw=0.8)
ax.scatter(dmd_std.eigs.real, dmd_std.eigs.imag, s=70, c="steelblue",
           edgecolor="k", zorder=3, label="DMD (r = 21)")
ax.set_aspect("equal")
ax.set(xlabel=r"Re($\lambda$)", ylabel=r"Im($\lambda$)",
       title="Standard DMD — discrete eigenvalues")
ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1]
ax.scatter(bop_omega.real, bop_omega.imag, s=70, c="coral",
           edgecolor="k", zorder=3, label="BOPDMD (r = 11)")
ax.axvline(0, color="gray", ls="--", lw=0.8)
ax.set(xlabel=r"Re($\omega$)", ylabel=r"Im($\omega$)",
       title="BOPDMD — continuous eigenvalues")
ax.legend(); ax.grid(True, alpha=0.3)

fig.suptitle("DMD vs. BOPDMD eigenvalues", fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig("figures/fig26_dmd_vs_bopdmd_eigenvalues.png")
plt.close()
print("  fig26_dmd_vs_bopdmd_eigenvalues.png")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
axes[0].stem(dmd_std.frequency, np.abs(dmd_std.amplitudes),
             basefmt=" ", linefmt="C0-", markerfmt="C0o")
axes[0].set(xlabel="Frequency $f$", ylabel="$|b_k|$",
            title="Standard DMD (r = 21)")
axes[0].grid(True, alpha=0.3)

axes[1].stem(bop_freqs, bop_amps,
             basefmt=" ", linefmt="C1-", markerfmt="C1o")
axes[1].set(xlabel="Frequency $f$", ylabel="$|b_k|$",
            title="BOPDMD (r = 11)")
axes[1].grid(True, alpha=0.3)

fig.suptitle("Amplitude spectra: DMD vs. BOPDMD", fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig("figures/fig27_dmd_vs_bopdmd_spectra.png")
plt.close()
print("  fig27_dmd_vs_bopdmd_spectra.png")

print("Done: 07_sensitivity_bopdmd")
