"""
04 — Reconstruction
How well DMD reconstructs the original data, and what each frequency contributes.

Generates:
  fig11_true_vs_reconstructed.png       — side-by-side single snapshot
  fig12_reconstruction_error_vs_rank.png — error sweep over ranks
  fig13_selective_reconstruction.png     — reconstruction by frequency group
  fig14_progressive_harmonic_buildup.png — adding harmonics one at a time
  fig15_error_by_harmonic.png            — bar chart of error vs. harmonic count
"""
import sys, numpy as np, matplotlib.pyplot as plt, matplotlib as mpl

sys.path.insert(0, ".")
from src.load import load_cylinder
from src.dmd_runner import run_dmd

mpl.rcParams.update({"font.family": "serif", "font.size": 11,
                      "savefig.dpi": 300, "savefig.bbox": "tight"})

X, dt, ny, nx = load_cylinder()
dmd = run_dmd(X, dt, svd_rank=21)

freqs = dmd.frequency
amps = np.abs(dmd.amplitudes)
keep = np.abs(freqs) > 1e-3
f1 = abs(freqs[np.where(keep)[0][np.argmax(amps[keep])]])
t_steps = np.arange(X.shape[1])
X_dmd = dmd.reconstructed_data.real


def find_modes(freqs, target, tol=0.015):
    return np.where(np.abs(np.abs(freqs) - target) < tol)[0]


def reconstruct_subset(dmd, indices, t_steps):
    Phi = dmd.modes[:, indices]
    b = dmd.amplitudes[indices]
    lam = dmd.eigs[indices]
    T = np.array([b[i] * lam[i] ** t_steps for i in range(len(indices))])
    return (Phi @ T).real


# ── fig11: true vs. reconstructed ────────────────────────────────────────
j = 75
fig, axes = plt.subplots(1, 2, figsize=(13, 3.5))
for ax, data, title in zip(axes, [X, X_dmd], ["Truth", "DMD (rank 21)"]):
    field = data[:, j].reshape(ny, nx, order="F")
    vmax = np.abs(field).max()
    ax.imshow(field, cmap="RdBu_r", origin="lower", vmin=-vmax, vmax=vmax)
    ax.set_title(title); ax.axis("off")
err = np.linalg.norm(X - X_dmd) / np.linalg.norm(X)
fig.suptitle(f"True vs. DMD reconstruction (snapshot 75, global error = {err:.3%})",
             fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig("figures/fig11_true_vs_reconstructed.png")
plt.close()
print(f"  fig11_true_vs_reconstructed.png  (error = {err:.3%})")

# ── fig12: error vs. rank ────────────────────────────────────────────────
ranks = [1, 3, 5, 7, 11, 15, 21, 31, 41, 61, 81]
errs = []
for r in ranks:
    d = run_dmd(X, dt, svd_rank=r)
    Xr = d.reconstructed_data.real
    errs.append(np.linalg.norm(X - Xr) / np.linalg.norm(X))

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.semilogy(ranks, errs, "o-", lw=2, markersize=7, color="steelblue")
for r, e in zip(ranks, errs):
    ax.annotate(f"{e:.1%}", (r, e), textcoords="offset points",
                xytext=(6, 8), fontsize=8)
ax.set(xlabel="SVD rank $r$", ylabel="Relative Frobenius error",
       title="Reconstruction error vs. truncation rank")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("figures/fig12_reconstruction_error_vs_rank.png")
plt.close()
print("  fig12_reconstruction_error_vs_rank.png")

# ── fig13: selective reconstruction ──────────────────────────────────────
idx_mean = find_modes(freqs, 0.0, tol=1e-3)
idx_f1 = find_modes(freqs, f1)
idx_2f1 = find_modes(freqs, 2 * f1)
idx_3f1 = find_modes(freqs, 3 * f1)
idx_4f1 = find_modes(freqs, 4 * f1)
idx_5f1 = find_modes(freqs, 5 * f1)

groups = [
    ("(a)  Original", None),
    ("(b)  Mean only ($f_0$)", idx_mean),
    ("(c)  Mean + $f_1$", np.concatenate([idx_mean, idx_f1])),
    ("(d)  Mean + $f_1$ + $2f_1$", np.concatenate([idx_mean, idx_f1, idx_2f1])),
    ("(e)  All harmonics ($f_0 ... 5f_1$)",
     np.concatenate([idx_mean, idx_f1, idx_2f1, idx_3f1, idx_4f1, idx_5f1])),
    ("(f)  Full DMD (rank 21)", np.arange(len(dmd.eigs))),
]

fig, axes = plt.subplots(len(groups), 1, figsize=(13, 2.8 * len(groups)))
vmax_g = np.abs(X[:, j]).max()
for ax, (title, indices) in zip(axes, groups):
    if indices is None:
        field = X[:, j]
    else:
        field = reconstruct_subset(dmd, indices, t_steps)[:, j]
    ax.imshow(field.reshape(ny, nx, order="F"), cmap="RdBu_r",
              origin="lower", vmin=-vmax_g, vmax=vmax_g, aspect="equal")
    ax.set_title(title, fontsize=12, loc="left"); ax.axis("off")
fig.suptitle("Selective DMD reconstruction by frequency group",
             fontsize=14, y=1.01)
fig.tight_layout()
fig.savefig("figures/fig13_selective_reconstruction.png")
plt.close()
print("  fig13_selective_reconstruction.png")

# ── fig14: progressive harmonic buildup ──────────────────────────────────
harm_groups = [idx_f1, idx_2f1, idx_3f1, idx_4f1, idx_5f1]
harm_names = ["$f_1$", "$f_1 + 2f_1$", "$f_1 + ... + 3f_1$",
              "$f_1 + ... + 4f_1$", "$f_1 + ... + 5f_1$"]

fig, axes = plt.subplots(6, 1, figsize=(13, 2.6 * 6))
vmax = np.abs(X[:, j]).max()
axes[0].imshow(X[:, j].reshape(ny, nx, order="F"), cmap="RdBu_r",
               origin="lower", vmin=-vmax, vmax=vmax, aspect="equal")
axes[0].set_title("(a) Original", fontsize=12, loc="left"); axes[0].axis("off")

cum = idx_mean.copy()
buildup_errors = []
for i, (g, name) in enumerate(zip(harm_groups, harm_names)):
    cum = np.concatenate([cum, g])
    Xr = reconstruct_subset(dmd, cum, t_steps)
    e = np.linalg.norm(X[:, j] - Xr[:, j]) / np.linalg.norm(X[:, j])
    buildup_errors.append(e)
    axes[i + 1].imshow(Xr[:, j].reshape(ny, nx, order="F"), cmap="RdBu_r",
                        origin="lower", vmin=-vmax, vmax=vmax, aspect="equal")
    axes[i + 1].set_title(f"({chr(98 + i)}) Mean + {name}  (error {e:.1%})",
                           fontsize=12, loc="left")
    axes[i + 1].axis("off")

fig.suptitle("Progressive harmonic buildup", fontsize=14, y=1.01)
fig.tight_layout()
fig.savefig("figures/fig14_progressive_harmonic_buildup.png")
plt.close()
print("  fig14_progressive_harmonic_buildup.png")

# ── fig15: error by harmonic count ───────────────────────────────────────
cum = idx_mean.copy()
all_errors = [np.linalg.norm(X - reconstruct_subset(dmd, cum, t_steps))
              / np.linalg.norm(X)]
names = ["$f_0$"]
for g, tag in zip(harm_groups, ["+$f_1$", "+$2f_1$", "+$3f_1$",
                                 "+$4f_1$", "+$5f_1$"]):
    cum = np.concatenate([cum, g])
    all_errors.append(np.linalg.norm(X - reconstruct_subset(dmd, cum, t_steps))
                      / np.linalg.norm(X))
    names.append(tag)

fig, ax = plt.subplots(figsize=(8, 4.5))
bars = ax.bar(range(len(all_errors)), [e * 100 for e in all_errors],
              color="steelblue", edgecolor="k")
ax.set_xticks(range(len(all_errors)))
ax.set_xticklabels(names, fontsize=10)
ax.set(xlabel="Modes included (cumulative)", ylabel="Relative error (%)",
       title="Reconstruction error as harmonics are added")
ax.grid(True, alpha=0.3, axis="y")
for bar, e in zip(bars, all_errors):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
            f"{e:.1%}", ha="center", va="bottom", fontsize=9)
fig.tight_layout()
fig.savefig("figures/fig15_error_by_harmonic.png")
plt.close()
print("  fig15_error_by_harmonic.png")

print("Done: 04_reconstruction")
