"""
03 — DMD Fitting & Validation
Fit exact DMD, validate Strouhal number, inspect eigenvalues and modes.

Generates:
  fig06_eigenvalues_unit_circle.png   — discrete-time eigenvalues
  fig07_frequency_spectrum.png        — labeled amplitude spectrum with harmonics
  fig08_mode_growth_rates.png         — Re(ω) for each mode
  fig09_dmd_modes.png                 — first 6 spatial modes
  fig10_temporal_dynamics.png         — time evolution of selected mode coefficients
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
print(f"Fundamental: f1 = {f1:.4f}  (St target: 0.166 ± 5%)")

# ── fig06: eigenvalues on unit circle ────────────────────────────────────
theta = np.linspace(0, 2 * np.pi, 400)
fig, ax = plt.subplots(figsize=(5.5, 5.5))
ax.plot(np.cos(theta), np.sin(theta), "k--", lw=0.8)
sc = ax.scatter(dmd.eigs.real, dmd.eigs.imag, s=70,
                c=amps, cmap="viridis", edgecolor="k", zorder=3)
ax.set_aspect("equal")
ax.set(xlabel=r"Re($\lambda$)", ylabel=r"Im($\lambda$)",
       title="DMD eigenvalues on the unit circle (rank 21)")
ax.axhline(0, color="gray", lw=0.5); ax.axvline(0, color="gray", lw=0.5)
plt.colorbar(sc, ax=ax, label=r"$|b_k|$", shrink=0.8)
fig.tight_layout()
fig.savefig("figures/fig06_eigenvalues_unit_circle.png")
plt.close()
print("  fig06_eigenvalues_unit_circle.png")

# ── fig07: labeled frequency spectrum ────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
harmonic_colors = {0: "0.5", 1: "C0", 2: "C1", 3: "C2", 4: "C3", 5: "C4"}
harmonic_labels = {0: "$f_0$ (mean)", 1: f"$f_1$ (St = {f1:.3f})",
                   2: "$2f_1$", 3: "$3f_1$", 4: "$4f_1$", 5: "$5f_1$"}
labeled = set()

for f, a in zip(freqs, amps):
    af = abs(f)
    harm = -1
    if af < 1e-3:
        harm = 0
    else:
        for n in range(1, 11):
            if abs(af - n * f1) < 0.015:
                harm = min(n, 5)
                break
    c = harmonic_colors.get(harm, "C5")
    label = harmonic_labels.get(harm) if harm not in labeled else None
    if harm >= 0:
        labeled.add(harm)
    ax.bar(f, np.log10(a) if a > 0 else 0, width=0.008,
           color=c, edgecolor="k", lw=0.5, label=label)

ax.set(xlabel="Strouhal number $St$", ylabel=r"$\log_{10}|b_k|$",
       title="DMD frequency spectrum with harmonics (rank 21)",
       xlim=(-0.05, max(abs(freqs)) * 1.1))
ax.legend(loc="upper right", fontsize=10)
ax.grid(True, alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig("figures/fig07_frequency_spectrum.png")
plt.close()
print("  fig07_frequency_spectrum.png")

# ── fig08: mode growth rates ─────────────────────────────────────────────
omega = np.log(dmd.eigs) / dt
growth = omega.real
freq_cont = omega.imag / (2 * np.pi)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
ax = axes[0]
ax.scatter(freq_cont, growth, c=amps, cmap="viridis",
           s=80, edgecolor="k", zorder=3)
ax.axhline(0, color="gray", ls="--", lw=1)
ax.set(xlabel="Frequency $f$", ylabel=r"Growth rate Re($\omega$)",
       title="Growth rate vs. frequency")
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.bar(range(len(growth)), growth, color="steelblue", edgecolor="k", lw=0.5)
ax.axhline(0, color="gray", ls="--", lw=1)
ax.set(xlabel="Mode index", ylabel=r"Growth rate Re($\omega$)",
       title="Growth rate per mode")
ax.grid(True, alpha=0.3, axis="y")

fig.tight_layout()
fig.savefig("figures/fig08_mode_growth_rates.png")
plt.close()
print("  fig08_mode_growth_rates.png")
print(f"    Max |growth|: {np.max(np.abs(growth)):.6f}")

# ── fig09: first 6 DMD modes ────────────────────────────────────────────
# Sort modes by amplitude so the most important are shown first
amp_order = np.argsort(-amps)

fig, axes = plt.subplots(2, 3, figsize=(14, 6))
for k, ax in enumerate(axes.flat):
    idx = amp_order[k]
    mode = dmd.modes[:, idx].real.reshape(ny, nx, order="F")
    vmax = np.abs(mode).max()
    ax.imshow(mode, cmap="RdBu_r", origin="lower", vmin=-vmax, vmax=vmax)
    f_k = freqs[idx]
    ax.set_title(f"Mode {k}  f = {f_k:+.3f}  |b| = {amps[idx]:.1f}",
                 fontsize=10)
    ax.axis("off")
fig.suptitle("Top 6 DMD modes by amplitude (Re = 100, rank = 21)",
             fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig("figures/fig09_dmd_modes.png")
plt.close()
print("  fig09_dmd_modes.png")

# ── fig10: temporal dynamics ─────────────────────────────────────────────
t_steps = np.arange(X.shape[1])
t_phys = t_steps * dt

idx_mean = np.argmin(np.abs(freqs))
idx_f1 = np.where(keep)[0][np.argmax(amps[keep])]
idx_2f1 = np.argmin(np.abs(np.abs(freqs) - 2 * f1))

fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
for ax, idx, name in zip(axes, [idx_mean, idx_f1, idx_2f1],
    ["Mean mode ($f_0$)", f"Fundamental ($f_1$ = {f1:.3f})",
     f"2nd harmonic ($2f_1$ = {2*f1:.3f})"]):
    coeff = dmd.amplitudes[idx] * dmd.eigs[idx] ** t_steps
    ax.plot(t_phys, coeff.real, "b-", lw=1, alpha=0.8)
    ax.plot(t_phys, np.abs(coeff), "r--", lw=1, alpha=0.6, label="Envelope")
    ax.set(ylabel="Re(coeff)", title=name)
    ax.grid(True, alpha=0.3)
    if idx != idx_mean:
        ax.legend(fontsize=9)
axes[-1].set_xlabel("Non-dimensional time $t$")
fig.suptitle("Temporal dynamics of DMD modes", fontsize=13, y=1.01)
fig.tight_layout()
fig.savefig("figures/fig10_temporal_dynamics.png")
plt.close()
print("  fig10_temporal_dynamics.png")

print("Done: 03_dmd_fitting")
