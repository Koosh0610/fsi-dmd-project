"""
05 — Spectral & Centreline Analysis
Deeper wake physics: centreline profiles, power spectra, spatiotemporal diagram.

Generates:
  fig16_centreline_velocity_deficit.png — mean wake deficit downstream
  fig17_centreline_time_traces.png      — vorticity oscillations at downstream stations
  fig18_power_spectra_downstream.png    — FFT-based PSD at different x-positions
  fig19_spatiotemporal_centreline.png   — x-t diagram of centreline vorticity
  fig20_vorticity_rms_profiles.png      — cross-stream RMS profiles
  fig21_phase_portrait.png              — near-wake vs. far-wake phase space
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
mid_y = ny // 2

# ── fig16: centreline velocity deficit ───────────────────────────────────
U_mean = U.mean(axis=1).reshape(ny, nx, order="F")
u_cl = U_mean[mid_y, :]

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(np.arange(nx), 1.0 - u_cl, "k-", lw=1.5)
ax.set(xlabel="x (grid index)", ylabel=r"Wake deficit $1 - \bar{u}/U_\infty$",
       title="Mean centreline velocity deficit (Re = 100)")
ax.axhline(0, color="gray", ls="--", lw=0.5)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("figures/fig16_centreline_velocity_deficit.png")
plt.close()
print("  fig16_centreline_velocity_deficit.png")

# ── fig17: centreline time traces ────────────────────────────────────────
x_stations = [50, 100, 150, 200, 300, 400]
t_phys = np.arange(X.shape[1]) * dt

fig, axes = plt.subplots(len(x_stations), 1,
                          figsize=(11, 2.0 * len(x_stations)), sharex=True)
for ax, xi in zip(axes, x_stations):
    flat_idx = mid_y + xi * ny
    ax.plot(t_phys, X[flat_idx, :], "b-", lw=0.8)
    ax.set(ylabel=r"$\zeta$")
    ax.text(0.98, 0.85, f"x = {xi}", transform=ax.transAxes,
            ha="right", fontsize=10, bbox=dict(fc="white", alpha=0.8))
    ax.grid(True, alpha=0.3)
axes[-1].set_xlabel("Non-dimensional time $t$")
fig.suptitle("Centreline vorticity at downstream stations",
             fontsize=13, y=1.01)
fig.tight_layout()
fig.savefig("figures/fig17_centreline_time_traces.png")
plt.close()
print("  fig17_centreline_time_traces.png")

# ── fig18: power spectra downstream ──────────────────────────────────────
x_spec = [60, 100, 150, 250, 350]

fig, ax = plt.subplots(figsize=(9, 6))
offsets = np.arange(len(x_spec)) * 8
for i, xi in enumerate(x_spec):
    sig = X[mid_y + xi * ny, :] - X[mid_y + xi * ny, :].mean()
    N = len(sig)
    fft_v = np.fft.rfft(sig)
    fft_f = np.fft.rfftfreq(N, d=dt)
    psd = np.abs(fft_v) ** 2
    psd_db = 10 * np.log10(psd / psd.max() + 1e-20)
    ax.plot(fft_f, psd_db + offsets[i], lw=1.2, label=f"x = {xi}")

ax.axvline(0.165, color="red", ls="--", lw=0.8, alpha=0.7)
ax.text(0.175, offsets[-1] + 2, "$f_1$", color="red", fontsize=10)
ax.set(xlabel="Strouhal number $St$", ylabel="Power (dB, stacked)",
       title="Power spectra at downstream positions", xlim=(0, 1.5))
ax.legend(loc="upper right", fontsize=10)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("figures/fig18_power_spectra_downstream.png")
plt.close()
print("  fig18_power_spectra_downstream.png")

# ── fig19: spatiotemporal diagram ────────────────────────────────────────
cl_xt = np.zeros((nx, X.shape[1]))
for j in range(X.shape[1]):
    cl_xt[:, j] = X[:, j].reshape(ny, nx, order="F")[mid_y, :]

fig, ax = plt.subplots(figsize=(10, 5))
vmax = np.abs(cl_xt).max() * 0.5
im = ax.imshow(cl_xt.T, cmap="RdBu_r", origin="lower", aspect="auto",
               extent=[0, nx, 0, X.shape[1] * dt],
               vmin=-vmax, vmax=vmax)
ax.set(xlabel="x (grid index)", ylabel="Time $t$",
       title="Spatiotemporal diagram — centreline vorticity")
plt.colorbar(im, ax=ax, label=r"$\zeta$", shrink=0.8)
fig.tight_layout()
fig.savefig("figures/fig19_spatiotemporal_centreline.png")
plt.close()
print("  fig19_spatiotemporal_centreline.png")

# ── fig20: cross-stream RMS profiles ─────────────────────────────────────
X_fluct = X - X.mean(axis=1, keepdims=True)
X_rms = np.sqrt((X_fluct ** 2).mean(axis=1)).reshape(ny, nx, order="F")
y_coords = np.arange(ny) - mid_y
x_prof = [50, 80, 120, 180, 300]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ax = axes[0]
im = ax.imshow(X_rms, cmap="hot_r", origin="lower", aspect="equal")
ax.set_title("RMS vorticity fluctuation"); ax.axis("off")
plt.colorbar(im, ax=ax, shrink=0.7)
for xi in x_prof:
    ax.axvline(xi, color="white", ls="--", lw=0.8, alpha=0.7)

ax = axes[1]
for xi in x_prof:
    prof = X_rms[:, xi]
    ax.plot(prof / prof.max(), y_coords, lw=1.5, label=f"x = {xi}")
ax.set(xlabel="Normalized RMS vorticity", ylabel="y (from centreline)",
       title="Cross-stream profiles")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
ax.axhline(0, color="gray", ls="--", lw=0.5)
fig.tight_layout()
fig.savefig("figures/fig20_vorticity_rms_profiles.png")
plt.close()
print("  fig20_vorticity_rms_profiles.png")

# ── fig21: phase portrait ────────────────────────────────────────────────
x_near, x_far = 80, 200
s_near = X[mid_y + x_near * ny, :]
s_far = X[mid_y + x_far * ny, :]

fig, ax = plt.subplots(figsize=(5.5, 5))
ax.plot(s_near, s_far, "b-", lw=0.8, alpha=0.7)
ax.scatter(s_near[0], s_far[0], c="red", s=60, zorder=5, label="$t = 0$")
ax.set(xlabel=f"Vorticity at x = {x_near}",
       ylabel=f"Vorticity at x = {x_far}",
       title="Phase portrait: near-wake vs. far-wake")
ax.set_aspect("equal"); ax.grid(True, alpha=0.3); ax.legend()
fig.tight_layout()
fig.savefig("figures/fig21_phase_portrait.png")
plt.close()
print("  fig21_phase_portrait.png")

print("Done: 05_spectral_centreline")
