# DMD Analysis of Flow Past a Cylinder — A Self-Contained Undergraduate Project Plan

> **For:** Kush · **Environment:** Remote SSH server + local laptop for editing · **Library:** PyDMD ≥ 2025.x · **Data:** Kutz & Brunton cylinder vorticity snapshots (Re = 100) · **Duration:** 3–4 weeks · **Style:** beginner-friendly, theory- and visuals-heavy.

This document is intended to be executed top-to-bottom. Anyone with shell access, a remote Python machine, and a few hours can start producing real DMD results within a day. No prior DMD or fluid-dynamics expertise is assumed.

---

## Table of Contents

0. [How to use this document](#0-how-to-use-this-document)
1. [Project at a glance](#1-project-at-a-glance)
2. [Background — vortex shedding in 30 minutes](#2-background--vortex-shedding-in-30-minutes)
3. [DMD theory — the math you actually need](#3-dmd-theory--the-math-you-actually-need)
4. [Why the three professor-shared papers, and what to take from each](#4-the-three-papers)
5. [The dataset — what it is and how to get it on the remote machine](#5-the-dataset)
6. [Remote SSH workflow & environment setup](#6-remote-ssh-workflow--environment-setup)
7. [Repository layout](#7-repository-layout)
8. [Week-by-week execution plan](#8-week-by-week-execution-plan)
9. [Key code snippets (PyDMD, plotting, validation)](#9-key-code-snippets)
10. [Visualization recipes — making figures that look professional](#10-visualization-recipes)
11. [Troubleshooting & sanity checks](#11-troubleshooting--sanity-checks)
12. [Stretch goals (only if you finish early)](#12-stretch-goals)
13. [Report skeleton (what to write)](#13-report-skeleton)
14. [Slide outline (10 slides, ~12 min talk)](#14-slide-outline)
15. [Grading rubric (use this to self-assess)](#15-grading-rubric)
16. [Citations & resources](#16-citations--resources)

---

## 0. How to use this document

- Treat sections 1–5 as **read-once reference**. Section 8 is your **weekly checklist**.
- Every code block is **pseudocode + minimal real PyDMD calls** — enough to run, but you still write the glue. This is intentional; the goal is learning, not copy-pasting a finished project.
- `[VERIFY]` markers flag claims you should reproduce numerically before trusting (e.g., the Strouhal number).
- `[ON-SERVER]` marks shell commands meant for the remote SSH machine.
- `[LOCAL]` marks commands meant for your laptop.

---

## 1. Project at a glance

### Title
**Dynamic Mode Decomposition of the Two-Dimensional Cylinder Wake at Re = 100: Identification of Coherent Structures and Reduced-Order Reconstruction Using PyDMD**

### One-paragraph elevator pitch
The flow behind a circular cylinder at Re ≈ 100 sheds vortices in a clean, periodic pattern called the von Kármán street. The full velocity field lives in tens of thousands of dimensions, but its dynamics are essentially low-dimensional: a steady mean plus a single shedding frequency and its harmonics. Dynamic Mode Decomposition (DMD) is a data-driven algorithm that recovers these few dynamically-relevant modes directly from snapshots, without knowing the governing equations. This project applies PyDMD to the canonical Kutz–Brunton cylinder dataset, identifies the dominant modes and their frequencies, validates them against the canonical Strouhal number St ≈ 0.166 (Williamson 1996), and quantifies how well a low-rank DMD model reconstructs the original snapshots.

### Concrete deliverables
1. A GitHub repository with notebooks, source modules, and a `requirements.txt`.
2. A short report (~10 pages) — see the skeleton in §13.
3. A 10-slide presentation — see §14.
4. Five key figures (singular spectrum, eigenvalues on the unit circle, dominant modes, reconstruction error vs. rank, true vs. reconstructed snapshot panel).

### Why this scope works for an undergrad
- The flow is **2D, laminar, periodic** — no turbulence modeling, no 3D rendering.
- The dataset is **151 snapshots** — fits in RAM as a ~110 MB matrix (`float64`, 449·199·151 ≈ 13.4 M doubles).
- DMD is **one SVD plus one small eigendecomposition** — runs in seconds.
- The expected answer (St ≈ 0.166 (Williamson 1996), ~3–5 dominant modes) is well-known, so you have a clear correctness target.

---

## 2. Background — vortex shedding in 30 minutes

### 2.1 The physical setup
A uniform free-stream of speed *U* approaches a circular cylinder of diameter *D*. The Reynolds number is

$$
\mathrm{Re} = \frac{U D}{\nu},
$$

where *ν* is the kinematic viscosity of the fluid. At Re ≈ 100 the flow is two-dimensional and laminar, but unsteady: vortices alternately shed from the top and bottom of the cylinder, forming the **von Kármán vortex street**.

### 2.2 The Strouhal number
The shedding frequency *f* is non-dimensionalized as

$$
\mathrm{St} = \frac{f D}{U}.
$$

For a circular cylinder at Re = 100, decades of experiments and DNS converge on the canonical value **St = 0.166** (Williamson 1996). The accepted tolerance for "we got it right" is ±5%, i.e., 0.158 ≤ St ≤ 0.174. This is the single most important number in the project — it is the number your DMD pipeline must reproduce. **Use 0.166 consistently throughout your report and tests.**

### 2.3 What "snapshot" means
A snapshot is the entire flow field — here, the scalar **vorticity** $\zeta = \partial v/\partial x - \partial u/\partial y$ — on a 2D grid at one instant in time. (We use $\zeta$ for vorticity and reserve $\omega$ for the continuous-time DMD eigenvalue introduced in §3.) The dataset contains 151 such snapshots, each on a 449 × 199 grid, separated by Δ*t* = 0.02 (in non-dimensional time units where *D* = *U* = 1).

### 2.4 Why look at vorticity instead of velocity?
Vorticity localizes the vortices into compact, easy-to-see blobs. Velocity components (*u*, *v*) work equally well for DMD, but vorticity gives the cleanest mode plots, which is why every textbook uses it.

---

## 3. DMD theory — the math you actually need

This section is intentionally rigorous. Read it once, then come back to it when writing the report.

### 3.1 Snapshots and the linear-evolution assumption
Stack the *m* + 1 snapshots as columns of a tall, thin matrix:

$$
X = [\mathbf{x}_0 \mid \mathbf{x}_1 \mid \cdots \mid \mathbf{x}_m] \in \mathbb{R}^{n \times (m+1)}, \qquad n \gg m.
$$

For our dataset *n* = 449·199 = 89 351 and *m* + 1 = 151. Define two shifted matrices:

$$
X_1 = [\mathbf{x}_0, \dots, \mathbf{x}_{m-1}] \in \mathbb{R}^{n \times m}, \qquad
X_2 = [\mathbf{x}_1, \dots, \mathbf{x}_{m}]   \in \mathbb{R}^{n \times m}.
$$

DMD's central assumption is that there is a *constant* linear operator *A* such that

$$
\mathbf{x}_{k+1} \approx A\, \mathbf{x}_k \quad \Longleftrightarrow \quad X_2 \approx A\, X_1.
$$

Even though the Navier–Stokes equations are nonlinear, this approximation is excellent for the cylinder wake because the dynamics live near a low-dimensional limit cycle — a setting where DMD coincides with a finite-dimensional approximation of the **Koopman operator**.

### 3.2 The exact-DMD algorithm (Tu et al. 2014)

**Step 1 — SVD of *X*₁:**
$$
X_1 = U \Sigma V^*.
$$
Truncate to rank *r* (keep the top *r* singular values). Call the truncated factors $U_r \in \mathbb{C}^{n \times r}$, $\Sigma_r \in \mathbb{R}^{r \times r}$, $V_r \in \mathbb{C}^{m \times r}$. The rank *r* is *the* user-set hyperparameter — pick it from the singular value plot (§3.6).

**Step 2 — Project the dynamics:**
$$
\tilde{A} = U_r^* X_2 V_r \Sigma_r^{-1}, \qquad \tilde{A} \in \mathbb{C}^{r \times r}.
$$
$\tilde{A}$ is *A* projected into the leading-singular-vector basis. It is small (*r* × *r*) and easy to eigendecompose.

**Step 3 — Eigendecomposition:**
$$
\tilde{A}\, W = W \Lambda, \qquad \Lambda = \mathrm{diag}(\lambda_1, \dots, \lambda_r).
$$

**Step 4 — Reconstruct the high-dimensional DMD modes (exact form):**
$$
\Phi = X_2 V_r \Sigma_r^{-1} W \in \mathbb{C}^{n \times r}.
$$
Each column $\boldsymbol\phi_k$ is a **DMD mode** — a coherent spatial structure. The "exact" qualifier (vs. "projected" $\Phi = U_r W$) means $\Phi$ lies in the column space of *X*₂ rather than *X*₁. PyDMD's `DMD(exact=True)` selects this form.

**Step 5 — Continuous-time eigenvalues (frequencies):**
Each discrete eigenvalue $\lambda_k$ corresponds to a continuous-time eigenvalue
$$
\omega_k = \frac{\ln \lambda_k}{\Delta t}.
$$
The **frequency** of mode *k* (in Hz, or whatever 1/Δ*t* units you're using) is
$$
f_k = \frac{\mathrm{Im}(\omega_k)}{2\pi} = \frac{\mathrm{Im}(\ln \lambda_k)}{2\pi\, \Delta t},
$$
and its **growth rate** is $\mathrm{Re}(\omega_k)$. Pure oscillation ⇔ $|\lambda_k| = 1$ ⇔ eigenvalue sits on the unit circle.

**Step 6 — Amplitudes:**
Find the coefficients $b_k$ that best represent the initial snapshot:
$$
\mathbf{x}_0 \approx \Phi\, \mathbf{b}, \qquad \mathbf{b} = \Phi^\dagger \mathbf{x}_0.
$$
This is the *standard* amplitude formula and is what PyDMD computes when `opt=False`. With **`opt=True`** (the default we recommend in §9.2), PyDMD instead solves a least-squares problem over **all** snapshots — minimizing reconstruction error globally rather than only matching the first frame. The result is more accurate on average, especially when the first snapshot is non-typical (e.g., a transient). Equation-wise, the global formulation is $\min_{\mathbf{b}} \|\,X - \Phi\,\mathrm{diag}(\mathbf{b})\,T\,\|_F^2$, where $T_{kj} = \lambda_k^{j}$. Cite this difference in the report.

**Step 7 — Reconstruction:**
$$
\mathbf{x}(t_j) \approx \sum_{k=1}^{r} \boldsymbol\phi_k \exp(\omega_k t_j)\, b_k = \Phi \exp(\Omega t_j) \mathbf{b}.
$$
In matrix form, $X_{\mathrm{DMD}} = \Phi\, T$, where the *k*-th row of *T* is $b_k\, e^{\omega_k t_j}$ for $j = 0, \dots, m$.

### 3.3 What each ingredient means physically
| Symbol | Math object | Physical meaning |
|---|---|---|
| $\boldsymbol\phi_k$ | column of $\Phi$ | coherent spatial structure (vortex pair, mean flow) |
| $\lambda_k$ | eigenvalue of $\tilde{A}$ | one-step growth/oscillation of mode *k* |
| $\omega_k = \ln\lambda_k/\Delta t$ | continuous-time eigenvalue | growth rate (real) + angular frequency (imag) |
| $f_k = \mathrm{Im}(\omega_k)/(2\pi)$ | frequency | how often the mode oscillates per unit time |
| $b_k$ | amplitude | how strongly the mode is excited by the initial state |

### 3.4 Conjugate pairs
For a real-valued field (vorticity is real), every complex eigenvalue comes with its conjugate, and the corresponding modes are conjugate as well. So if you see eigenvalues $\lambda$ and $\bar\lambda$, treat them as a single physical "mode" and add their contributions to get a real-valued spatial pattern.

### 3.5 DMD vs. POD in one sentence
**POD** ranks modes by how much *energy* they explain (ranks by variance, gives orthogonal modes, no time information). **DMD** ranks modes by *frequency content* (gives non-orthogonal modes, each with a single frequency and growth rate). For periodic flows like ours, both pick out the same coherent structures — but DMD also tells you what frequency each one oscillates at, which POD cannot.

### 3.6 How to choose the rank *r*
1. Plot the singular values $\sigma_1 \ge \sigma_2 \ge \dots$ on a log scale.
2. Look for an **elbow** — a sudden drop after which $\sigma_k$ becomes essentially noise.
3. For Re = 100 cylinder, the elbow is around *r* = 20–25. PyDMD's BOPDMD tutorial uses *r* = 11. Any *r* in [11, 25] gives sensible results.
4. Always include an odd plus the mean — for the cylinder you typically take 1 (mean) + several conjugate pairs, so an odd *r* is natural.

---

## 4. The three papers

### 4.1 Sharma & Bhardwaj (2023) — tandem cylinders with FSI
- *J. Fluid Mech.* **976**, A22.
- Re = 100 (single value). Two coupled cylinders, transverse vibration, sharp-interface immersed boundary method.
- **Uses DMD** on the vorticity field to interpret different vortex-shedding regimes (initial branch, mode-mixed, lock-in, galloping). Cites Schmid 2010 and Jovanović sparsity-promoting DMD.
- No public dataset. Methodology is heavy (FSI + IBM) — beyond an undergrad first project, but **read the DMD section** to see how DMD is used to *interpret* a complex flow.
- The third uploaded PDF (`urn_cambridge…sup011.pdf`) is this paper's supplementary methods file (grid independence, IBM verification). Contains no data.

### 4.2 Thompson, Radi, Rao, Sheridan & Hourigan (2014) — elliptical cylinders
- *J. Fluid Mech.* **751**, 570–600.
- Re ≤ 200 (well within our regime). Aspect ratios *Ar* = 0.0 (flat plate) to 1.0 (circular cylinder).
- **Uses DMD** with a clean, beginner-readable Section 2.4 — this is the best pedagogical DMD writeup of the three.
- The *Ar* = 1.0 limit *is* our project case.
- **Use this paper as the primary anchor**: cite the DMD section in your method, and (as a stretch goal) reproduce one elliptical case.

### 4.3 What to take from each in one sentence
- **Sharma & Bhardwaj** → motivation paragraph: "DMD is used in current research on FSI to disentangle multi-regime wake dynamics."
- **Thompson et al.** → method section: "We follow the DMD formulation of Schmid (2010) as presented in Thompson et al. (2014, §2.4)."
- **Supplementary PDF** → don't cite; it's just a methods appendix.

---

## 5. The dataset

### 5.1 What & where
- **Source:** Kutz, Brunton, Brunton & Proctor (2016), *Dynamic Mode Decomposition: Data-Driven Modeling of Complex Systems*, SIAM. Companion site **dmdbook.com**.
- **Direct URL:** `http://dmdbook.com/DATA.zip` (verified live April 2026; 437 504 893 bytes ≈ 417 MB; HTTP 200, last modified 2017-01-06; mirror also at `http://dmdbook.com/CODE.zip`).
- **Inside the zip:** several `.mat` files. The relevant one is **`CYLINDER_ALL.mat`**, a MATLAB v5 file containing:
  - `VORTALL` — vorticity, shape `(89351, 151)` = `(449·199, 151)`, already flattened.
  - `UALL`, `VALL` — velocity components, same shape.
  - `nx = 449`, `ny = 199`.
- **Snapshot spacing:** Δ*t* = 0.02 in non-dimensional units (cylinder diameter *D* = 1, free-stream *U* = 1).
- **Reynolds number:** Re = 100. Computed from a 2D incompressible Navier–Stokes simulation by Kutz et al.
- **Reshape order:** the flat columns reshape to `(ny, nx) = (199, 449)` with **Fortran order** (`order='F'`) — this matches MATLAB's column-major layout. Always pass `order='F'` to NumPy's `reshape` when un-flattening.

### 5.2 The low-resolution variant (faster iteration)
PyDMD's official BOPDMD tutorial uses **`CYLINDER_ALL_LOW_RES.mat`**, with `(nx, ny) = (149, 66)` and the same 151 snapshots — useful for fast prototyping. It ships in the same `DATA.zip` (or you can downsample yourself with `scipy.ndimage.zoom`).

### 5.2a Verification & fallback if dmdbook.com is unreachable
After downloading, record the file size and a hash so you can verify a re-download later:
```bash
ls -l data/DATA.zip            # expect 437,504,893 bytes (~417 MB; zip contains many SIAM-book datasets, not just cylinder)
sha256sum data/DATA.zip > data/DATA.zip.sha256   # save for reproducibility
ls -l data/CYLINDER_ALL.mat    # expect ~108 MB after unzip
```
**If `dmdbook.com` is unreachable**, fall back in this order:
1. The Jupyter port that bundles the same arrays: `git clone https://github.com/florisvb/DMDbookJupyter`
2. Xinyu Chen's cleaned CSV mirror (~121 MB) linked in §16.
3. Ask your advisor — most groups have the file on a shared drive.

### 5.3 Reshape order — the one detail that breaks every newcomer
The `VORTALL` matrix is stored column-major (MATLAB convention). To reshape a single column back to a 2D field, **always** use:
```python
field2d = X[:, j].reshape(ny, nx, order="F")    # ny=199, nx=449
```
This matches PyDMD's own `plot_summary(..., snapshots_shape=(ny, nx), order="F")` in the official cylinder tutorial. Using `order='C'` (NumPy default) will produce a scrambled image — not just transposed, **scrambled** — and you will think DMD failed when it didn't.

### 5.3 Sanity check on the size
- 89 351 × 151 doubles = 13.5 M doubles ≈ **108 MB** in float64 (single channel: vorticity).
- All three channels (u, v, ω) ≈ 324 MB. Comfortable on any remote node.

---

## 6. Remote SSH workflow & environment setup

### 6.1 Recommended local–remote split
| Task | Where |
|------|-------|
| Editing notebooks/scripts | Local laptop (VS Code Remote-SSH, or vim) |
| Running PyDMD, generating figures | Remote server |
| Storing the 417 MB dataset | Remote server |
| Reading/sharing the report | Local laptop |

### 6.2 Get onto the server [LOCAL → ON-SERVER]
First time, set up an SSH key and a config entry on your **laptop** so you don't type the password every time:
```bash
# Generate key if you don't have one
ssh-keygen -t ed25519 -C "kush@adalat.ai"            # accept defaults
ssh-copy-id <user>@<host>                            # uploads ~/.ssh/id_ed25519.pub

# Make a friendly alias in ~/.ssh/config
cat >> ~/.ssh/config <<'EOF'
Host cylinder
    HostName <host>
    User <user>
    IdentityFile ~/.ssh/id_ed25519
    ServerAliveInterval 60
    # If you must hop through a bastion:
    # ProxyJump <bastion-user>@<bastion-host>
EOF

# 2FA / OTP server? Either install `oathtool` and pipe, or just type the code each session.
```
Then connect:
```bash
ssh cylinder                         # uses the alias above
cd ~                                 # or whatever scratch dir you have quota in
mkdir -p projects/cylinder-dmd && cd projects/cylinder-dmd
```

### 6.3 Create an isolated Python environment
Pick one of the two — don't mix.

**Option A — `venv` (lightest):**
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

**Option B — `conda` / `micromamba`:**
```bash
conda create -n cylinder-dmd python=3.11 -y
conda activate cylinder-dmd
```

### 6.4 Install dependencies
Create `requirements.txt`:
```
numpy>=1.26
scipy>=1.11
matplotlib>=3.8
pydmd>=2024.1
jupyterlab>=4.0
ipykernel
imageio>=2.34          # for animations
ffmpeg-python>=0.2     # for mp4 export
```
Then:
```bash
pip install -r requirements.txt
python -c "import pydmd; print(pydmd.__version__)"   # verify
```

### 6.5 Download the dataset to the server [ON-SERVER]
```bash
mkdir -p data && cd data
wget -c http://dmdbook.com/DATA.zip
unzip DATA.zip
ls -lh CYLINDER_ALL.mat               # ~108 MB expected
cd ..
```
If `wget` is missing on the cluster, `curl -OL http://dmdbook.com/DATA.zip` works equally well. If the download is interrupted, `wget -c` resumes it.

### 6.6 Reach Jupyter from your laptop
The clean way: **VS Code Remote-SSH**. Install the extension, open the folder over SSH, and Jupyter notebooks just work natively.

The classic way: **port-forwarded Jupyter**.
```bash
# On the server:
jupyter lab --no-browser --port=8889
# On your laptop, in another terminal:
ssh -L 8888:localhost:8889 <user>@<host>
# Then open http://localhost:8888 in your browser.
```

### 6.7 Avoiding "I lost my session" pain
Use **tmux** or **screen** so disconnecting from SSH doesn't kill your work:
```bash
tmux new -s cylinder
# ... run things ...
# detach: Ctrl-B then D
# later: tmux attach -t cylinder
```

### 6.8 Git from day 1
```bash
git init
echo 'data/'         >  .gitignore
echo '.venv/'        >> .gitignore
echo '__pycache__/'  >> .gitignore
echo '*.ipynb_checkpoints/' >> .gitignore
git add .gitignore requirements.txt
git commit -m "Initial commit"
```
Push to a private GitHub repo. Commit at the end of every working session.

---

## 7. Repository layout

```
cylinder-dmd/
├── README.md                  # how to reproduce; one paragraph + the wget command
├── LICENSE                    # MIT is the safe default for an undergrad academic repo
├── requirements.txt
├── .gitignore
├── data/                      # gitignored
│   └── CYLINDER_ALL.mat
├── src/
│   ├── __init__.py
│   ├── load.py                # load_cylinder() -> (X, dt, ny, nx)
│   ├── plotting.py            # plot_modes, plot_eigs, plot_singular_values, animate
│   ├── dmd_runner.py          # thin wrapper around pydmd.DMD
│   └── pod.py                 # plain-numpy POD for comparison
├── notebooks/
│   ├── 01_explore_data.ipynb
│   ├── 02_dmd_basic.ipynb
│   ├── 03_reconstruction.ipynb
│   ├── 04_pod_comparison.ipynb
│   └── 05_stretch_bopdmd.ipynb         # optional
├── figures/                   # all .png/.pdf saved here
├── report/
│   ├── report.tex             # or report.md if you prefer pandoc
│   └── refs.bib
└── tests/
    └── test_strouhal.py       # automated check that DMD recovers St ≈ 0.166
```

The `tests/` folder is non-negotiable. A single test that asserts the dominant non-zero frequency is within 5% of the expected Strouhal frequency catches 90% of pipeline bugs.

---

## 8. Week-by-week execution plan

Each week ends with a **deliverable** and a **commit**.

### Week 1 — Foundations and data
**Goals:** install everything, watch one DMD lecture, get vorticity data plotted on the server.

| Day | Task |
|---|---|
| 1 | Set up SSH, env, repo (§6). Push first commit. |
| 1 | Watch Brunton's *DMD Overview* video (~30 min). |
| 2 | Download `DATA.zip` to the server. Inspect `CYLINDER_ALL.mat` keys. |
| 2 | Read Section 2.4 of Thompson et al. (2014). |
| 3 | Notebook `01_explore_data.ipynb`: load, plot snapshot 0, 25, 50, 75, 100, 150 as a 2×3 vorticity panel. Confirm vortex shedding is visible. |
| 4 | Reproduce **PyDMD Tutorial 1** on the synthetic 1D wave example. Just to learn the API. |
| 5 | Read §3 of *this document* carefully. Re-derive the algorithm on paper. |
| 6 | Buffer day. |
| 7 | **Deliverable:** `01_explore_data.ipynb` clean and committed; you can confidently say "yes, those are vortices and they're being shed at ~constant frequency". |

### Week 2 — DMD on the cylinder
**Goals:** run exact DMD, plot eigenvalues + modes, confirm Strouhal ≈ 0.16.

| Day | Task |
|---|---|
| 8 | `src/load.py`: write `load_cylinder()` returning `(X, dt, ny, nx)` from the `.mat`. |
| 8 | `src/dmd_runner.py`: thin wrapper that takes (X, rank, dt) and returns a fitted `DMD`. |
| 9 | `02_dmd_basic.ipynb`: SVD plot, pick rank, fit `DMD(svd_rank=21, exact=True)`. |
| 10 | Set `dmd.original_time['dt'] = 0.02` so frequencies come out in physical units. Print `dmd.frequency`. |
| 10 | **Strouhal check:** dominant non-zero frequency × 1 (since *D* = *U* = 1) should be ≈ **0.166**. |
| 11 | Plot eigenvalues on the unit circle (§10.2). Confirm they sit on or extremely near the unit circle. |
| 12 | Plot the first six DMD modes as 199×449 vorticity fields (§10.3). |
| 13 | Plot the mode-amplitude spectrum: |b_k| vs. f_k (§10.4). |
| 14 | **Deliverable:** `02_dmd_basic.ipynb` + `tests/test_strouhal.py` passing. |

### Week 3 — Reconstruction and comparison
**Goals:** show DMD captures the dynamics with a tiny number of modes; compare to POD.

| Day | Task |
|---|---|
| 15 | `03_reconstruction.ipynb`: compute `dmd.reconstructed_data`, reshape, animate. |
| 15 | Side-by-side animation: true vs. DMD reconstruction (§10.5). |
| 16 | Sweep `svd_rank ∈ {1, 3, 5, 7, 11, 21, 41, 81}`. Plot relative Frobenius reconstruction error vs. rank. |
| 17 | `src/pod.py`: implement POD with `np.linalg.svd`. Plot the first six POD modes. |
| 18 | One side-by-side figure: POD mode 2 vs. real-part of DMD mode 2. They will look very similar — note this in the report. |
| 19 | Discussion paragraph in your notes: when is each better, and why? |
| 20 | Buffer day. |
| 21 | **Deliverable:** `03_reconstruction.ipynb` and `04_pod_comparison.ipynb`. All five key figures saved to `figures/` as 300 dpi PNG **and** PDF. |

### Week 4 — Writing & polish
**Goals:** report, slides, final repo cleanup.

| Day | Task |
|---|---|
| 22 | Draft introduction + background (§13 sections 1–2). |
| 23 | Method + dataset (§13 sections 3–4). |
| 24 | Results (§13 section 5) — paste in figures, write captions first, then prose. |
| 25 | Discussion + conclusion + references (§13 sections 6–8). |
| 26 | Build slide deck (§14). |
| 27 | Practice the talk twice out loud. Time it. |
| 28 | **Final deliverables:** report PDF, slides PDF, GitHub repo public link. |

---

## 9. Key code snippets

These are the snippets that contain the actual *DMD work*. Everything else is glue you can write yourself once these run.

### 9.1 Load the data — `src/load.py`
```python
import numpy as np
import scipy.io as sio
from pathlib import Path

def load_cylinder(path: str | Path = "data/CYLINDER_ALL.mat",
                  dt: float = 0.02):
    """
    Load Kutz/Brunton cylinder vorticity snapshots.

    Returns
    -------
    X  : (n, m) float64    flattened vorticity snapshots
    dt : float             snapshot spacing (default 0.02, non-dim time)
    ny : int               grid height (199)
    nx : int               grid width  (449)
    """
    mat = sio.loadmat(str(path))
    X  = mat["VORTALL"].astype(np.float64)         # (89351, 151)
    nx = int(np.squeeze(mat["nx"]))
    ny = int(np.squeeze(mat["ny"]))
    assert X.shape == (ny * nx, 151), f"unexpected shape {X.shape}"
    return X, dt, ny, nx
```

### 9.2 Run exact DMD — `src/dmd_runner.py`
```python
from pydmd import DMD

def run_dmd(X, dt, svd_rank=21, exact=True, opt=True):
    """Fit exact DMD and set physical-time metadata."""
    dmd = DMD(svd_rank=svd_rank, exact=exact, opt=opt, sorted_eigs="abs")
    dmd.fit(X)
    # Tell PyDMD the real Δt so .frequency returns physical Hz-equivalent units.
    dmd.original_time["dt"] = dt
    dmd.dmd_time["dt"]      = dt
    dmd.original_time["tend"] = (X.shape[1] - 1) * dt
    dmd.dmd_time["tend"]      = (X.shape[1] - 1) * dt
    return dmd
```
The `sorted_eigs="abs"` flag sorts modes by descending |λ|, which makes the first few entries the most important physically — handy for plotting.

### 9.3 Singular-value diagnostic
```python
import numpy as np
import matplotlib.pyplot as plt

def plot_singular_spectrum(X, max_show=50, save=None):
    s = np.linalg.svd(X[:, :-1], compute_uv=False)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    k = np.arange(1, len(s) + 1)
    ax[0].semilogy(k[:max_show], s[:max_show], "o-")
    ax[0].set(xlabel="index k", ylabel=r"$\sigma_k$",
              title="Singular spectrum (log scale)")
    ax[1].plot(k[:max_show], np.cumsum(s[:max_show]) / s.sum(), "o-")
    ax[1].set(xlabel="index k", ylabel="cumulative energy",
              title="Energy capture", ylim=(0, 1.02))
    fig.tight_layout()
    if save: fig.savefig(save, dpi=300)
    return s
```

### 9.4 Eigenvalues on the unit circle
```python
def plot_eigs_unit_circle(dmd, save=None):
    eigs = dmd.eigs
    theta = np.linspace(0, 2*np.pi, 400)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(np.cos(theta), np.sin(theta), 'k--', lw=1)
    ax.scatter(eigs.real, eigs.imag, s=60,
               c=np.abs(dmd.amplitudes), cmap="viridis", edgecolor='k')
    ax.set_aspect("equal")
    ax.set(xlabel="Re(λ)", ylabel="Im(λ)",
           title="Discrete-time DMD eigenvalues")
    ax.axhline(0, color='gray', lw=0.5); ax.axvline(0, color='gray', lw=0.5)
    if save: fig.savefig(save, dpi=300)
```
**What to look for:** every eigenvalue should sit essentially on the unit circle ($|λ_k| \approx 1$). One eigenvalue at λ = 1 + 0i is the steady mean. The rest come in conjugate pairs above/below the real axis.

### 9.5 Plot DMD modes as flow fields
```python
def plot_modes(dmd, ny, nx, n_modes=6, save=None):
    fig, axes = plt.subplots(2, 3, figsize=(13, 5.5))
    freqs = dmd.frequency
    for k, ax in enumerate(axes.flat[:n_modes]):
        mode = dmd.modes[:, k].real.reshape(ny, nx, order="F")
        vmax = np.max(np.abs(mode))
        ax.imshow(mode, cmap="RdBu_r", origin="lower",
                  vmin=-vmax, vmax=vmax)
        ax.set_title(f"Mode {k}    f = {freqs[k]:+.3f}")
        ax.axis("off")
    fig.tight_layout()
    if save: fig.savefig(save, dpi=300)
```
Use `order='F'` — MATLAB column-major. Without it the modes will look correct in shape but rotated/transposed.

### 9.6 Reconstruction-error sweep
```python
def reconstruction_error(X, dt, ranks):
    errs = []
    for r in ranks:
        dmd = run_dmd(X, dt, svd_rank=r)
        Xr  = dmd.reconstructed_data.real
        err = np.linalg.norm(X - Xr) / np.linalg.norm(X)
        errs.append(err)
    return np.asarray(errs)
```
Plot `errs` against `ranks` on a semilog-y axis. You should see a sharp drop from r=1 → r=3 (mean + first conjugate pair), another drop at r=5–7, then a slow tail.

### 9.7 The Strouhal sanity test — `tests/test_strouhal.py`
```python
import numpy as np
from src.load import load_cylinder
from src.dmd_runner import run_dmd

def test_strouhal_recovered():
    X, dt, ny, nx = load_cylinder()
    dmd = run_dmd(X, dt, svd_rank=21)

    freqs = dmd.frequency                       # signed, in Hz-equivalent
    amps  = np.abs(dmd.amplitudes)

    # Drop the (near-)zero-frequency mean mode.
    keep = np.abs(freqs) > 1e-3
    freqs_nz, amps_nz = freqs[keep], amps[keep]

    # Pick the largest-amplitude oscillating mode — this is the fundamental.
    # (Picking the min |f| is brittle: a near-zero spurious mode can win.)
    k_dom = np.argmax(amps_nz)
    f_dom = abs(freqs_nz[k_dom])

    St = f_dom * 1.0 / 1.0                      # D=1, U=1 in non-dim units
    assert 0.158 < St < 0.174, f"Got St={St:.4f}, expected 0.166 ± 5%"
```
Run with `pytest tests/`. If this fails, your pipeline is broken — fix it before doing anything else.

### 9.8 End-to-end "smoke test" — paste this into a fresh notebook on day 1
Once you have `data/CYLINDER_ALL.mat` on the server and `pydmd` installed, this single script should run cleanly and tell you in ~5 seconds whether your environment is sane. If `St` prints inside the 0.158–0.174 window, you're done with setup.
```python
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from pydmd import DMD

# 1. Load
mat = sio.loadmat("data/CYLINDER_ALL.mat")
X   = mat["VORTALL"].astype(np.float64)         # (89351, 151)
ny, nx = int(np.squeeze(mat["ny"])), int(np.squeeze(mat["nx"]))
dt = 0.02

# 2. Fit
dmd = DMD(svd_rank=21, exact=True, opt=True, sorted_eigs="abs")
dmd.fit(X)
dmd.original_time["dt"] = dt
dmd.dmd_time["dt"]      = dt

# 3. Sanity check: dominant non-zero frequency → Strouhal
freqs = dmd.frequency
amps  = np.abs(dmd.amplitudes)
keep  = np.abs(freqs) > 1e-3
k_dom = np.argmax(amps[keep])
St    = abs(freqs[keep][k_dom])
print(f"Recovered St = {St:.4f}   (target 0.166 ± 5%)")

# 4. Reconstruction error
err = np.linalg.norm(X - dmd.reconstructed_data.real) / np.linalg.norm(X)
print(f"Relative reconstruction error at r=21: {err:.3%}")

# 5. Eyeball the dominant mode
mode = dmd.modes[:, np.argmax(amps[keep])+1].real.reshape(ny, nx, order="F")
plt.figure(figsize=(8, 3))
plt.imshow(mode, cmap="RdBu_r", origin="lower",
           vmin=-np.abs(mode).max(), vmax=np.abs(mode).max())
plt.title(f"Dominant DMD mode, f ≈ {St:.3f}")
plt.axis("off"); plt.tight_layout()
plt.savefig("figures/smoke_test_mode.png", dpi=200)
```
**Expected output (verbatim ranges):**
- `Recovered St = 0.16xx`  (between 0.158 and 0.174)
- `Relative reconstruction error at r=21: ~3%`
- A vorticity field showing two horizontally-arranged red/blue lobes — the von Kármán shedding mode.

If any of these is off, jump straight to §11.

### 9.9 Stretch — BOPDMD comparison
```python
from pydmd import BOPDMD
from pydmd.preprocessing import zero_mean_preprocessing

def run_bopdmd(X, dt, svd_rank=11, num_trials=100):
    t = np.arange(X.shape[1]) * dt
    bop = BOPDMD(svd_rank=svd_rank, num_trials=num_trials, trial_size=0.8,
                 eig_constraints={"imag", "conjugate_pairs"}, seed=1234)
    bop = zero_mean_preprocessing(bop)
    bop.fit(X, t=t)
    return bop
```
BOPDMD constrains eigenvalues to the imaginary axis (pure oscillation, no spurious growth/decay) and uses bootstrap resampling for uncertainty quantification — a very clean upgrade for the report's discussion.

---

## 10. Visualization recipes

### 10.1 Universal style
Add this to `src/plotting.py` once and import everywhere:
```python
import matplotlib as mpl
mpl.rcParams.update({
    "font.family":     "serif",
    "font.size":       11,
    "axes.titlesize":  12,
    "axes.labelsize":  11,
    "figure.dpi":      120,
    "savefig.dpi":     300,
    "savefig.bbox":    "tight",
    "lines.linewidth": 1.5,
})
```

### 10.2 Eigenvalue plot — what good looks like
- Equal aspect ratio (`ax.set_aspect('equal')`) — otherwise the unit circle becomes an ellipse and you'll think eigenvalues are off-circle when they aren't.
- Color points by `|b_k|` (amplitude) so the visually large ones are the dynamically important ones.
- Add a faint dashed unit circle.
- Annotate the dominant frequency near the topmost point: `ax.annotate(f"f = {f:.3f}", ...)`.

### 10.3 Mode plots — what good looks like
- Use a **diverging colormap** (`RdBu_r` or `seismic`) because vorticity is signed.
- Symmetric color limits: `vmin=-vmax, vmax=vmax` where `vmax = np.max(|mode|)`.
- Equal aspect ratio.
- Title: `Mode k, f = ±0.166`. Mention the conjugate pair sign explicitly.
- Hide axes (`ax.axis('off')`) — you're showing flow structure, not coordinates.

### 10.4 Mode-amplitude spectrum — what good looks like
A stem plot of |b_k| vs. f_k. The mean mode at f = 0 will be tallest; then a tall pair at f ≈ ±0.166; smaller pairs at the harmonics ±0.328, ±0.492.
```python
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.stem(dmd.frequency, np.abs(dmd.amplitudes), basefmt=" ")
ax.set(xlabel="frequency f", ylabel="|b_k|",
       title="DMD mode-amplitude spectrum")
ax.grid(True, alpha=0.3)
```

### 10.5 True vs. reconstructed animation
Use `matplotlib.animation.FuncAnimation` — it works across all Matplotlib backends and writes mp4 directly via the bundled ffmpeg writer.
```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

def animate_comparison(X, X_dmd, ny, nx, fname="figures/comparison.mp4",
                       fps=15):
    truth_max = np.abs(X).max()
    rec_max   = np.abs(X_dmd.real).max()
    v = max(truth_max, rec_max)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3))
    ims = []
    for ax, title in zip(axes, ["truth", "DMD r=21"]):
        im = ax.imshow(np.zeros((ny, nx)), cmap="RdBu_r",
                       vmin=-v, vmax=v, origin="lower")
        ax.set_title(title); ax.axis("off")
        ims.append(im)
    fig.tight_layout()

    def update(j):
        ims[0].set_data(X[:, j].reshape(ny, nx, order="F"))
        ims[1].set_data(X_dmd[:, j].real.reshape(ny, nx, order="F"))
        return ims

    anim = FuncAnimation(fig, update, frames=X.shape[1], blit=True)
    anim.save(fname, writer=FFMpegWriter(fps=fps, bitrate=2400))
    plt.close(fig)
```
Drop this in your slide deck — it sells the project in 5 seconds.

### 10.6 Color and accessibility
- Avoid `jet`. Use `viridis` for sequential, `RdBu_r` for diverging.
- All saved figures: 300 dpi, transparent background optional, vector PDF for the report and PNG for the slides.

---

## 11. Troubleshooting & sanity checks

| Symptom | Likely cause | Fix |
|---|---|---|
| `dmd.frequency` returns garbage like 0.5, 1.7, 2.4… | You forgot to set `dmd.original_time['dt']`. PyDMD defaults Δt = 1, so frequencies are off by a factor of 50. | Set `dmd.original_time['dt'] = 0.02` (and `dmd_time['dt']`) right after `dmd.fit(X)`. |
| Modes look "tilted" or transposed | Wrong reshape order. | Use `order='F'` in `.reshape(ny, nx, order='F')`. |
| Eigenvalues sit *inside* the unit circle | Truncated rank too low → spurious damping. | Increase `svd_rank` (try 11, 21, 31 — pick the smallest that gives `|λ| ≈ 1`). |
| Eigenvalues *outside* the unit circle | Numerical noise + low rank. | Same fix; or use BOPDMD with `eig_constraints={"imag"}`. |
| Reconstruction error never drops below ~30% even at high rank | You're including a transient at the start of the snapshots. | Drop the first ~10 snapshots (`X = X[:, 10:]`) so all data is in the saturated periodic regime. |
| Strouhal ≈ 0.32, not 0.16 | You're picking up the second harmonic as the "dominant". | Sort frequencies and take the smallest *non-zero* one (`freqs[freqs > 1e-3].min()`), not the largest amplitude. |
| `MemoryError` during SVD | Storing too many channels at once. | Use only `VORTALL`, in float32 if needed: `X = X.astype(np.float32)`. |
| Plots are blocky/pixelated | Saving at default 100 dpi. | `plt.savefig(..., dpi=300)` and prefer PDF for the report. |
| `pip install pydmd` fails | Old pip or old Python. | Upgrade to Python ≥ 3.10 and `pip install --upgrade pip` first. |

### Quick numeric sanity table
After fitting with `svd_rank=21` you should see roughly:
- `|dmd.eigs|` all within `1 ± 0.01`.
- One real eigenvalue ≈ 1 + 0i (the mean mode).
- A complex-conjugate pair at angle $\theta = 2\pi \cdot \mathrm{St} \cdot \Delta t = 2\pi \cdot 0.166 \cdot 0.02 \approx 0.0209$ rad above/below the real axis (i.e., very close to the +1 point on the unit circle).
- Reconstruction error at r = 21: a few percent. At r = 3: ~30%. At r = 1: ~70%.

The eigenvalue angle is small because Δ*t* is small; geometrically, neighbouring snapshots advance the shedding cycle by only a tiny fraction of a full revolution.

---

## 12. Stretch goals

If Week 3 finishes early, pick one. Don't try all four.

1. **BOPDMD comparison.** Run §9.8. Show that BOPDMD eigenvalues have tighter UQ ellipses and sit cleaner on the imaginary axis. One paragraph + one figure in the report.
2. **Sparsity-promoting DMD (Jovanović et al. 2014).** Use `pydmd.SpDMD`. Sweep the sparsity weight γ; show that the algorithm naturally selects ~3 modes (mean + first pair) for the cylinder. This connects directly to Sharma & Bhardwaj (2023).
3. **Rank-sensitivity heatmap.** A 2D heatmap of reconstruction error vs. (`svd_rank`, # snapshots used). Cheap to produce, looks impressive.
4. **Elliptical-cylinder extension.** Self-simulate Thompson et al.'s *Ar* = 0.5 ellipse case in **FEniCSx** (DFG 2D-3 benchmark variant; ~15 min runtime). Run the same DMD pipeline. Compare modes and Strouhal to *Ar* = 1.0.

---

## 13. Report skeleton

Target length: ~10 pages, 11 pt serif, 1.5-line spacing. Use LaTeX (`article` class is fine) or Pandoc-Markdown → PDF.

1. **Abstract** (150 words). The pipeline + the headline result (St recovered to within X% with r DMD modes; reconstruction error Y% at rank r).
2. **Introduction** (~1 page). Why coherent structures matter (drag, vortex-induced vibration — cite Sharma & Bhardwaj 2023). Why data-driven methods (you may not have a model). What DMD does in one paragraph. Project objectives in three bullet points.
3. **Background** (~1 page). Vortex shedding, Reynolds and Strouhal numbers. One labeled diagram of the von Kármán street.
4. **Method** (~2 pages). The exact-DMD algorithm from §3, with all six equations. Short paragraph on POD as the comparison baseline. Cite Schmid 2010, Tu et al. 2014, Demo et al. 2018, Thompson et al. 2014 §2.4.
5. **Dataset** (~½ page). Source, dimensions, Re, Δt, snapshot count. Cite the SIAM book.
6. **Results** (~3 pages). In this order: snapshot panel (Fig. 1), singular spectrum (Fig. 2), eigenvalues on unit circle (Fig. 3), DMD modes (Fig. 4), mode-amplitude spectrum (Fig. 5), reconstruction error vs. rank (Fig. 6), true vs. reconstructed snapshot at one time (Fig. 7), DMD-vs-POD mode comparison (Fig. 8). Each figure has a 2–3 sentence caption.
7. **Discussion** (~1 page). Strouhal recovered; rank choice trade-off; POD vs. DMD differences; limitations (linearity assumption, fixed sampling rate, transient handling); how the method extends to the elliptical (Thompson et al.) and FIV (Sharma & Bhardwaj) cases.
8. **Conclusion** (~½ page). One paragraph summary + one paragraph future work.
9. **References** (BibTeX file).

### Captions worth the effort
> **Fig. 4.** *First six DMD modes of the cylinder wake at Re = 100, ranked by descending eigenvalue magnitude. Mode 0 is the steady mean. Modes 1–2 form the dominant conjugate pair at the von Kármán shedding frequency f ≈ 0.166. Modes 3–4 form the second harmonic at f ≈ 0.328. Color is vorticity, normalized symmetrically per panel.*

---

## 14. Slide outline (10 slides, ~12 min talk)

1. **Title.** Project name, your name, advisor, date.
2. **The flow** (1 figure: snapshot of vorticity). "This is what we're trying to understand. Vortices shed periodically — but the data is 89 351-dimensional."
3. **Goal of DMD** (1 sentence + 1 schematic). "Find a small linear model that captures the dynamics directly from snapshots."
4. **Algorithm in one slide.** The six-step recipe from §3.2 as a vertical list with the key equations.
5. **Data.** Source, dimensions, Re, snapshot count.
6. **Singular spectrum.** Justifies your rank choice.
7. **Eigenvalues on the unit circle.** "All on the circle → pure oscillation. Pair at ±0.166 → von Kármán."
8. **Modes.** 2×3 panel of the first six modes.
9. **Reconstruction.** Side-by-side animation (10–15 s loop). Plot of error vs. rank. Headline number: "21 DMD modes → ~3% error from 89 351-dimensional data."
10. **Conclusion + future work.** One bullet per: (i) Strouhal recovered ✓, (ii) low-rank reduction ✓, (iii) extends to ellipses (Thompson 2014) and FIV (Sharma 2023).

---

## 15. Grading rubric (use this to self-assess)

| Criterion | Weight | What "excellent" looks like |
|---|---|---|
| Correctness of DMD pipeline | 25% | `tests/test_strouhal.py` passes; eigenvalues on unit circle; modes look like vortex pairs. |
| Clarity of theory section | 20% | Notation consistent with §3 of this plan; every equation is motivated. |
| Quality of visualizations | 20% | All eight figures present, 300 dpi, captioned, diverging colormap for vorticity. |
| Discussion depth | 15% | DMD vs. POD comparison; explicit Strouhal validation; limitations honestly stated. |
| Reproducibility | 10% | Anyone can `git clone`, `pip install -r requirements.txt`, `wget` the data, and re-run. |
| Writing & references | 10% | Cites Schmid, Tu, Demo, Kutz, Thompson; no hand-waving. |

---

## 16. Citations & resources

### Primary literature (cite these)
- Schmid, P. J. (2010). *Dynamic mode decomposition of numerical and experimental data.* J. Fluid Mech. **656**, 5–28.
- Rowley, C. W., Mezić, I., Bagheri, S., Schlatter, P., Henningson, D. S. (2009). *Spectral analysis of nonlinear flows.* J. Fluid Mech. **641**, 115–127.
- Tu, J. H., Rowley, C. W., Luchtenburg, D. M., Brunton, S. L., Kutz, J. N. (2014). *On dynamic mode decomposition: theory and applications.* J. Comp. Dyn. **1**(2), 391–421.
- Williamson, C. H. K. (1996). *Vortex dynamics in the cylinder wake.* Annu. Rev. Fluid Mech. **28**, 477–539. (Source for St ≈ 0.166 at Re = 100.)
- Jovanović, M. R., Schmid, P. J., Nichols, J. W. (2014). *Sparsity-promoting dynamic mode decomposition.* Phys. Fluids **26**, 024103.
- Kutz, J. N., Brunton, S. L., Brunton, B. W., Proctor, J. L. (2016). *Dynamic Mode Decomposition: Data-Driven Modeling of Complex Systems.* SIAM. (`http://dmdbook.com`)
- Demo, N., Tezzele, M., Rozza, G. (2018). *PyDMD: Python Dynamic Mode Decomposition.* JOSS **3**(22), 530.
- Ichinaga, S. M. et al. (2024). *PyDMD: A Python package for robust dynamic mode decomposition.* arXiv:2402.07463.
- Thompson, M. C., Radi, A., Rao, A., Sheridan, J., Hourigan, K. (2014). *Low-Reynolds-number wakes of elliptical cylinders.* J. Fluid Mech. **751**, 570–600.
- Sharma, G., Bhardwaj, R. (2023). *Flow-induced vibrations of elastically coupled tandem cylinders.* J. Fluid Mech. **976**, A22.

### Starter `report/refs.bib`
```bibtex
@article{schmid2010,
  author={Schmid, Peter J.}, title={Dynamic mode decomposition of numerical and experimental data},
  journal={J. Fluid Mech.}, volume={656}, pages={5--28}, year={2010}}
@article{tu2014,
  author={Tu, J. H. and Rowley, C. W. and Luchtenburg, D. M. and Brunton, S. L. and Kutz, J. N.},
  title={On dynamic mode decomposition: theory and applications},
  journal={J. Comp. Dyn.}, volume={1}, number={2}, pages={391--421}, year={2014}}
@article{williamson1996,
  author={Williamson, C. H. K.}, title={Vortex dynamics in the cylinder wake},
  journal={Annu. Rev. Fluid Mech.}, volume={28}, pages={477--539}, year={1996}}
@book{kutz2016,
  author={Kutz, J. N. and Brunton, S. L. and Brunton, B. W. and Proctor, J. L.},
  title={Dynamic Mode Decomposition: Data-Driven Modeling of Complex Systems},
  publisher={SIAM}, year={2016}}
@article{demo2018pydmd,
  author={Demo, Nicola and Tezzele, Marco and Rozza, Gianluigi},
  title={PyDMD: Python Dynamic Mode Decomposition},
  journal={Journal of Open Source Software}, volume={3}, number={22}, pages={530}, year={2018}}
@article{thompson2014elliptical,
  author={Thompson, M. C. and Radi, A. and Rao, A. and Sheridan, J. and Hourigan, K.},
  title={Low-Reynolds-number wakes of elliptical cylinders: from the circular cylinder to the normal flat plate},
  journal={J. Fluid Mech.}, volume={751}, pages={570--600}, year={2014}}
@article{sharma2023fiv,
  author={Sharma, Gaurav and Bhardwaj, Rajneesh},
  title={Flow-induced vibrations of elastically coupled tandem cylinders},
  journal={J. Fluid Mech.}, volume={976}, pages={A22}, year={2023}}
```

### Tools and tutorials
- PyDMD repo: https://github.com/PyDMD/PyDMD
- PyDMD docs: https://pydmd.github.io/PyDMD/
- PyDMD tutorial 1 (synthetic): https://pydmd.github.io/PyDMD/tutorial1dmd.html
- PyDMD official cylinder tutorial (BOPDMD): https://github.com/PyDMD/PyDMD/blob/master/tutorials/user-manual1/user-manual-bopdmd.py
- Dataset (verified live April 2026): http://dmdbook.com/DATA.zip
- Companion code zip: http://dmdbook.com/CODE.zip
- DMD book companion site: http://dmdbook.com/

### Lectures & long-form
- Steve Brunton's "Eigensteve" channel: https://www.youtube.com/c/eigensteve
- DMD overview lecture: https://www.youtube.com/watch?v=sQvrK8AGCAo
- End-to-end notebook walkthrough: https://predictivesciencelab.github.io/advanced-scientific-machine-learning/dynamics/05_dmd_example.html

### Backup data sources
- DMDbook Jupyter port: https://github.com/florisvb/DMDbookJupyter
- Cleaned CSV mirror (Xinyu Chen): https://medium.com/@xinyu.chen/reproducing-dynamic-mode-decomposition-on-fluid-flow-data-in-python-94b8d7e1f203
- Self-simulate fallback (FEniCSx DFG 2D-3): https://jsdokken.com/dolfinx-tutorial/chapter2/ns_code2.html

---

*End of plan. If anything in §3 (theory) or §9 (snippets) breaks during execution, fix it in this document so the next person inherits a working plan.*