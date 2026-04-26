#!/usr/bin/env python3
"""
Run all figure-generation scripts in order.
Usage: python scripts/run_all.py       (from project root)
"""
import subprocess, sys, time

SCRIPTS = [
    "scripts/01_data_exploration.py",
    "scripts/02_svd_rank.py",
    "scripts/03_dmd_fitting.py",
    "scripts/04_reconstruction.py",
    "scripts/05_spectral_centreline.py",
    "scripts/06_pod_comparison.py",
    "scripts/07_sensitivity_bopdmd.py",
    "scripts/08_animation.py",
]

t0 = time.time()
for script in SCRIPTS:
    print(f"\n{'='*60}")
    print(f"  Running {script}")
    print(f"{'='*60}")
    result = subprocess.run([sys.executable, script], cwd=".")
    if result.returncode != 0:
        print(f"  FAILED: {script}")
        sys.exit(1)

elapsed = time.time() - t0
print(f"\n{'='*60}")
print(f"  All scripts complete in {elapsed:.0f}s")
print(f"  Figures saved to figures/fig01_*.png ... fig28_*.mp4")
print(f"{'='*60}")
