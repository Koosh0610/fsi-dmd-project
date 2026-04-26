import numpy as np
import scipy.io as sio
from pathlib import Path


def load_cylinder(path: str | Path = "data/CYLINDER_ALL.mat",
                  dt: float = 0.2):
    mat = sio.loadmat(str(path))
    X = mat["VORTALL"].astype(np.float64)
    # File stores nx=199, ny=449 but physical grid is 449 wide x 199 tall
    ny = int(np.squeeze(mat["nx"]))  # 199 cross-stream points
    nx = int(np.squeeze(mat["ny"]))  # 449 streamwise points
    assert X.shape == (ny * nx, 151), f"unexpected shape {X.shape}"
    return X, dt, ny, nx
