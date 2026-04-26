import numpy as np


def compute_pod(X, n_modes=6):
    X_mean = X.mean(axis=1, keepdims=True)
    X_fluct = X - X_mean
    U, S, Vt = np.linalg.svd(X_fluct, full_matrices=False)
    return U[:, :n_modes], S, Vt[:n_modes, :], X_mean
