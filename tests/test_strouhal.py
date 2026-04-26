import numpy as np
from src.load import load_cylinder
from src.dmd_runner import run_dmd


def test_strouhal_recovered():
    X, dt, ny, nx = load_cylinder()
    dmd = run_dmd(X, dt, svd_rank=21)

    freqs = dmd.frequency
    amps = np.abs(dmd.amplitudes)

    keep = np.abs(freqs) > 1e-3
    freqs_nz, amps_nz = freqs[keep], amps[keep]

    k_dom = np.argmax(amps_nz)
    f_dom = abs(freqs_nz[k_dom])

    St = f_dom * 1.0 / 1.0  # D=1, U=1
    assert 0.158 < St < 0.174, f"Got St={St:.4f}, expected 0.166 +/- 5%"
