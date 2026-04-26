from pydmd import DMD


def run_dmd(X, dt, svd_rank=21, exact=True, opt=True):
    dmd = DMD(svd_rank=svd_rank, exact=exact, opt=opt, sorted_eigs="abs")
    dmd.fit(X)
    dmd.original_time["t0"] = 0.0
    dmd.original_time["dt"] = dt
    dmd.original_time["tend"] = (X.shape[1] - 1) * dt
    dmd.dmd_time["t0"] = 0.0
    dmd.dmd_time["dt"] = dt
    dmd.dmd_time["tend"] = (X.shape[1] - 1) * dt
    return dmd
