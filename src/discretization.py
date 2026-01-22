#! /usr/bin/python3
from math import pi
import os
import csv
import time

import numpy as np
na = np.newaxis
import numpy.linalg as la
import scipy.sparse.linalg as spla
from scipy.sparse import csr_matrix, csc_matrix
from fem_core import mesh, boundary, mass, stiffness, point_source

def run_one(m, Lx, Ly, k, sp, rtol, gmres_maxiter, restart):
    """
    Runs the baseline FE assembly + (optional) direct solve + GMRES,
    and returns metrics used for the discretization refinement study.
    """
    nx = int(1 + Lx * m)
    ny = int(1 + Ly * m)

    vtx, elt = mesh(nx, ny, Lx, Ly)
    belt = boundary(nx, ny)

    M = mass(vtx, elt)
    Mb = mass(vtx, belt)
    K = stiffness(vtx, elt)

    A = K - k**2 * M - 1j * k * Mb
    b = M @ point_source(sp, k)(vtx)

    t0 = time.time()
    x = spla.spsolve(A, b)
    t_direct = time.time() - t0

    residuals = []
    def callback(rn):
        residuals.append(rn)

    t0 = time.time()
    y, info = spla.gmres(
        A, b,
        rtol=rtol,
        maxiter=gmres_maxiter,
        restart=restart,
        callback=callback,
        callback_type="pr_norm",
    )
    t_gmres = time.time() - t0

    iters = len(residuals)
    err = la.norm(y - x)

    hx = Lx / (nx - 1)
    hy = Ly / (ny - 1)
    h = max(hx, hy)
    ndof = nx * ny

    return {
        "m": m,
        "nx": nx,
        "ny": ny,
        "ndof": ndof,
        "hx": hx,
        "hy": hy,
        "h": h,
        "iters": iters,
        "gmres_info": info,
        "rtol": rtol,
        "restart": restart,
        "gmres_maxiter": gmres_maxiter,
        "t_direct_s": t_direct,
        "t_gmres_s": t_gmres,
        "direct_vs_gmres_err": err,
    }

def main():
    Lx = 1
    Ly = 2
    k = 16
    ns = 8

    # Reproducibility
    seed = 0
    np.random.seed(seed)
    sp = [np.random.rand(3) * [Lx, Ly, 50.0] for _ in np.arange(ns)]

    # GMRES parameters
    rtol = 1e-12
    restart = 20
    gmres_maxiter = 20000

    # Mesh refinement values
    ms = [32, 48, 64, 80, 96, 112, 128]

    # Output CSV
    outdir = os.path.join("results", "discretization")
    os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(outdir, "mesh_refinement.csv")

    write_header = not os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow([
                "seed","ns","k",
                "m","nx","ny","ndof",
                "hx","hy","h",
                "rtol","restart","gmres_maxiter",
                "iters","gmres_info",
                "t_direct_s","t_gmres_s",
                "direct_vs_gmres_err"
            ])

        for m in ms:
            print(f"\nRunning m={m} (nx={int(1+Lx*m)}, ny={int(1+Ly*m)}) ---")
            out = run_one(m, Lx, Ly, k, sp, rtol, gmres_maxiter, restart)

            print("Total number of GMRES iterations = ", out["iters"])
            print("Direct vs GMRES error            = ", out["direct_vs_gmres_err"])

            w.writerow([
                seed, ns, k,
                out["m"], out["nx"], out["ny"], out["ndof"],
                out["hx"], out["hy"], out["h"],
                out["rtol"], out["restart"], out["gmres_maxiter"],
                out["iters"], out["gmres_info"],
                out["t_direct_s"], out["t_gmres_s"],
                out["direct_vs_gmres_err"]
            ])

    print(f"\nWrote/updated CSV: {csv_path}")

if __name__ == "__main__":
    main()