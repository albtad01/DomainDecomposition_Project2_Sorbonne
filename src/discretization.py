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

def mesh(nx, ny, Lx, Ly):
    i = np.arange(0, nx)[na, :] * np.ones((ny, 1), np.int64)
    j = np.arange(0, ny)[:, na] * np.ones((1, nx), np.int64)
    p = np.zeros((2, ny - 1, nx - 1, 3), np.int64)
    q = i + nx * j
    p[:, :, :, 0] = q[:-1, :-1]
    p[0, :, :, 1] = q[1:, 1:]
    p[0, :, :, 2] = q[1:, :-1]
    p[1, :, :, 1] = q[:-1, 1:]
    p[1, :, :, 2] = q[1:, 1:]
    v = np.concatenate(
        ((Lx / (nx - 1) * i)[:, :, na], (Ly / (ny - 1) * j)[:, :, na]),
        axis=2,
    )
    vtx = np.reshape(v, (nx * ny, 2))
    elt = np.reshape(p, (2 * (nx - 1) * (ny - 1), 3))
    return vtx, elt

def boundary(nx, ny):
    bottom = np.hstack((np.arange(0, nx - 1, 1)[:, na], np.arange(1, nx, 1)[:, na]))
    top = np.hstack(
        (
            np.arange(nx * (ny - 1), nx * ny - 1, 1)[:, na],
            np.arange(nx * (ny - 1) + 1, nx * ny, 1)[:, na],
        )
    )
    left = np.hstack(
        (np.arange(0, nx * (ny - 1), nx)[:, na], np.arange(nx, nx * ny, nx)[:, na])
    )
    right = np.hstack(
        (
            np.arange(nx - 1, nx * (ny - 1), nx)[:, na],
            np.arange(2 * nx - 1, nx * ny, nx)[:, na],
        )
    )
    return np.vstack((bottom, top, left, right))

def get_area(vtx, elt):
    d = np.size(elt, 1)
    if d == 2:
        e = vtx[elt[:, 1], :] - vtx[elt[:, 0], :]
        areas = la.norm(e, axis=1)
    else:
        e1 = vtx[elt[:, 1], :] - vtx[elt[:, 0], :]
        e2 = vtx[elt[:, 2], :] - vtx[elt[:, 0], :]
        areas = 0.5 * np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
    return areas

def mass(vtx, elt):
    nv = np.size(vtx, 0)
    d = np.size(elt, 1)
    areas = get_area(vtx, elt)
    M = csr_matrix((nv, nv), dtype=np.float64)
    for j in range(d):
        for k in range(d):
            row = elt[:, j]
            col = elt[:, k]
            val = areas * (1 + (j == k)) / (d * (d + 1))
            M += csr_matrix((val, (row, col)), shape=(nv, nv))
    return M

def stiffness(vtx, elt):
    nv = np.size(vtx, 0)
    d = np.size(elt, 1)
    areas = get_area(vtx, elt)
    ne, d = np.shape(elt)
    E = np.empty((ne, d, d - 1), dtype=np.float64)
    E[:, 0, :] = 0.5 * (vtx[elt[:, 1], 0:2] - vtx[elt[:, 2], 0:2])
    E[:, 1, :] = 0.5 * (vtx[elt[:, 2], 0:2] - vtx[elt[:, 0], 0:2])
    E[:, 2, :] = 0.5 * (vtx[elt[:, 0], 0:2] - vtx[elt[:, 1], 0:2])
    K = csr_matrix((nv, nv), dtype=np.float64)
    for j in range(d):
        for k in range(d):
            row = elt[:, j]
            col = elt[:, k]
            val = np.sum(E[:, j, :] * E[:, k, :], axis=1) / areas
            K += csr_matrix((val, (row, col)), shape=(nv, nv))
    return K

def point_source(sp, k):
    def ps(x):
        v = np.zeros(np.size(x, 0), float)
        for s in sp:
            v += s[2] * np.exp(
                -10 * (k / (2.0 * pi)) ** 2 * la.norm(x - s[na, 0:2], axis=1) ** 2
            )
        return v

    return ps

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