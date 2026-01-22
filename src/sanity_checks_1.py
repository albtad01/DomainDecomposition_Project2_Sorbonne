#! /usr/bin/python3
from __future__ import annotations

import numpy as np
import numpy.linalg as la
import scipy.sparse as sp

from dd_mesh import local_mesh, local_boundary
from dd_restrictions import Rj_matrix, Bj_matrix, Cj_matrix
from dd_local_problems import Aj_matrix, Tj_matrix, Sj_factorization, bj_vector
from dd_operators import build_subdomains, S_operator, Pi_operator, g_vector


# ----------------------------
# tiny test harness utilities
# ----------------------------
def _ok(msg: str):
    print(f"[ OK ] {msg}")

def _fail(msg: str):
    raise AssertionError(f"[FAIL] {msg}")

def assert_true(cond: bool, msg: str):
    if not cond:
        _fail(msg)
    _ok(msg)

def assert_allclose(a, b, tol=1e-12, msg="allclose"):
    a = np.asarray(a)
    b = np.asarray(b)
    err = la.norm(a - b) / max(1.0, la.norm(b))
    if err > tol:
        _fail(f"{msg} (relerr={err:.3e} > {tol:.3e})")
    _ok(f"{msg} (relerr={err:.3e})")

def assert_shape(A, shape, msg="shape"):
    if tuple(A.shape) != tuple(shape):
        _fail(f"{msg}: got {A.shape}, expected {shape}")
    _ok(f"{msg}: {shape}")


# ----------------------------
# helpers coherent with your conventions
# ----------------------------
def ny_loc_from(Ny: int, J: int) -> int:
    return (Ny - 1) // J + 1

def nsigma_for(Nx: int, j: int, J: int) -> int:
    if J == 1:
        return 0
    if j == 0 or j == J - 1:
        return Nx
    return 2 * Nx

def block_offsets(Nx: int, J: int):
    sizes = [nsigma_for(Nx, j, J) for j in range(J)]
    offsets = [0]
    for s in sizes[:-1]:
        offsets.append(offsets[-1] + s)
    total = sum(sizes)
    return sizes, offsets, total

def build_ps_from_sources(Lx, Ly, ns=8, seed=0):
    """
    Returns ps callable matching your fem_core point_source style.
    Here we just build the same list of sources and make a callable.
    """
    from math import pi
    import numpy.linalg as la

    rng = np.random.default_rng(seed)
    sp_list = [rng.random(3) * np.array([Lx, Ly, 50.0]) for _ in range(ns)]

    def ps(x, kappa):
        v = np.zeros(x.shape[0], float)
        for s in sp_list:
            v += s[2] * np.exp(-10.0 * (kappa / (2.0 * pi))**2 * la.norm(x - s[None, 0:2], axis=1)**2)
        return v

    return sp_list, ps


# ----------------------------
# TESTS 2.1 + 2.2 + 2.3 + 2.4
# ----------------------------
def test_local_mesh_and_boundary(Lx, Ly, Nx, Ny, J):
    print("\n=== A/B: local_mesh + local_boundary ===")
    assert_true((Ny - 1) % J == 0, "Ny-1 multiple of J")

    Ly_loc = Ly / J
    ny_loc = ny_loc_from(Ny, J)

    for j in range(J):
        vtxj, eltj = local_mesh(Lx, Ly, Nx, Ny, j, J)
        belt_phys, belt_artf = local_boundary(Nx, Ny, j, J)

        # A) mesh shapes
        assert_shape(vtxj, (Nx * ny_loc, 2), f"vtxj shape (j={j})")
        assert_shape(eltj, (2 * (Nx - 1) * (ny_loc - 1), 3), f"eltj shape (j={j})")

        # connectivity bounds
        assert_true(eltj.min() >= 0, f"eltj min>=0 (j={j})")
        assert_true(eltj.max() < vtxj.shape[0], f"eltj max < nloc (j={j})")

        # y-range check (global coords)
        y_min = vtxj[:, 1].min()
        y_max = vtxj[:, 1].max()
        assert_allclose(y_min, j * Ly_loc, tol=1e-12, msg=f"y_min slab (j={j})")
        assert_allclose(y_max, (j + 1) * Ly_loc, tol=1e-12, msg=f"y_max slab (j={j})")

        # B) boundary indices within local range
        if belt_phys.size > 0:
            assert_true(belt_phys.min() >= 0, f"belt_phys min>=0 (j={j})")
            assert_true(belt_phys.max() < Nx * ny_loc, f"belt_phys max<nloc (j={j})")
        if belt_artf.size > 0:
            assert_true(belt_artf.min() >= 0, f"belt_artf min>=0 (j={j})")
            assert_true(belt_artf.max() < Nx * ny_loc, f"belt_artf max<nloc (j={j})")

        # quick logical checks on artificial boundary existence
        if J == 1:
            assert_true(belt_artf.size == 0, "J=1 => no artificial interfaces")
        else:
            if j == 0:
                # only top interface
                assert_true(belt_artf.size > 0, "j=0 has top artificial interface")
            elif j == J - 1:
                assert_true(belt_artf.size > 0, "j=J-1 has bottom artificial interface")
            else:
                assert_true(belt_artf.size > 0, "internal slab has two artificial interfaces (edges exist)")

    _ok("local_mesh/local_boundary basic sanity passed")


def test_restriction_matrices(Nx, Ny, J):
    print("\n=== C/D/E: Rj, Bj, Cj ===")
    assert_true((Ny - 1) % J == 0, "Ny-1 multiple of J")

    ny_cells_loc = (Ny - 1) // J
    ny_loc = ny_cells_loc + 1
    nglob = Nx * Ny

    # C) Rj correctness on an arange global vector
    xg = np.arange(nglob, dtype=np.float64)
    for j in range(J):
        Rj = Rj_matrix(Nx, Ny, j, J).tocsr()
        assert_shape(Rj, (Nx * ny_loc, nglob), f"Rj shape (j={j})")
        xj = Rj @ xg

        y0 = j * ny_cells_loc
        y1 = (j + 1) * ny_cells_loc
        gy = np.arange(y0, y1 + 1, dtype=np.int64)
        gx = np.arange(0, Nx, dtype=np.int64)
        expected = (gy[:, None] * Nx + gx[None, :]).reshape(-1).astype(np.float64)

        assert_allclose(xj, expected, tol=0.0, msg=f"Rj @ arange exact match (j={j})")
        assert_true(np.all(Rj.data == 1.0), f"Rj boolean entries are 1 (j={j})")

    # D) Bj correctness w.r.t belt_artf vertices
    for j in range(J):
        belt_phys, belt_artf = local_boundary(Nx, Ny, j, J)
        Bj = Bj_matrix(Nx, Ny, j, J, belt_artf).tocsr()

        nloc = Nx * ny_loc
        assert_shape(Bj, (Bj.shape[0], nloc), f"Bj cols==nloc (j={j})")

        u_local = np.arange(nloc, dtype=np.float64)
        trace = Bj @ u_local

        if belt_artf.size == 0:
            assert_true(trace.size == 0, f"Bj empty when no artificial interface (j={j})")
            continue

        sigma_vertices = np.unique(belt_artf.reshape(-1)).astype(np.int64)
        sigma_vertices.sort()
        expected = u_local[sigma_vertices]
        assert_allclose(trace, expected, tol=0.0, msg=f"Bj selects sigma vertices exactly (j={j})")

        # size check: slab endpoints vs internal
        exp_nsigma = nsigma_for(Nx, j, J)
        assert_true(trace.size == exp_nsigma, f"|V(Sigma_j)| == {exp_nsigma} (j={j})")

    # E) Cj extracts the block from V(S)
    sizes, offsets, total = block_offsets(Nx, J)
    xS = np.zeros(total, dtype=np.float64)
    for j in range(J):
        off = offsets[j]
        xS[off : off + sizes[j]] = 100.0 * (j + 1)

    for j in range(J):
        Cj = Cj_matrix(Nx, Ny, j, J).tocsr()
        assert_shape(Cj, (sizes[j], total), f"Cj shape (j={j})")
        out = Cj @ xS
        assert_true(out.size == sizes[j], f"Cj output size (j={j})")
        if out.size > 0:
            assert_allclose(out, 100.0 * (j + 1), tol=0.0, msg=f"Cj extracts constant block (j={j})")

    _ok("Rj/Bj/Cj sanity passed")


def test_local_problems(Lx, Ly, Nx, Ny, J, kappa, ns=8, seed=0):
    print("\n=== F/G/H: Aj, Tj, Sj factorization, bj ===")
    assert_true((Ny - 1) % J == 0, "Ny-1 multiple of J")

    sp_list, ps_callable = build_ps_from_sources(Lx, Ly, ns=ns, seed=seed)
    ny_loc = ny_loc_from(Ny, J)
    nloc = Nx * ny_loc

    for j in range(J):
        vtxj, eltj = local_mesh(Lx, Ly, Nx, Ny, j, J)
        belt_phys, belt_artf = local_boundary(Nx, Ny, j, J)

        Bj = Bj_matrix(Nx, Ny, j, J, belt_artf).tocsr()

        # F) Aj
        Aj = Aj_matrix(vtxj, eltj, belt_phys, kappa).tocsr()
        assert_shape(Aj, (nloc, nloc), f"Aj shape (j={j})")
        assert_true(sp.issparse(Aj), f"Aj is sparse (j={j})")

        # G) Tj
        Tj = Tj_matrix(vtxj, belt_artf, Bj, kappa).tocsr()
        nsigma = nsigma_for(Nx, j, J)
        assert_shape(Tj, (nsigma, nsigma), f"Tj shape (j={j})")

        # quick symmetry check (mass-like)
        if nsigma > 0:
            Tdiff = (Tj - Tj.T)
            assert_true(Tdiff.nnz == 0 or la.norm(Tdiff.data) < 1e-12, f"Tj ~ symmetric (j={j})")

        # H) Sj factorization solve test
        lu, Sj_mat = Sj_factorization(Aj, Tj, Bj)
        rhs = (np.random.default_rng(1234 + j).standard_normal(nloc)
               + 1j * np.random.default_rng(5678 + j).standard_normal(nloc))
        u = lu.solve(rhs)
        res = Sj_mat @ u - rhs
        rel = la.norm(res) / max(1.0, la.norm(rhs))
        assert_true(rel < 1e-9, f"Sj solve residual small (j={j}), rel={rel:.3e}")

        # bj vector sanity (dimension + deterministic)
        bj = bj_vector(vtxj, eltj, lambda x: ps_callable(x, kappa), kappa)
        assert_true(bj.shape == (nloc,), f"bj shape (j={j})")

    _ok("Local problems sanity passed")


def test_global_operators(Lx, Ly, Nx, Ny, J, kappa, ns=8, seed=0):
    print("\n=== I/J/K: Pi, S, g ===")
    sp_list, ps_callable = build_ps_from_sources(Lx, Ly, ns=ns, seed=seed)

    subs = build_subdomains(Lx, Ly, Nx, Ny, J, kappa, ps=lambda x: ps_callable(x, kappa))
    sizes, offsets, total = block_offsets(Nx, J)

    # Create a random interface vector x in V(S)
    rng = np.random.default_rng(42)
    xS = rng.standard_normal(total) + 1j * rng.standard_normal(total)

    # I) Pi involution: Pi(Pi(x)) = x
    y = Pi_operator(xS, Nx, J)
    z = Pi_operator(y, Nx, J)

    # This SHOULD pass if Pi fills every entry and is a swap.
    # With your current Pi_operator, it will FAIL for J>=3 because many entries are left 0.
    assert_allclose(z, xS, tol=1e-12, msg="Pi(Pi(x)) == x (involution)")

    # Also check that Pi is a permutation (norm preserved)
    assert_allclose(la.norm(y), la.norm(xS), tol=1e-12, msg="||Pi(x)|| == ||x||")

    # J) S operator output size + no NaN/Inf
    Sx = S_operator(xS, subs)
    assert_true(Sx.shape == xS.shape, "S_operator output size equals input")
    assert_true(np.isfinite(Sx.real).all() and np.isfinite(Sx.imag).all(), "S_operator finite entries")

    # K) g vector size + reproducibility
    g1 = g_vector(subs)
    g2 = g_vector(subs)
    assert_true(g1.shape == (total,), "g has correct size")
    assert_allclose(g1, g2, tol=0.0, msg="g reproducible (same subs)")

    _ok("Global operators sanity passed")


def run_case(name, Lx, Ly, Nx, Ny, J, kappa):
    print("\n" + "="*80)
    print(f"CASE: {name}  (Nx={Nx}, Ny={Ny}, J={J}, kappa={kappa})")
    print("="*80)

    test_local_mesh_and_boundary(Lx, Ly, Nx, Ny, J)
    test_restriction_matrices(Nx, Ny, J)
    test_local_problems(Lx, Ly, Nx, Ny, J, kappa, ns=8, seed=0)
    test_global_operators(Lx, Ly, Nx, Ny, J, kappa, ns=8, seed=0)


def main():
    Lx, Ly = 1.0, 2.0
    kappa = 16.0

    # Case 1: micro
    run_case("micro", Lx, Ly, Nx=9, Ny=17, J=2, kappa=kappa)

    # Case 2: medium
    run_case("medium", Lx, Ly, Nx=33, Ny=65, J=4, kappa=kappa)

    # Case 3: stress-light (comment if too slow locally)
    # run_case("stress-light", Lx, Ly, Nx=65, Ny=129, J=8, kappa=kappa)


if __name__ == "__main__":
    main()
