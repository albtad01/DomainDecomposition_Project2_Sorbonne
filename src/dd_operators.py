#! /usr/bin/python3
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import scipy.sparse as sp

from dd_mesh import local_mesh, local_boundary
from dd_restrictions import Bj_matrix, Cj_matrix
from dd_local_problems import Aj_matrix, Tj_matrix, Sj_factorization, bj_vector


def _nsigma_for(Nx, j, J):
    # Σj = artificial interfaces only (as in your 2.2.3)
    if J == 1:
        return 0
    if j == 0 or j == J - 1:
        return Nx
    return 2 * Nx


def _block_offsets(Nx, J):
    sizes = [_nsigma_for(Nx, j, J) for j in range(J)]
    offsets = [0]
    for s in sizes[:-1]:
        offsets.append(offsets[-1] + s)
    total = sum(sizes)
    return sizes, offsets, total


def _top_slice_in_block(Nx, j, J):
    # returns (start_in_block, length) for the TOP interface part of subdomain j
    # ordering is consistent with Bj_matrix using sorted local vertex indices:
    # for internal slabs: bottom first (0..Nx-1), then top (Nx..2Nx-1)
    if J == 1:
        return (0, 0)
    if j == 0:
        return (0, Nx)          # only top
    if j == J - 1:
        return (0, 0)           # no top (only bottom)
    return (Nx, Nx)             # internal: top is second chunk


def _bottom_slice_in_block(Nx, j, J):
    # returns (start_in_block, length) for the BOTTOM interface part of subdomain j
    if J == 1:
        return (0, 0)
    if j == 0:
        return (0, 0)           # no bottom (only top)
    return (0, Nx)              # j>0 always has bottom interface


@dataclass
class SubdomainData:
    j: int
    Nx: int
    Ny: int
    J: int
    kappa: float

    vtxj: np.ndarray
    eltj: np.ndarray
    beltj_phys: np.ndarray
    beltj_artf: np.ndarray

    Bj: sp.csr_matrix
    Cj: sp.csr_matrix

    Aj: sp.csr_matrix
    Tj: sp.csr_matrix
    lu: object  # SuperLU

    bj: np.ndarray


def build_subdomains(Lx, Ly, Nx, Ny, J, kappa, ps):
    """
    Precompute everything needed for operators S, Π and g.

    ps: callable or sp-list (see bj_vector doc)
    """
    if (Ny - 1) % J != 0:
        raise ValueError(f"Ny-1 must be multiple of J for slab decomposition. Got Ny={Ny}, J={J}")

    subs = []
    for j in range(J):
        vtxj, eltj = local_mesh(Lx, Ly, Nx, Ny, j, J)
        beltj_phys, beltj_artf = local_boundary(Nx, Ny, j, J)

        Bj = Bj_matrix(Nx, Ny, j, J, beltj_artf).tocsr()
        Cj = Cj_matrix(Nx, Ny, j, J).tocsr()

        Aj = Aj_matrix(vtxj, eltj, beltj_phys, kappa).tocsr()
        Tj = Tj_matrix(vtxj, beltj_artf, Bj, kappa).tocsr()
        lu, _ = Sj_factorization(Aj, Tj, Bj)

        bj = bj_vector(vtxj, eltj, ps, kappa)

        subs.append(SubdomainData(
            j=j, Nx=Nx, Ny=Ny, J=J, kappa=kappa,
            vtxj=vtxj, eltj=eltj,
            beltj_phys=beltj_phys, beltj_artf=beltj_artf,
            Bj=Bj, Cj=Cj,
            Aj=Aj, Tj=Tj, lu=lu,
            bj=bj
        ))
    return subs


def S_operator(p, subs):
    """
    Apply scattering operator S to p in V(S), consistently with the PDF:

      S = I + 2 i B (A - i B^* T B)^{-1} B^* T

    For each j:
      solve  (Aj - i Bj^* Tj Bj) vj = Bj^* (Tj pj)
      output sj = pj + 2 i (Bj vj)
    """
    p = np.asarray(p).reshape(-1)
    y = np.zeros_like(p, dtype=np.complex128)

    Nx = subs[0].Nx
    J = subs[0].J
    sizes, offsets, _ = _block_offsets(Nx, J)

    for sd in subs:
        j = sd.j
        off = offsets[j]
        ns = sizes[j]
        if ns == 0:
            continue

        pj = p[off:off+ns].astype(np.complex128)

        rhs = sd.Bj.conjugate().T @ (sd.Tj @ pj)   # Bj^* Tj pj
        vj = sd.lu.solve(rhs)

        yj = pj + 2j * (sd.Bj @ vj)                # pj + 2 i Bj vj
        y[off:off+ns] = yj

    return y


def Pi_operator(x, Nx, J):
    """
    Π exchanges interface data between neighbouring subdomains.
    With slab decomposition, it is just swapping:
      top(Ωj) <-> bottom(Ω_{j+1})   for j=0..J-2

    No linear system.
    """
    x = np.asarray(x).reshape(-1)
    y = x.copy()

    sizes, offsets, _ = _block_offsets(Nx, J)

    # Swap each interface pair once
    for j in range(J - 1):
        off_j = offsets[j]
        off_k = offsets[j + 1]   # neighbour above

        # top slice in block j
        tj0, tjL = _top_slice_in_block(Nx, j, J)
        # bottom slice in block j+1
        bk0, bkL = _bottom_slice_in_block(Nx, j + 1, J)

        assert tjL == bkL == Nx

        top_j = x[off_j + tj0 : off_j + tj0 + tjL]
        bot_k = x[off_k + bk0 : off_k + bk0 + bkL]

        # swap
        y[off_j + tj0 : off_j + tj0 + tjL] = bot_k
        y[off_k + bk0 : off_k + bk0 + bkL] = top_j

    # For internal subdomains, also fill their other interface part via swaps above.
    # After the loop, everything is filled exactly once.
    return y


def g_vector(subs):
    """
    Build g consistently with the PDF:

      g = -2 i Π B (A - i B^* T B)^{-1} b

    For each j:
      solve (Aj - i Bj^* Tj Bj) u0j = bj
      take qj = Bj u0j
    Assemble q = concat(qj), then g = -2 i Π q
    """
    Nx = subs[0].Nx
    J = subs[0].J
    sizes, offsets, total = _block_offsets(Nx, J)

    q = np.zeros(total, dtype=np.complex128)

    for sd in subs:
        j = sd.j
        off = offsets[j]
        ns = sizes[j]
        if ns == 0:
            continue

        rhs = sd.bj.astype(np.complex128)
        u0j = sd.lu.solve(rhs)

        qj = (sd.Bj @ u0j).astype(np.complex128)   # Bj u0j
        q[off:off+ns] = qj

    g = (-2j) * Pi_operator(q, Nx, J)
    return g
