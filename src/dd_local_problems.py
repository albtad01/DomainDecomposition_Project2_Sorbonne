#! /usr/bin/python3
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import numpy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from fem_core import mass, stiffness, point_source


def Aj_matrix(vtxj, eltj, beltj_phys, kappa):
    """
    Local FEM matrix on subdomain Ωj:
        Aj = K - kappa^2 M - i*kappa*Mb_phys

    Parameters
    ----------
    vtxj : (nloc, 2) coordinates
    eltj : (ntri, 3) triangle connectivity (local)
    beltj_phys : (nedge, 2) boundary edges on ∂Ωj ∩ ∂Ω (local indices)
    kappa : float

    Returns
    -------
    Aj : sparse (nloc, nloc) complex
    """
    M = mass(vtxj, eltj)                     # (nloc, nloc)
    K = stiffness(vtxj, eltj)                # (nloc, nloc)

    if beltj_phys is None or beltj_phys.size == 0:
        Mb_phys = sp.csr_matrix(M.shape, dtype=np.float64)
    else:
        Mb_phys = mass(vtxj, beltj_phys)     # (nloc, nloc), edge mass

    Aj = K - (kappa**2) * M - 1j * kappa * Mb_phys
    return Aj.astype(np.complex128)


def Tj_matrix(vtxj, beltj_artf, Bj, kappa):
    """
    Transmission matrix:
        Tj = kappa * M_{Σj}   in interface DOF space

    We build it as:
        M_sigma_vol = mass(vtxj, beltj_artf)     (nloc x nloc, only interface vertices touched)
        Tj = kappa * (Bj * M_sigma_vol * Bj^T)   (nsigma x nsigma)

    Parameters
    ----------
    vtxj : (nloc, 2)
    beltj_artf : (nedge, 2) artificial boundary edges (local indices)
    Bj : sparse (nsigma, nloc) boolean selector
    kappa : float

    Returns
    -------
    Tj : sparse (nsigma, nsigma) real/float (but we keep float64)
    """
    nsigma = Bj.shape[0]
    if nsigma == 0:
        return sp.csr_matrix((0, 0), dtype=np.float64)

    if beltj_artf is None or beltj_artf.size == 0:
        return sp.csr_matrix((nsigma, nsigma), dtype=np.float64)

    M_sigma_vol = mass(vtxj, beltj_artf).tocsr()  # (nloc, nloc)
    Tj = (Bj @ (M_sigma_vol @ Bj.T)).tocsr()       # (nsigma, nsigma)
    Tj = (kappa * Tj).astype(np.float64)
    return Tj


def Sj_factorization(Aj, Tj, Bj):
    """
    Build and factorize:
        Sj = Aj - i * Bj^* Tj Bj

    Returns
    -------
    lu : SuperLU factorization object (splu)
    Sj : sparse CSC matrix
    """
    # Bj : (nsigma, nloc)
    # Bj^* Tj Bj : (nloc, nloc)
    Bt = Bj.conjugate().T
    Robin_artf = (Bt @ (Tj @ Bj)).tocsr()
    Sj = (Aj - 1j * Robin_artf).tocsc()
    lu = spla.splu(Sj)
    return lu, Sj


def bj_vector(vtxj, eltj, ps, kappa):
    """
    Local RHS:
        bj = M * ps(vtxj)

    ps can be:
    - a callable ps(x) returning values at vertices, OR
    - a list/array of sources sp (same as in your baseline), in which case we build point_source(sp, kappa).

    Returns
    -------
    bj : (nloc,) complex or float
    """
    M = mass(vtxj, eltj)

    if callable(ps):
        f = ps(vtxj)
    else:
        # assume ps is "sp" list as in baseline
        f = point_source(ps, kappa)(vtxj)

    bj = M @ f
    return np.asarray(bj).reshape(-1)
