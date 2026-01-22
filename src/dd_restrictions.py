import numpy as np
from scipy.sparse import csr_matrix



def Rj_matrix(Nx, Ny, j, J):
    """
    Rj : V(Ω) -> V(Ωj)
    Boolean restriction matrix selecting the global DOFs belonging to subdomain j.

    Shape: (|V(Ωj)|, |V(Ω)|) = (Nx*ny_loc, Nx*Ny)
    """
    if not (0 <= j < J):
        raise ValueError(f"j must be in [0, J-1], got j={j}, J={J}")
    if (Ny - 1) % J != 0:
        raise ValueError(f"Ny-1 must be a multiple of J. Got Ny={Ny}, J={J}")

    ny_cells_loc = (Ny - 1) // J
    ny_loc = ny_cells_loc + 1  # local points in y

    # global y-indices covered by slab j (inclusive endpoints)
    y0 = j * ny_cells_loc
    y1 = (j + 1) * ny_cells_loc
    gy = np.arange(y0, y1 + 1, dtype=np.int64)              # (ny_loc,)
    gx = np.arange(0, Nx, dtype=np.int64)                   # (Nx,)

    # global vertex indices in the same row-major ordering as mesh()
    gidx = (gy[:, None] * Nx + gx[None, :]).reshape(-1)      # (Nx*ny_loc,)

    nloc = Nx * ny_loc
    nglob = Nx * Ny

    rows = np.arange(nloc, dtype=np.int64)
    cols = gidx
    data = np.ones(nloc, dtype=np.float64)

    return csr_matrix((data, (rows, cols)), shape=(nloc, nglob))


def Bj_matrix(Nx, Ny, j, J, beltj_artf):
    """
    Bj : V(Ωj) -> V(Σj)
    Boolean trace/restriction matrix selecting DOFs on the artificial interfaces Σj
    from the local volume DOFs.

    beltj_artf uses LOCAL vertex indices (as returned by local_boundary()).

    Shape: (|V(Σj)|, |V(Ωj)|)
    """
    if not (0 <= j < J):
        raise ValueError(f"j must be in [0, J-1], got j={j}, J={J}")
    if (Ny - 1) % J != 0:
        raise ValueError(f"Ny-1 must be a multiple of J. Got Ny={Ny}, J={J}")

    ny_cells_loc = (Ny - 1) // J
    ny_loc = ny_cells_loc + 1
    nloc = Nx * ny_loc

    # vertices on Σj = unique vertices touched by artificial edges
    if beltj_artf.size == 0:
        # no artificial interface (e.g. J=1) -> empty
        return csr_matrix((0, nloc), dtype=np.float64)

    sigma_vertices = np.unique(beltj_artf.reshape(-1)).astype(np.int64)
    sigma_vertices.sort()  # deterministic ordering: left->right on each interface line

    nsigma = sigma_vertices.size
    rows = np.arange(nsigma, dtype=np.int64)
    cols = sigma_vertices
    data = np.ones(nsigma, dtype=np.float64)

    return csr_matrix((data, (rows, cols)), shape=(nsigma, nloc))


def Cj_matrix(Nx, Ny, j, J):
    """
    Cj : V(S) -> V(Σj)
    Boolean matrix extracting the j-th block xj from x = (x0, x1, ..., x_{J-1}) in V(S).

    Shape: (|V(Σj)|, |V(S)|) where |V(S)| = sum_k |V(Σk)|
    """
    if not (0 <= j < J):
        raise ValueError(f"j must be in [0, J-1], got j={j}, J={J}")
    if (Ny - 1) % J != 0:
        raise ValueError(f"Ny-1 must be a multiple of J. Got Ny={Ny}, J={J}")

    # Σj = artificial interfaces only:
    # if J==1 => no artificial interfaces
    if J == 1:
        return csr_matrix((0, 0), dtype=np.float64)

    # number of interface vertices per subdomain (slab decomposition):
    # - j=0: only top interface -> Nx
    # - j=J-1: only bottom interface -> Nx
    # - internal: bottom + top -> 2*Nx
    def nsigma_for(k):
        if k == 0 or k == J - 1:
            return Nx
        return 2 * Nx

    nsigma_j = nsigma_for(j)
    nsigma_total = sum(nsigma_for(k) for k in range(J))

    offset = sum(nsigma_for(k) for k in range(j))  # block start index in V(S)

    rows = np.arange(nsigma_j, dtype=np.int64)
    cols = offset + rows
    data = np.ones(nsigma_j, dtype=np.float64)

    return csr_matrix((data, (rows, cols)), shape=(nsigma_j, nsigma_total))
