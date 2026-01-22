import numpy as np
na = np.newaxis

from fem_core import mesh


def local_mesh(Lx, Ly, Nx, Ny, j, J):
    """
    Build local mesh (vtxj, eltj) for subdomain Omega_j.
    Decomposition is in y only => J horizontal slabs.

    Parameters
    ----------
    Lx, Ly : floats
    Nx, Ny : ints (GLOBAL number of points in x and y)
    j      : int in [0, J-1]
    J      : int, number of subdomains

    Returns
    -------
    vtxj : (Nx*ny_loc, 2) float array (local vertex coords in GLOBAL coordinates)
    eltj : (2*(Nx-1)*(ny_loc-1), 3) int array (local triangles connectivity)
    """
    if not (0 <= j < J):
        raise ValueError(f"j must be in [0, J-1], got j={j}, J={J}")
    if (Ny - 1) % J != 0:
        raise ValueError(f"Ny-1 must be a multiple of J. Got Ny={Ny}, J={J}")

    # local number of points in y for a slab:
    ny_loc = (Ny - 1) // J + 1
    Ly_loc = Ly / J

    # Build a local mesh on a rectangle (0,Lx) x (0,Ly_loc)
    vtxj, eltj = mesh(Nx, ny_loc, Lx, Ly_loc)

    # Shift y-coordinates so that Omega_j is placed correctly in the global domain
    vtxj = vtxj.copy()
    vtxj[:, 1] += j * Ly_loc

    return vtxj, eltj


def local_boundary(Nx, Ny, j, J):
    """
    Build local boundary edges arrays for subdomain Omega_j:
    - beltj_phys : edges on physical boundary (∂Ωj ∩ ∂Ω)
    - beltj_artf : edges on artificial interfaces (∂Ωj \\ ∂Ω)

    Parameters
    ----------
    Nx, Ny : ints (GLOBAL number of points in x and y)
    j      : int in [0, J-1]
    J      : int

    Returns
    -------
    beltj_phys : (n_phys_edges, 2) int array of local vertex indices
    beltj_artf : (n_artf_edges, 2) int array of local vertex indices
    """
    if not (0 <= j < J):
        raise ValueError(f"j must be in [0, J-1], got j={j}, J={J}")
    if (Ny - 1) % J != 0:
        raise ValueError(f"Ny-1 must be a multiple of J. Got Ny={Ny}, J={J}")

    ny_loc = (Ny - 1) // J + 1

    # Local boundary edges for a (Nx, ny_loc) grid (same logic as boundary())
    bottom = np.hstack((np.arange(0, Nx-1, 1)[:, na],
                        np.arange(1, Nx,   1)[:, na])).astype(np.int64)

    top = np.hstack((np.arange(Nx*(ny_loc-1), Nx*ny_loc-1, 1)[:, na],
                     np.arange(Nx*(ny_loc-1)+1, Nx*ny_loc,   1)[:, na])).astype(np.int64)

    left = np.hstack((np.arange(0, Nx*(ny_loc-1), Nx)[:, na],
                      np.arange(Nx, Nx*ny_loc,    Nx)[:, na])).astype(np.int64)

    right = np.hstack((np.arange(Nx-1, Nx*(ny_loc-1), Nx)[:, na],
                       np.arange(2*Nx-1, Nx*ny_loc,   Nx)[:, na])).astype(np.int64)

    # Helper to stack possibly-empty lists
    def stack_edges(edges_list):
        if len(edges_list) == 0:
            return np.zeros((0, 2), dtype=np.int64)
        return np.vstack(edges_list).astype(np.int64)

    # Physical boundary: always left & right (x=0, x=Lx)
    phys = [left, right]
    # plus bottom if this is the first slab (touches y=0)
    if j == 0:
        phys.append(bottom)
    # plus top if this is the last slab (touches y=Ly)
    if j == J - 1:
        phys.append(top)
    beltj_phys = stack_edges(phys)

    # Artificial interfaces: internal horizontal boundaries only
    artf = []
    # bottom is artificial if there is a neighbor below (j>0)
    if j > 0:
        artf.append(bottom)
    # top is artificial if there is a neighbor above (j<J-1)
    if j < J - 1:
        artf.append(top)
    beltj_artf = stack_edges(artf)

    return beltj_phys, beltj_artf
