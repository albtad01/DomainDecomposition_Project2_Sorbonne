#! /usr/bin/python3
"""
MPI-parallel versions of domain decomposition operators.

Each MPI rank owns ONE subdomain and performs local operations.
Communication happens only in the Π operator (exchange_with_neighbors).
"""
from __future__ import annotations

import numpy as np
from mpi4py import MPI


def apply_S_local(p_local, local_sub):
    """
    Apply the local scattering operator S_j to interface data p_j.
    
    For subdomain j, this computes:
        S_j(p_j) = p_j + 2i * B_j * (A_j - i B_j^* T_j B_j)^{-1} * B_j^* T_j p_j
    
    This is a PURELY LOCAL operation - no MPI communication needed.
    
    Parameters
    ----------
    p_local : (n_interface_j,) complex
        Interface data for this subdomain
    local_sub : SubdomainData
        Precomputed local subdomain data (includes LU factorization)
    
    Returns
    -------
    Sp_local : (n_interface_j,) complex
        Result of S_j applied to p_j
    """
    p_local = np.asarray(p_local).astype(np.complex128)
    
    # Compute right-hand side: B_j^* T_j p_j
    rhs = local_sub.Bj.conjugate().T @ (local_sub.Tj @ p_local)
    
    # Solve: (A_j - i B_j^* T_j B_j) v_j = rhs
    vj = local_sub.lu.solve(rhs)
    
    # Return: S_j(p_j) = p_j + 2i B_j v_j
    Sp_local = p_local + 2j * (local_sub.Bj @ vj)
    
    return Sp_local


def exchange_with_neighbors(data_local, rank, J, comm):
    """
    MPI implementation of the Π operator for slab decomposition.
    
    Exchanges interface data with neighboring subdomains:
    - Subdomain j sends its TOP interface to subdomain j+1
    - Subdomain j sends its BOTTOM interface to subdomain j-1
    - After exchange, each subdomain receives data from its neighbors
    
    For slab decomposition in y-direction:
    - Rank 0 (bottom):    has only TOP interface    (size Nx)
    - Rank J-1 (top):     has only BOTTOM interface (size Nx)
    - Rank j (internal):  has BOTTOM + TOP          (size 2*Nx)
    
    Parameters
    ----------
    data_local : (n_interface_j,) complex
        Interface data for this subdomain (before exchange)
    rank : int
        MPI rank of this process (subdomain index j)
    J : int
        Total number of subdomains
    comm : MPI.Comm
        MPI communicator
    
    Returns
    -------
    result : (n_interface_j,) complex
        Interface data after exchange with neighbors
    """
    data_local = np.asarray(data_local).astype(np.complex128)
    
    if J == 1:
        # No neighbors, no exchange
        return data_local.copy()
    
    # Determine interface sizes
    if rank == 0:
        # Only TOP interface
        Nx = data_local.shape[0]
        result = np.zeros_like(data_local)
        
        # Sendrecv: send TOP to rank 1, receive from rank 1 (its BOTTOM)
        top_data = data_local[:].copy()
        received = comm.sendrecv(top_data, dest=rank+1, sendtag=0,
                                 source=rank+1, recvtag=1)
        result[:] = received
        
    elif rank == J - 1:
        # Only BOTTOM interface
        Nx = data_local.shape[0]
        result = np.zeros_like(data_local)
        
        # Sendrecv: send BOTTOM to rank J-2, receive from rank J-2 (its TOP)
        bottom_data = data_local[:].copy()
        received = comm.sendrecv(bottom_data, dest=rank-1, sendtag=rank,
                                 source=rank-1, recvtag=rank-1)
        result[:] = received
        
    else:
        # Internal subdomain: BOTTOM [0:Nx], TOP [Nx:2*Nx]
        Nx = data_local.shape[0] // 2
        result = np.zeros_like(data_local)
        
        bottom_data = data_local[0:Nx].copy()
        top_data = data_local[Nx:2*Nx].copy()
        
        # Exchange with lower neighbor (rank-1)
        # Sendrecv: send our BOTTOM, receive their TOP
        received_bottom = comm.sendrecv(bottom_data, dest=rank-1, sendtag=rank,
                                        source=rank-1, recvtag=rank-1)
        
        # Exchange with upper neighbor (rank+1)
        # Sendrecv: send our TOP, receive their BOTTOM
        received_top = comm.sendrecv(top_data, dest=rank+1, sendtag=rank,
                                     source=rank+1, recvtag=rank+1)
        
        result[0:Nx] = received_bottom
        result[Nx:2*Nx] = received_top
    
    return result


def compute_g_local(local_sub):
    """
    Compute the LOCAL part of the g vector for subdomain j.
    
    For subdomain j, compute:
        q_j = B_j * (A_j - i B_j^* T_j B_j)^{-1} * b_j
    
    The global g vector is then: g = -2i * Π(q)
    
    This function returns q_j (before applying Π and -2i).
    
    Parameters
    ----------
    local_sub : SubdomainData
        Precomputed local subdomain data
    
    Returns
    -------
    qj : (n_interface_j,) complex
        Local contribution to g vector (before Π and scaling)
    """
    # Solve: (A_j - i B_j^* T_j B_j) u0_j = b_j
    rhs = local_sub.bj.astype(np.complex128)
    u0j = local_sub.lu.solve(rhs)
    
    # Compute: q_j = B_j u0_j
    qj = (local_sub.Bj @ u0j).astype(np.complex128)
    
    return qj


def compute_g_parallel(local_sub, rank, J, comm):
    """
    Compute the full g vector in parallel using MPI.
    
    Each rank computes its local q_j, then applies Π via MPI exchange,
    and finally scales by -2i.
    
    g = -2i * Π(q)
    
    Parameters
    ----------
    local_sub : SubdomainData
        Precomputed local subdomain data
    rank : int
        MPI rank of this process
    J : int
        Total number of subdomains
    comm : MPI.Comm
        MPI communicator
    
    Returns
    -------
    g_local : (n_interface_j,) complex
        Local g vector for this subdomain
    """
    # Compute local part
    qj = compute_g_local(local_sub)
    
    # Apply Π operator via MPI exchange
    Pi_qj = exchange_with_neighbors(qj, rank, J, comm)
    
    # Scale by -2i
    g_local = -2j * Pi_qj
    
    return g_local
