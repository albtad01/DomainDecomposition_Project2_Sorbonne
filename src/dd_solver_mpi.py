#! /usr/bin/python3
"""
MPI-parallel domain decomposition solver using fixed-point iteration.

Each MPI rank owns ONE subdomain and performs local operations.
Communication happens only during the Π operator (neighbor exchange).

Usage example:
    mpirun -np 4 python dd_solver_mpi.py --Lx 2.0 --Ly 2.0 --m 64 --J 4

Algorithm:
    Solve: (I + Π S) p = g
    
    via Richardson iteration:
        p_{n+1} = [(1-ω)I - ω Π S] p_n + ω g
    
    Steps per iteration:
        1. LOCAL:  Sp_j = S_j(p_j)           [no communication]
        2. COMM:   PiSp_j = Π(Sp)            [MPI exchange]
        3. LOCAL:  p_j^{n+1} = update        [no communication]
        4. COMM:   check convergence          [MPI Allreduce]
"""
from __future__ import annotations

import numpy as np
import numpy.linalg as la
from mpi4py import MPI

from dd_operators import build_subdomains, SubdomainData
from dd_operators_mpi import (
    apply_S_local,
    exchange_with_neighbors,
    compute_g_parallel
)


def parallel_fixed_point_solver_mpi(
    local_sub: SubdomainData,
    comm: MPI.Comm,
    tol: float = 1e-6,
    maxiter: int = 200,
    omega: float = 1.0,
    verbose: bool = False
):
    """
    MPI-parallel fixed-point solver for the interface problem.
    
    Each MPI rank calls this function with its LOCAL subdomain data.
    
    Solves: (I + Π S) p = g
    
    via: p_{n+1} = [(1-ω)I - ω Π S] p_n + ω g
    
    Parameters
    ----------
    local_sub : SubdomainData
        Precomputed subdomain data for THIS rank's subdomain
    comm : MPI.Comm
        MPI communicator (typically MPI.COMM_WORLD)
    tol : float
        Convergence tolerance ||p_{n+1} - p_n|| / ||p_n||
    maxiter : int
        Maximum number of iterations
    omega : float
        Relaxation parameter (ω ∈ (0, 2) for convergence)
    verbose : bool
        If True, rank 0 prints convergence history
    
    Returns
    -------
    p_local : (n_interface_j,) complex
        Converged interface solution for this subdomain
    history : dict
        'iterations': int, number of iterations
        'residuals': list of relative residual norms
        'rank': int, this process's rank
    """
    rank = comm.Get_rank()
    size = comm.Get_size()
    J = local_sub.J
    
    if size != J:
        if rank == 0:
            print(f"ERROR: MPI size ({size}) must equal number of subdomains J ({J})")
        comm.Abort(1)
    
    if verbose and rank == 0:
        print(f"  MPI Fixed-Point Solver Initialization:")
        print(f"    Ranks: {size}")
        print(f"    Subdomains: {J}")
    
    # Build right-hand side g using MPI communication
    if verbose and rank == 0:
        print(f"    Computing g vector (with MPI exchange)...")
    
    g_local = compute_g_parallel(local_sub, rank, J, comm)
    
    if verbose:
        print(f"    [Rank {rank}] g_local size: {g_local.shape[0]}")
    
    # Initialize local interface data
    n_interface_local = g_local.shape[0]
    p_local = np.zeros(n_interface_local, dtype=np.complex128)
    
    residuals = []
    
    if verbose and rank == 0:
        print(f"    Starting iterations...")
        print()
        print(f"{'Iter':<6} {'||p_n+1 - p_n|| / ||p_n||':<30}")
        print("-" * 40)
    
    for n in range(maxiter):
        # ===== STEP 1: LOCAL S operator =====
        # Each rank computes S_j(p_j) independently (no communication)
        if verbose and n == 0 and rank == 0:
            print(f"  [Iteration structure]")
            print(f"    Step 1: LOCAL - Apply S_j operator")
            print(f"    Step 2: MPI - Exchange via Π operator")
            print(f"    Step 3: LOCAL - Update p")
            print(f"    Step 4: MPI - Check convergence")
            print()
        
        Sp_local = apply_S_local(p_local, local_sub)
        
        # ===== STEP 2: COMMUNICATION - Π operator =====
        # Exchange interface data with neighbors via MPI
        PiSp_local = exchange_with_neighbors(Sp_local, rank, J, comm)
        
        # ===== STEP 3: LOCAL UPDATE =====
        # Each rank updates its local interface data
        # p_{n+1} = (1-ω) p_n - ω Π S p_n + ω g
        p_new_local = (1 - omega) * p_local - omega * PiSp_local + omega * g_local
        
        # ===== STEP 4: CONVERGENCE CHECK (with MPI communication) =====
        # Compute local contribution to global norm
        dp_local = p_new_local - p_local
        local_norm_sq = np.linalg.norm(dp_local) ** 2
        p_norm_sq = np.linalg.norm(p_local) ** 2
        
        # Global reduction to compute ||p_n+1 - p_n|| and ||p_n||
        global_norm_sq = comm.allreduce(local_norm_sq, op=MPI.SUM)
        p_global_norm_sq = comm.allreduce(p_norm_sq, op=MPI.SUM)
        
        rel_err = np.sqrt(global_norm_sq) / (np.sqrt(p_global_norm_sq) + 1e-14)
        residuals.append(rel_err)
        
        if verbose and rank == 0:
            print(f"{n:<6} {rel_err:<30.4e}")
        
        # Update for next iteration
        p_local = p_new_local
        
        # Check convergence
        if rel_err < tol:
            if verbose and rank == 0:
                print()
                print(f"  ✓ Converged in {n+1} iterations")
                print(f"  ✓ Final residual: {rel_err:.4e}")
            return p_local, {
                'iterations': n + 1,
                'residuals': residuals,
                'rank': rank
            }
    
    if verbose and rank == 0:
        print()
        print(f"  ⚠ Warning: did not converge after {maxiter} iterations")
        print(f"  ⚠ Final residual: {residuals[-1]:.4e}")
    
    return p_local, {
        'iterations': maxiter,
        'residuals': residuals,
        'rank': rank
    }


def reconstruct_local_solution(p_local, local_sub, verbose=False, rank=None):
    """
    Reconstruct the local solution u_j from interface data p_j.
    
    For subdomain j:
        (A_j - i B_j^* T_j B_j) u_j = b_j + B_j^* T_j p_j
    
    Parameters
    ----------
    p_local : (n_interface_j,) complex
        Converged interface solution for this subdomain
    local_sub : SubdomainData
        Precomputed subdomain data
    verbose : bool, optional
        Print verbose output
    rank : int, optional
        MPI rank (for verbose output)
    
    Returns
    -------
    u_local : (n_dof_j,) complex
        Local FEM solution on subdomain j
    """
    if verbose and rank is not None:
        print(f"  [Rank {rank}] Reconstructing local solution...")
        print(f"    Interface DOFs: {p_local.shape[0]}")
        print(f"    Total DOFs: {local_sub.bj.shape[0]}")
    
    # Compute right-hand side: b_j + B_j^* T_j p_j
    rhs = local_sub.bj.astype(np.complex128) + (
        local_sub.Bj.conjugate().T @ (local_sub.Tj @ p_local)
    )
    
    # Solve: (A_j - i B_j^* T_j B_j) u_j = rhs
    u_local = local_sub.lu.solve(rhs)
    
    if verbose and rank is not None:
        print(f"    Solution norm: {np.linalg.norm(u_local):.6e}")
    
    return u_local


def gather_global_solution(u_local, local_sub, comm, verbose=False):
    """
    Gather local solutions from all ranks to rank 0.
    
    This is useful for visualization or post-processing.
    Only rank 0 will have the full solution u_global.
    
    Parameters
    ----------
    u_local : (n_dof_j,) complex
        Local solution on this subdomain
    local_sub : SubdomainData
        Local subdomain data (contains vertex coordinates)
    comm : MPI.Comm
        MPI communicator
    verbose : bool, optional
        Print verbose output
    
    Returns
    -------
    u_global : dict or None
        Dictionary with global solution data (only on rank 0)
        Keys: 'u_values', 'vertices', 'subdomains'
        None on all other ranks
    """
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if verbose and rank == 0:
        print(f"  Gathering solutions from all {size} ranks...")
    
    # Gather all local data to rank 0
    all_u = comm.gather(u_local, root=0)
    all_vtx = comm.gather(local_sub.vtxj, root=0)
    all_elt = comm.gather(local_sub.eltj, root=0)
    
    if rank == 0:
        if verbose:
            total_dofs = sum(u.shape[0] for u in all_u)
            print(f"    Gathered {size} local solutions")
            print(f"    Total DOFs: {total_dofs}")
        
        return {
            'u_values': all_u,
            'vertices': all_vtx,
            'elements': all_elt,
            'J': size
        }
    else:
        return None
