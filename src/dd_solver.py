#! /usr/bin/python3
"""
Fixed-point iterative solver for the interface problem.

Given the interface problem:
    (I + Π S) p = g

we solve it using the Richardson fixed-point iteration:
    p_{n+1} = [(1-ω)I - ω Π S] p_n + ω g

After convergence on p, we recover the volume solution u via local solves:
    u = (A - i B^* T B)^{-1} (B^* T p + b)
"""

from __future__ import annotations
import numpy as np
import numpy.linalg as la
import time
from scipy.sparse.linalg import gmres, LinearOperator
from dd_operators import S_operator, Pi_operator, g_vector, SubdomainData, _block_offsets


def fixed_point_solver(subs: list[SubdomainData], 
                       tol: float = 1e-6, 
                       maxiter: int = 200, 
                       omega: float = 1.0,
                       verbose: bool = False):
    """
    Solve the interface problem using fixed-point iteration:

        (I + Π S) p = g

    via the Richardson scheme:
        p_{n+1} = [(1-ω)I - ω Π S] p_n + ω g

    Parameters
    ----------
    subs : list of SubdomainData
        Precomputed subdomain data from build_subdomains()
    tol : float
        Convergence tolerance ||p_{n+1} - p_n|| / ||p_n||
    maxiter : int
        Maximum number of iterations
    omega : float
        Relaxation parameter (ω ∈ (0, 2) for convergence)
    verbose : bool
        Print convergence history

    Returns
    -------
    p : (total_interface_dof,) complex
        Converged interface solution
    history : dict
        'iterations': int, number of iterations
        'residuals': list of relative residual norms
    """
    Nx = subs[0].Nx
    J = subs[0].J
    sizes, offsets, total = _block_offsets(Nx, J)

    # Build right-hand side
    g = g_vector(subs)

    # Initialize p
    p = np.zeros(total, dtype=np.complex128)

    residuals = []

    if verbose:
        print(f"{'Iter':<6} {'||p_n+1 - p_n|| / ||p_n||':<30}")
        print("-" * 40)

    for n in range(maxiter):
        # Compute S p_n
        Sp = S_operator(p, subs)

        # Compute Π S p_n
        PiSp = Pi_operator(Sp, Nx, J)

        # Update: p_{n+1} = [(1-ω)I - ω Π S] p_n + ω g
        #                 = (1-ω) p_n - ω Π S p_n + ω g
        p_new = (1 - omega) * p - omega * PiSp + omega * g

        # Check convergence
        dp = p_new - p
        rel_err = la.norm(dp) / (la.norm(p) + 1e-14)
        residuals.append(rel_err)

        if verbose:
            print(f"{n:<6} {rel_err:<30.4e}")

        p = p_new

        if rel_err < tol:
            if verbose:
                print(f"Converged in {n+1} iterations")
            return p, {'iterations': n+1, 'residuals': residuals}

    if verbose:
        print(f"Warning: did not converge after {maxiter} iterations")

    return p, {'iterations': maxiter, 'residuals': residuals}


def gmres_interface_solver(subs: list[SubdomainData],
                           tol: float = 1e-8,
                           maxiter: int = 500,
                           verbose: bool = False):
    """
    Solve the interface problem using GMRES (Krylov subspace method).

    Solves the linear system:
        (I + Π S) p = g

    via scipy.sparse.linalg.gmres using a LinearOperator that computes
    matrix-vector products without explicitly assembling the matrix.

    Parameters
    ----------
    subs : list of SubdomainData
        Precomputed subdomain data from build_subdomains()
    tol : float
        Convergence tolerance (relative residual)
    maxiter : int
        Maximum number of GMRES iterations
    verbose : bool
        Print convergence information

    Returns
    -------
    p : (total_interface_dof,) complex
        Converged interface solution
    info : int
        0 if convergence achieved, otherwise number of iterations before breakdown
    history : dict
        'iterations': int, number of iterations performed
        'residuals': list of residual norms from callback (if available)
    """
    Nx = subs[0].Nx
    J = subs[0].J
    sizes, offsets, total = _block_offsets(Nx, J)

    # Define matrix-vector product: (I + Π S) p
    def matvec_interface_system(p_in):
        """
        Compute: (I + Π S) p
        """
        p_in = np.asarray(p_in, dtype=np.complex128).reshape(-1)
        Sp = S_operator(p_in, subs)
        PiSp = Pi_operator(Sp, Nx, J)
        return p_in + PiSp  # I*p + Π*S*p

    # Create LinearOperator
    A_op = LinearOperator(
        shape=(total, total),
        matvec=matvec_interface_system,
        dtype=np.complex128
    )

    # Build right-hand side
    g = g_vector(subs)

    # Callback for residual tracking
    residuals = []
    def callback(rk):
        residuals.append(rk)

    if verbose:
        print(f"{'Iter':<6} {'||r_k||':<20}")
        print("-" * 30)

    # Solve using GMRES
    p, info = gmres(
        A_op, g,
        atol=tol,
        maxiter=maxiter,
        callback=callback,
        callback_type='pr_norm',
        restart=100
    )

    if verbose:
        if info == 0:
            print(f"GMRES converged in {len(residuals)} iterations")
            if residuals:
                print(f"Final residual: {residuals[-1]:.4e}")
        else:
            print(f"GMRES did not converge (info={info}, iterations={len(residuals)})")
            if residuals:
                print(f"Final residual: {residuals[-1]:.4e}")

    return p, info, {'iterations': len(residuals), 'residuals': residuals}


def recover_local_solutions(subs: list[SubdomainData], 
                             p: np.ndarray) -> dict:
    """
    Recover volume solution for each subdomain given interface solution p.

    For each subdomain j, solve:
        (A_j - i B_j^* T_j B_j) u_j = B_j^* T_j p_j + b_j

    Parameters
    ----------
    subs : list of SubdomainData
        Precomputed subdomain data
    p : (total_interface_dof,) complex
        Interface solution from fixed_point_solver()

    Returns
    -------
    u_dict : dict
        Keys are subdomain indices j
        Values are (nloc_j,) complex arrays with local solution u_j
    """
    Nx = subs[0].Nx
    J = subs[0].J
    sizes, offsets, _ = _block_offsets(Nx, J)

    u_dict = {}

    for sd in subs:
        j = sd.j
        off = offsets[j]
        ns = sizes[j]

        pj = p[off:off+ns].astype(np.complex128) if ns > 0 else np.array([], dtype=np.complex128)

        # Build RHS: B_j^* T_j p_j + b_j
        rhs = sd.bj.astype(np.complex128)
        if ns > 0:
            rhs = rhs + (sd.Bj.conjugate().T @ (sd.Tj @ pj))

        # Solve: u_j = (A_j - i B_j^* T_j B_j)^{-1} rhs
        uj = sd.lu.solve(rhs) # contains precomputed factorization
        u_dict[j] = np.asarray(uj).reshape(-1)

    return u_dict


def assemble_global_solution(subs: list[SubdomainData],
                              u_dict: dict) -> np.ndarray:
    """
    Assemble global solution vector from local subdomain solutions.

    Parameters
    ----------
    subs : list of SubdomainData
    u_dict : dict from recover_local_solutions()

    Returns
    -------
    u_global : (global_ndof,) complex
        Global solution (with overlaps averaged or properly handled)
    """
    # For slab decomposition with non-overlapping interiors,
    # we can simply concatenate the local solutions in order.
    # For overlapping methods, additional averaging would be needed.

    u_parts = []
    for j in range(len(subs)):
        if j in u_dict:
            u_parts.append(u_dict[j])

    # Simple concatenation (assumes non-overlapping partitioning of interior DOFs)
    u_global = np.concatenate(u_parts)
    return u_global


def baseline_gmres_solver(Lx, Ly, Nx, Ny, kappa, ps, 
                          tol=1e-8, 
                          maxiter=500, 
                          verbose=False):
    """
    Baseline GMRES solver for the full global FEM system (no domain decomposition).
    
    Solves:
        (K - κ² M - iκ Mb) u = M f
    
    where K is stiffness, M is mass, Mb is boundary mass.
    
    Parameters
    ----------
    Lx, Ly : float
        Domain dimensions
    Nx, Ny : int
        Number of mesh points in x and y directions
    kappa : float
        Wavenumber
    ps : callable or list
        Point source function or source list
    tol : float
        GMRES convergence tolerance
    maxiter : int
        Maximum GMRES iterations
    verbose : bool
        Print convergence info
    
    Returns
    -------
    u : (Nx*Ny,) complex
        Global solution vector
    info : int
        GMRES convergence info (0 = converged)
    history : dict
        'solve_time': float, solve time in seconds
        'iterations': int, number of iterations
        'residuals': list of residual norms
    """
    from fem_core import mesh, boundary, mass, stiffness, point_source
    
    if verbose:
        print("Building global FEM system...")
    
    t0 = time.time()
    
    # Build global mesh
    vtx, elt = mesh(Nx, Ny, Lx, Ly)
    belt = boundary(Nx, Ny)
    
    # Assemble matrices
    M = mass(vtx, elt)
    Mb = mass(vtx, belt)
    K = stiffness(vtx, elt)
    
    # Global system matrix
    A_global = K - kappa**2 * M - 1j*kappa*Mb
    
    # Right-hand side
    if callable(ps):
        f = ps(vtx)
    else:
        f = point_source(ps, kappa)(vtx)
    b_global = M @ f
    
    t_build = time.time() - t0
    
    if verbose:
        print(f"  Build time: {t_build:.3f}s")
        print(f"  System size: {A_global.shape[0]} DOFs")
        print(f"  Solving with GMRES...")
    
    # Callback for residual tracking
    residuals = []
    def callback(rk):
        residuals.append(rk)
        if verbose and len(residuals) % 10 == 0:
            print(f"    Iteration {len(residuals)}: ||r|| = {rk:.4e}")
    
    t0 = time.time()
    u, info = gmres(
        A_global, b_global,
        atol=tol,
        maxiter=maxiter,
        callback=callback,
        callback_type='pr_norm',
        restart=100
    )
    t_solve = time.time() - t0
    
    if verbose:
        if info == 0:
            print(f"  GMRES converged in {len(residuals)} iterations")
            if residuals:
                print(f"  Final residual: {residuals[-1]:.4e}")
        else:
            print(f"  GMRES did not converge (info={info})")
        print(f"  Solve time: {t_solve:.3f}s")
    
    history = {
        'solve_time': t_solve,
        'build_time': t_build,
        'total_time': t_build + t_solve,
        'iterations': len(residuals),
        'residuals': [float(r) for r in residuals]
    }
    
    return u, info, history


def compare_solvers(Lx, Ly, Nx, Ny, J, kappa, ps, 
                   tol=1e-8, 
                   verbose=False):
    """
    Compare domain decomposition GMRES vs baseline full GMRES.
    
    Parameters
    ----------
    Lx, Ly : float
        Domain dimensions
    Nx, Ny : int
        Number of mesh points
    J : int
        Number of subdomains for DD method
    kappa : float
        Wavenumber
    ps : callable or list
        Point sources
    tol : float
        Convergence tolerance
    verbose : bool
        Print detailed info
    
    Returns
    -------
    comparison : dict
        Results from both methods with timing and error analysis
    """
    print("=" * 70)
    print("SOLVER COMPARISON: DD-GMRES vs BASELINE-GMRES")
    print("=" * 70)
    print(f"Domain: [{0}, {Lx}] × [{0}, {Ly}]")
    print(f"Mesh: {Nx} × {Ny} = {Nx*Ny} DOFs")
    print(f"Wavenumber κ: {kappa}")
    print(f"Subdomains (DD): {J}")
    print(f"Tolerance: {tol:.2e}")
    print()
    
    # ===== Baseline GMRES =====
    print("1. BASELINE FULL GMRES")
    print("-" * 70)
    t0_baseline = time.time()
    u_baseline, info_baseline, hist_baseline = baseline_gmres_solver(
        Lx, Ly, Nx, Ny, kappa, ps, tol=tol, verbose=verbose
    )
    t_total_baseline = time.time() - t0_baseline
    print(f"Total time: {t_total_baseline:.3f}s")
    print()
    
    # ===== Domain Decomposition GMRES =====
    print("2. DOMAIN DECOMPOSITION GMRES")
    print("-" * 70)
    from dd_operators import build_subdomains
    
    t0_dd = time.time()
    
    # Build subdomains
    if verbose:
        print("Building subdomains...")
    t0_build = time.time()
    subs = build_subdomains(Lx, Ly, Nx, Ny, J, kappa, ps)
    t_build_dd = time.time() - t0_build
    if verbose:
        print(f"  Build time: {t_build_dd:.3f}s")
    
    # Solve interface problem
    if verbose:
        print("Solving interface problem...")
    t0_solve = time.time()
    p, info_dd, hist_dd = gmres_interface_solver(subs, tol=tol, verbose=verbose)
    t_solve_dd = time.time() - t0_solve
    
    # Recover local solutions
    if verbose:
        print("Recovering local solutions...")
    t0_recovery = time.time()
    u_dict = recover_local_solutions(subs, p)
    t_recovery_dd = time.time() - t0_recovery
    
    t_total_dd = time.time() - t0_dd
    
    print(f"  Interface solve time: {t_solve_dd:.3f}s")
    print(f"  Recovery time: {t_recovery_dd:.3f}s")
    print(f"Total time: {t_total_dd:.3f}s")
    print()
    
    # ===== Comparison =====
    print("3. COMPARISON")
    print("-" * 70)
    
    # Reconstruct DD solution on global mesh
    from fem_core import mesh
    vtx_global, _ = mesh(Nx, Ny, Lx, Ly)
    u_dd_reconstructed = np.zeros(vtx_global.shape[0], dtype=np.complex128)
    
    for j, sd in enumerate(subs):
        vtxj = sd.vtxj
        uj = u_dict[j]
        
        for local_idx, local_vtx in enumerate(vtxj):
            dists = np.linalg.norm(vtx_global - local_vtx, axis=1)
            global_idx = np.argmin(dists)
            if dists[global_idx] < 1e-10:
                u_dd_reconstructed[global_idx] = uj[local_idx]
    
    # Compute error
    diff = u_baseline - u_dd_reconstructed
    rel_error = la.norm(diff) / la.norm(u_baseline)
    
    speedup = t_total_baseline / t_total_dd
    
    print(f"Baseline GMRES:")
    print(f"  Time: {t_total_baseline:.3f}s")
    print(f"  Iterations: {hist_baseline['iterations']}")
    print(f"  Solution norm: {la.norm(u_baseline):.6e}")
    print()
    
    print(f"DD GMRES:")
    print(f"  Time: {t_total_dd:.3f}s")
    print(f"  Interface iterations: {hist_dd['iterations']}")
    print(f"  Solution norm: {la.norm(u_dd_reconstructed):.6e}")
    print()
    
    print(f"Speedup: {speedup:.2f}x")
    print(f"Relative error: {rel_error:.6e}")
    print()
    
    if rel_error < 1e-6:
        print("✓ Solutions match (error < 1e-6)")
    else:
        print("⚠ Solutions differ (error > 1e-6)")
    
    print("=" * 70)
    
    return {
        'baseline': {
            'time': t_total_baseline,
            'solve_time': hist_baseline['solve_time'],
            'iterations': hist_baseline['iterations'],
            'solution': u_baseline,
            'norm': float(la.norm(u_baseline))
        },
        'dd': {
            'time': t_total_dd,
            'build_time': t_build_dd,
            'solve_time': t_solve_dd,
            'recovery_time': t_recovery_dd,
            'iterations': hist_dd['iterations'],
            'solution': u_dd_reconstructed,
            'norm': float(la.norm(u_dd_reconstructed))
        },
        'speedup': speedup,
        'relative_error': rel_error
    }

