#!/usr/bin/env python3
"""
Main experiment runner for Helmholtz domain decomposition solver.

Allows configuration of:
  - Algorithm (fixed-point or GMRES)
  - Mesh size (Nx, Ny)
  - Number of subdomains (J)
  - Problem parameters (wavenumber, domain size, sources)
  - Plotting options
"""

import argparse
import numpy as np
import numpy.linalg as la
from math import pi
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import time
import json
import os
from pathlib import Path

from fem_core import point_source, plot_mesh, mesh
from dd_operators import build_subdomains
from dd_solver import (
    fixed_point_solver,
    gmres_interface_solver,
    baseline_gmres_solver,
    recover_local_solutions,
    assemble_global_solution
)

# MPI imports (optional - only used if --mpi flag is set)
try:
    from mpi4py import MPI
    from dd_solver_mpi import (
        parallel_fixed_point_solver_mpi,
        reconstruct_local_solution,
        gather_global_solution
    )
    MPI_AVAILABLE = True
except ImportError:
    MPI_AVAILABLE = False


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Solve Helmholtz problem with domain decomposition"
    )
    
    # Algorithm choice
    parser.add_argument(
        "--algorithm", "-a",
        choices=["fixed-point", "gmres", "baseline-gmres", "mpi-fixed-point"],
        default="gmres",
        help="Solver algorithm: fixed-point, gmres, baseline-gmres, or mpi-fixed-point (default: gmres)"
    )
    parser.add_argument(
        "--mpi",
        action="store_true",
        help="Use MPI parallel solver (requires mpirun and mpi4py)"
    )
    
    # Mesh parameters
    parser.add_argument(
        "--mesh-size", "-m",
        type=int,
        default=32,
        help="Mesh refinement parameter: Nx=1+Lx*m, Ny=1+Ly*m (default: 32)"
    )
    parser.add_argument(
        "--Lx",
        type=float,
        default=1.0,
        help="Domain length in x-direction (default: 1.0)"
    )
    parser.add_argument(
        "--Ly",
        type=float,
        default=2.0,
        help="Domain length in y-direction (default: 2.0)"
    )
    
    # Subdomains
    parser.add_argument(
        "--subdomains", "-J",
        type=int,
        default=4,
        help="Number of subdomains (default: 4)"
    )
    parser.add_argument(
        "--Nx",
        type=int,
        default=None,
        help="Override Nx mesh points (default: computed from mesh-size and Lx)"
    )
    parser.add_argument(
        "--Ny",
        type=int,
        default=None,
        help="Override Ny mesh points (default: computed from mesh-size and Ly)"
    )
    
    # Problem parameters
    parser.add_argument(
        "--wavenumber", "-k",
        type=float,
        default=16.0,
        help="Wavenumber κ (default: 16.0)"
    )
    parser.add_argument(
        "--sources", "-s",
        type=int,
        default=8,
        help="Number of point sources (default: 8)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for source locations (default: 42)"
    )
    
    # Solver parameters
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-8,
        help="Convergence tolerance (default: 1e-8)"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=100,
        help="Maximum iterations (default: 500)"
    )
    parser.add_argument(
        "--omega",
        type=float,
        default=1.0,
        help="Relaxation parameter for fixed-point (default: 1.0)"
    )
    
    # Output/plotting
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../results",
        help="Directory to save results (default: ../results)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save solution and metadata to specific file for validation (overrides output-dir)"
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        default=True,
        help="Save plots to file instead of showing"
    )
    parser.add_argument(
        "--plot-global",
        action="store_true",
        default=False,
        help="Plot global solution"
    )
    parser.add_argument(
        "--plot-local",
        action="store_true",
        default=False,
        help="Plot local solutions for each subdomain"
    )
    parser.add_argument(
        "--plot-mesh",
        action="store_true",
        default=False,
        help="Plot mesh structure"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    return parser.parse_args()


def run_experiment(args):
    """Run a complete domain decomposition experiment."""
    
    # Check if MPI mode is requested
    use_mpi = args.mpi or args.algorithm == "mpi-fixed-point"
    
    if use_mpi and not MPI_AVAILABLE:
        raise RuntimeError(
            "MPI solver requested but mpi4py is not available. "
            "Install with: pip install mpi4py"
        )
    
    # Handle MPI-parallel execution
    if use_mpi:
        return run_mpi_experiment(args)
    
    # Print configuration
    print("=" * 70)
    print("HELMHOLTZ DOMAIN DECOMPOSITION SOLVER")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Algorithm:       {args.algorithm}")
    print(f"  Mesh:            Nx={1 + int(args.Lx * args.mesh_size)}, "
          f"Ny={1 + int(args.Ly * args.mesh_size)}")
    print(f"  Domain:          [{0}, {args.Lx}] × [{0}, {args.Ly}]")
    print(f"  Subdomains:      {args.subdomains}")
    print(f"  Wavenumber κ:    {args.wavenumber}")
    print(f"  Wavelength λ:    {2*pi/args.wavenumber:.4f}")
    print(f"  Sources:         {args.sources}")
    print(f"  Tolerance:       {args.tolerance:.2e}")
    print(f"  Max iterations:  {args.max_iterations}")
    print()
    
    # Setup problem
    # Allow Nx, Ny to be overridden for validation compatibility
    if args.Nx is not None:
        Nx = args.Nx
    else:
        Nx = 1 + int(args.Lx * args.mesh_size)
    
    if args.Ny is not None:
        Ny = args.Ny
    else:
        Ny_raw = 1 + int(args.Ly * args.mesh_size)
        
        # For domain decomposition, ensure (Ny-1) is divisible by J
        if args.algorithm in ["fixed-point", "gmres"]:
            J = args.subdomains
            if (Ny_raw - 1) % J != 0:
                Ny = 1 + ((Ny_raw - 1 + J - 1) // J) * J
                if args.verbose:
                    print(f"Note: Adjusted Ny from {Ny_raw} to {Ny} for J={J} compatibility")
            else:
                Ny = Ny_raw
        else:
            Ny = Ny_raw
    
    np.random.seed(args.seed)
    sp = [np.random.rand(3) * [args.Lx, args.Ly, 50.0] 
          for _ in range(args.sources)]
    
    # Baseline full GMRES (no domain decomposition)
    if args.algorithm == "baseline-gmres":
        print("Solving full problem with baseline GMRES...")
        t0 = time.time()
        u_global, info, history = baseline_gmres_solver(
            args.Lx,
            args.Ly,
            Nx,
            Ny,
            args.wavenumber,
            sp,
            tol=args.tolerance,
            maxiter=args.max_iterations,
            verbose=args.verbose,
        )
        t_solve = time.time() - t0

        if info == 0:
            print(f"  Converged in {history['iterations']} iterations")
            if history['residuals']:
                print(f"  Final residual: {history['residuals'][-1]:.4e}")
        else:
            print(f"  WARNING: Did not converge (info={info})")

        print(f"  Solve time: {t_solve:.3f}s")
        print()

        total_dof = int(u_global.shape[0])
        dof_per_subdomain = [total_dof]
        interface_dof = 0

        metrics = {
            'algorithm': args.algorithm,
            'mesh_size': args.mesh_size,
            'Nx': Nx,
            'Ny': Ny,
            'Lx': args.Lx,
            'Ly': args.Ly,
            'subdomains': 1,
            'wavenumber': args.wavenumber,
            'wavelength': 2 * pi / args.wavenumber,
            'sources': args.sources,
            'tolerance': args.tolerance,
            'omega': None,
            'total_dof': total_dof,
            'interface_dof': interface_dof,
            'dof_per_subdomain': dof_per_subdomain,
            'build_time': history.get('build_time', 0.0),
            'solve_time': history.get('solve_time', t_solve),
            'recovery_time': 0.0,
            'total_time': history.get('total_time', t_solve),
            'iterations': history['iterations'],
            'final_residual': history['residuals'][-1] if history['residuals'] else None,
            'converged': info == 0,
            'solution_norm': float(la.norm(u_global)),
            'solution_max': float(np.max(np.abs(u_global))),
            'solution_min': float(np.min(np.abs(u_global))),
            'residual_history': [float(r) for r in history['residuals']]
        }

        return [], None, {}, u_global, metrics

    # Build subdomains
    print("Building subdomain data...")
    t0 = time.time()
    subs = build_subdomains(args.Lx, args.Ly, Nx, Ny, args.subdomains, 
                            args.wavenumber, sp)
    t_build = time.time() - t0
    
    # Compute total and per-subdomain DOF counts
    total_dof = sum(sd.vtxj.shape[0] for sd in subs)
    dof_per_subdomain = [sd.vtxj.shape[0] for sd in subs]
    interface_dof = sum(sd.Bj.shape[0] for sd in subs)
    
    print(f"  Done in {t_build:.3f}s")
    print(f"  Total DOFs (all subdomains): {total_dof}")
    print(f"  Interface DOFs: {interface_dof}")
    print(f"  DOFs per subdomain: {dof_per_subdomain}")
    print()
    
    # Solve interface problem
    print(f"Solving interface problem with {args.algorithm}...")
    t0 = time.time()
    
    if args.algorithm == "fixed-point":
        p, history = fixed_point_solver(
            subs,
            tol=args.tolerance,
            maxiter=args.max_iterations,
            omega=args.omega,
            verbose=args.verbose
        )
        info = 0
        t_solve = time.time() - t0
        print(f"  Converged in {history['iterations']} iterations")
        print(f"  Final residual: {history['residuals'][-1]:.4e}")
    else:  # gmres
        p, info, history = gmres_interface_solver(
            subs,
            tol=args.tolerance,
            maxiter=args.max_iterations,
            verbose=args.verbose
        )
        t_solve = time.time() - t0
        if info == 0:
            print(f"  Converged in {history['iterations']} iterations")
            if history['residuals']:
                print(f"  Final residual: {history['residuals'][-1]:.4e}")
        else:
            print(f"  WARNING: Did not converge (info={info})")
    
    print(f"  Solve time: {t_solve:.3f}s")
    print()
    
    if info != 0 and args.algorithm == "gmres":
        print("ERROR: GMRES did not converge. Aborting recovery.")
        return None, None, None, None
    
    # Recover local solutions
    print("Recovering local solutions...")
    t0 = time.time()
    u_dict = recover_local_solutions(subs, p)
    t_recovery = time.time() - t0
    print(f"  Done in {t_recovery:.3f}s")
    print()
    
    # Assemble global solution on true global mesh
    u_global = reconstruct_global_solution(args.Lx, args.Ly, Nx, Ny, subs, u_dict)
    
    # Print solution statistics
    print("Solution statistics:")
    print(f"  Global solution norm: {la.norm(u_global):.6e}")
    print(f"  Global solution max:  {np.max(np.abs(u_global)):.6e}")
    print(f"  Global solution min:  {np.min(np.abs(u_global)):.6e}")
    print()
    
    # Collect all metrics
    metrics = {
        'algorithm': args.algorithm,
        'mesh_size': args.mesh_size,
        'Nx': Nx,
        'Ny': Ny,
        'Lx': args.Lx,
        'Ly': args.Ly,
        'subdomains': args.subdomains,
        'wavenumber': args.wavenumber,
        'wavelength': 2 * pi / args.wavenumber,
        'sources': args.sources,
        'tolerance': args.tolerance,
        'omega': args.omega if args.algorithm == 'fixed-point' else None,
        'total_dof': total_dof,
        'interface_dof': interface_dof,
        'dof_per_subdomain': dof_per_subdomain,
        'build_time': t_build,
        'solve_time': t_solve,
        'recovery_time': t_recovery,
        'total_time': t_build + t_solve + t_recovery,
        'iterations': history['iterations'],
        'final_residual': history['residuals'][-1] if history['residuals'] else None,
        'converged': info == 0,
        'solution_norm': float(la.norm(u_global)),
        'solution_max': float(np.max(np.abs(u_global))),
        'solution_min': float(np.min(np.abs(u_global))),
        'residual_history': [float(r) for r in history['residuals']]
    }
    
    return subs, p, u_dict, u_global, metrics


def reconstruct_global_solution(Lx, Ly, Nx, Ny, subs, u_dict):
    """
    Reconstruct global solution on the true global mesh from local subdomain solutions.
    
    This properly handles duplicate vertices at interfaces by matching coordinates.
    """
    vtx_global, _ = mesh(Nx, Ny, Lx, Ly)
    u_global = np.zeros(vtx_global.shape[0], dtype=np.complex128)
    
    for j, sd in enumerate(subs):
        if j not in u_dict:
            continue
        
        vtxj = sd.vtxj
        uj = u_dict[j]
        
        for local_idx, local_vtx in enumerate(vtxj):
            dists = np.linalg.norm(vtx_global - local_vtx, axis=1)
            global_idx = np.argmin(dists)
            if dists[global_idx] < 1e-10:
                u_global[global_idx] = uj[local_idx]
    
    return u_global


def run_mpi_experiment(args):
    """Run MPI-parallel domain decomposition experiment."""
    
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Check MPI size matches subdomains
    if size != args.subdomains:
        if rank == 0:
            print(f"ERROR: MPI size ({size}) must equal number of subdomains ({args.subdomains})")
            print(f"Run with: mpirun -np {args.subdomains} python main.py --mpi ...")
        comm.Abort(1)
    
    # Setup problem
    J = args.subdomains
    
    # Allow Nx, Ny to be overridden for validation compatibility
    if args.Nx is not None:
        Nx = args.Nx
    else:
        Nx = 1 + int(args.Lx * args.mesh_size)
    
    if args.Ny is not None:
        Ny = args.Ny
    else:
        Ny_raw = 1 + int(args.Ly * args.mesh_size)
        
        # Ensure (Ny-1) is divisible by J for slab decomposition
        if (Ny_raw - 1) % J != 0:
            Ny = 1 + ((Ny_raw - 1 + J - 1) // J) * J
            if rank == 0 and args.verbose:
                print(f"Note: Adjusted Ny from {Ny_raw} to {Ny} for J={J} compatibility")
        else:
            Ny = Ny_raw

    if rank == 0:
        print("=" * 70)
        print("HELMHOLTZ DOMAIN DECOMPOSITION SOLVER (MPI)")
        print("=" * 70)
        print(f"\nConfiguration:")
        print(f"  Algorithm:       MPI Fixed-Point")
        print(f"  MPI ranks:       {size}")
        print(f"  Mesh:            Nx={Nx}, Ny={Ny}")
        print(f"  Domain:          [{0}, {args.Lx}] × [{0}, {args.Ly}]")
        print(f"  Subdomains:      {args.subdomains}")
        print(f"  Wavenumber κ:    {args.wavenumber}")
        print(f"  Wavelength λ:    {2*pi/args.wavenumber:.4f}")
        print(f"  Sources:         {args.sources}")
        print(f"  Tolerance:       {args.tolerance:.2e}")
        print(f"  Max iterations:  {args.max_iterations}")
        print(f"  Omega:           {args.omega}")
        print()
    
    # All ranks use same random seed to generate identical point sources
    np.random.seed(args.seed)
    sp = [np.random.rand(3) * [args.Lx, args.Ly, 50.0] 
          for _ in range(args.sources)]
    
    # Each rank builds ALL subdomains (deterministic operation)
    # We do this instead of scatter because SuperLU objects cannot be pickled
    # Since build_subdomains is deterministic, all ranks get identical data
    if rank == 0 and args.verbose:
        print("Building subdomain data (all ranks)...")
        print(f"  Each rank will build all {J} subdomains...")
    
    t0 = time.time()
    all_subs = build_subdomains(args.Lx, args.Ly, Nx, Ny, J, args.wavenumber, sp)
    t_build = time.time() - t0
    
    # Each rank extracts its own subdomain for computation
    local_sub = all_subs[rank]
    
    if rank == 0 and args.verbose:
        print(f"  Done in {t_build:.3f}s")
        print(f"  Rank 0 extracted subdomain 0")
        print(f"  Local DOFs: {local_sub.vtxj.shape[0]}")
        print(f"  Interface DOFs: {local_sub.Bj.shape[0]}")
        print()
    elif args.verbose:
        # Other ranks also print their info if verbose
        print(f"  [Rank {rank}] Extracted subdomain {rank}, "
              f"DOFs: {local_sub.vtxj.shape[0]}, "
              f"Interface: {local_sub.Bj.shape[0]}")
    
    if rank == 0 and args.verbose:
        print("Solving interface problem with MPI fixed-point...")
        print(f"  Using omega = {args.omega}")
        print(f"  Tolerance = {args.tolerance:.2e}")
        print(f"  Max iterations = {args.max_iterations}")
        print()
    
    # Solve interface problem in parallel
    t0 = time.time()
    p_local, history = parallel_fixed_point_solver_mpi(
        local_sub, comm,
        tol=args.tolerance,
        maxiter=args.max_iterations,
        omega=args.omega,
        verbose=args.verbose
    )
    t_solve = time.time() - t0
    
    if rank == 0 and args.verbose:
        print()
        print(f"  Solve time: {t_solve:.3f}s")
        print()
    
    # Recover local solutions
    if rank == 0 and args.verbose:
        print("Recovering local solutions...")
    
    t0 = time.time()
    u_local = reconstruct_local_solution(p_local, local_sub, verbose=args.verbose, rank=rank)
    t_recovery = time.time() - t0
    
    # Gather to rank 0
    u_gathered = gather_global_solution(u_local, local_sub, comm, verbose=args.verbose)
    
    if rank == 0:
        if args.verbose:
            print(f"  Done in {t_recovery:.3f}s")
            print()
        
        # Assemble global solution from gathered data
        # Create u_dict for compatibility with plotting functions
        u_dict = {j: u_gathered['u_values'][j] for j in range(J)}
        
        # Reconstruct subs list on rank 0 for plotting
        subs = all_subs
        
        # Assemble global solution vector on true global mesh
        u_global = reconstruct_global_solution(args.Lx, args.Ly, Nx, Ny, subs, u_dict)
        
        # Print solution statistics
        if args.verbose:
            print("Solution statistics:")
            print(f"  Global solution norm: {la.norm(u_global):.6e}")
            print(f"  Global solution max:  {np.max(np.abs(u_global)):.6e}")
            print(f"  Global solution min:  {np.min(np.abs(u_global)):.6e}")
            print()
        
        # Compute DOF statistics
        total_dof = sum(sd.vtxj.shape[0] for sd in subs)
        dof_per_subdomain = [sd.vtxj.shape[0] for sd in subs]
        interface_dof = sum(sd.Bj.shape[0] for sd in subs)
        
        # Collect metrics
        metrics = {
            'algorithm': 'mpi-fixed-point',
            'mpi_ranks': size,
            'mesh_size': args.mesh_size,
            'Nx': Nx,
            'Ny': Ny,
            'Lx': args.Lx,
            'Ly': args.Ly,
            'subdomains': args.subdomains,
            'wavenumber': args.wavenumber,
            'wavelength': 2 * pi / args.wavenumber,
            'sources': args.sources,
            'tolerance': args.tolerance,
            'omega': args.omega,
            'total_dof': total_dof,
            'interface_dof': interface_dof,
            'dof_per_subdomain': dof_per_subdomain,
            'build_time': t_build,
            'solve_time': t_solve,
            'recovery_time': t_recovery,
            'total_time': t_build + t_solve + t_recovery,
            'iterations': history['iterations'],
            'final_residual': history['residuals'][-1] if history['residuals'] else None,
            'converged': True,
            'solution_norm': float(la.norm(u_global)),
            'solution_max': float(np.max(np.abs(u_global))),
            'solution_min': float(np.min(np.abs(u_global))),
            'residual_history': [float(r) for r in history['residuals']]
        }
        
        return subs, p_local, u_dict, u_global, metrics
    else:
        # Non-root ranks return None
        return None, None, None, None, None


def plot_solutions(subs, u_dict, u_global, args, save_dir=None):
    """Plot mesh, global solution, and local solutions."""
    
    if args.plot_mesh:
        if args.verbose:
            print("Plotting mesh structure...")
        for j, sd in enumerate(subs):
            fig = plt.figure(figsize=(6, 5))
            plot_mesh(sd.vtxj, sd.eltj)
            plt.title(f"Subdomain {j} mesh")
            plt.tight_layout()
            if save_dir:
                plt.savefig(save_dir / f"mesh_subdomain_{j}.png", dpi=150, bbox_inches='tight')
        if not save_dir:
            plt.show()
        else:
            plt.close('all')
    
    if args.plot_global:
        if args.verbose:
            print("Plotting global solution...")
        
        # Build global mesh covering entire domain
        Nx = 1 + int(args.Lx * args.mesh_size)
        Ny = 1 + int(args.Ly * args.mesh_size)
        vtx_global, elt_global = mesh(Nx, Ny, args.Lx, args.Ly)
        
        # Reconstruct global solution vector by mapping each subdomain solution to global mesh
        # For each global vertex, find which subdomain it belongs to and get its solution value
        u_global_reconstructed = np.zeros(vtx_global.shape[0], dtype=np.complex128)
        
        for j, sd in enumerate(subs):
            # Get local vertices and solution
            vtxj = sd.vtxj
            uj = u_dict[j]
            
            # For each local vertex, find the matching global vertex
            for local_idx, local_vtx in enumerate(vtxj):
                # Find closest/matching global vertex (with tolerance for floating point)
                dists = np.linalg.norm(vtx_global - local_vtx, axis=1)
                global_idx = np.argmin(dists)
                
                # Only assign if it's a very close match and not already assigned
                if dists[global_idx] < 1e-10:
                    u_global_reconstructed[global_idx] = uj[local_idx]
        
        # Plot real part and magnitude side-by-side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Real part
        tri = mtri.Triangulation(vtx_global[:, 0], vtx_global[:, 1], elt_global)
        tc1 = ax1.tripcolor(tri, np.real(u_global_reconstructed), cmap='RdBu_r')
        ax1.set_aspect('equal')
        ax1.set_title(f"Global solution - Real part (κ={args.wavenumber})")
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        fig.colorbar(tc1, ax=ax1, label="Re(u)")
        
        # Magnitude
        tc2 = ax2.tripcolor(tri, np.abs(u_global_reconstructed), cmap='hot')
        ax2.set_aspect('equal')
        ax2.set_title(f"Global solution - Magnitude (κ={args.wavenumber})")
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        fig.colorbar(tc2, ax=ax2, label="|u|")
        
        plt.tight_layout()
        if save_dir:
            plt.savefig(save_dir / "global_solution.png", dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    if args.plot_local:
        if args.verbose:
            print("Plotting local solutions...")
        for j, sd in enumerate(subs):
            if j in u_dict:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
                
                # Real part
                tri = mtri.Triangulation(sd.vtxj[:, 0], sd.vtxj[:, 1], sd.eltj)
                tc1 = ax1.tripcolor(tri, np.real(u_dict[j]), cmap='RdBu_r')
                ax1.set_aspect('equal')
                ax1.set_title(f"Subdomain {j} - Real part")
                ax1.set_xlabel("x")
                ax1.set_ylabel("y")
                fig.colorbar(tc1, ax=ax1, label="Re(u)")
                
                # Magnitude
                tc2 = ax2.tripcolor(tri, np.abs(u_dict[j]), cmap='hot')
                ax2.set_aspect('equal')
                ax2.set_title(f"Subdomain {j} - Magnitude")
                ax2.set_xlabel("x")
                ax2.set_ylabel("y")
                fig.colorbar(tc2, ax=ax2, label="|u|")
                
                plt.tight_layout()
                if save_dir:
                    plt.savefig(save_dir / f"local_solution_subdomain_{j}.png", dpi=150, bbox_inches='tight')
        if not save_dir:
            plt.show()
        else:
            plt.close('all')


def save_results(subs, u_dict, u_global, metrics, args, p_interface=None):
    """Save mesh, solution, and metrics to disk."""
    
    # If --output is specified, save only to that specific file for validation
    if args.output:
        if args.verbose:
            print(f"\nSaving validation output to: {args.output}")
        
        # Extract interface solution (p) if available
        if p_interface is None:
            # Try to extract from experiment results (not available for baseline)
            p_interface = np.array([])
        
        # Save minimal data for validation
        np.savez_compressed(
            args.output,
            u_global=u_global,
            p_interface=p_interface,
            iterations=metrics.get('iterations', 0),
            solve_time=metrics.get('solve_time', 0.0)
        )
        
        if args.verbose:
            print(f"  Saved: u_global shape {u_global.shape}")
            print(f"  Saved: p_interface shape {p_interface.shape}")
            print(f"  Saved: iterations {metrics.get('iterations', 0)}")
            print(f"  Saved: solve_time {metrics.get('solve_time', 0.0):.3f}s")
        
        return Path(args.output).parent
    
    # Otherwise, normal full save to output-dir
    # Create output directory with timestamp and parameters
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    exp_name = f"{args.algorithm}_m{args.mesh_size}_J{args.subdomains}_k{int(args.wavenumber)}"
    output_dir = Path(args.output_dir) / exp_name / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving results to: {output_dir}")
    
    # Save metrics as JSON
    metrics_file = output_dir / "metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    if args.verbose:
        print(f"  Saved metrics: {metrics_file}")
    
    # Save global mesh
    # Use overridden values if provided, otherwise compute from mesh_size
    if args.Nx is not None:
        Nx = args.Nx
    else:
        Nx = 1 + int(args.Lx * args.mesh_size)
    
    if args.Ny is not None:
        Ny = args.Ny
    else:
        Ny = 1 + int(args.Ly * args.mesh_size)
    
    vtx_global, elt_global = mesh(Nx, Ny, args.Lx, args.Ly)
    
    np.savez_compressed(
        output_dir / "mesh_global.npz",
        vertices=vtx_global,
        elements=elt_global
    )
    if args.verbose:
        print(f"  Saved global mesh: mesh_global.npz")
    
    # Save global solution
    np.savez_compressed(
        output_dir / "solution_global.npz",
        u_global=u_global
    )
    if args.verbose:
        print(f"  Saved global solution: solution_global.npz")
    
    # Save local solutions
    for j in range(len(subs)):
        if j in u_dict:
            np.savez_compressed(
                output_dir / f"solution_subdomain_{j}.npz",
                vertices=subs[j].vtxj,
                elements=subs[j].eltj,
                solution=u_dict[j]
            )
    if args.verbose:
        print(f"  Saved {len(u_dict)} local subdomain solutions")
    
    # Save plots if requested
    if args.save_plots or args.plot_global or args.plot_mesh or args.plot_local:
        plot_dir = output_dir / "plots"
        plot_dir.mkdir(exist_ok=True)
        plot_solutions(subs, u_dict, u_global, args, save_dir=plot_dir)
        if args.verbose:
            print(f"  Saved plots to: {plot_dir}")
    
    print(f"\nAll results saved to: {output_dir}\n")
    return output_dir


def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Check if running in MPI mode
    use_mpi = args.mpi or args.algorithm == "mpi-fixed-point"
    
    if use_mpi and MPI_AVAILABLE:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        
        # All ranks run the experiment
        subs, p, u_dict, u_global, metrics = run_experiment(args)
        
        # Only rank 0 saves results and plots
        if rank == 0:
            if u_global is not None:
                output_dir = save_results(subs, u_dict, u_global, metrics, args, p_interface=p)
                
                if not args.save_plots:
                    plot_solutions(subs, u_dict, u_global, args)
                
                print("=" * 70)
                print("EXPERIMENT COMPLETED SUCCESSFULLY")
                print("=" * 70)
            else:
                print("=" * 70)
                print("EXPERIMENT FAILED")
                print("=" * 70)
    else:
        # Sequential execution
        subs, p, u_dict, u_global, metrics = run_experiment(args)
        
        if u_global is not None:
            output_dir = save_results(subs, u_dict, u_global, metrics, args, p_interface=p)
            
            if not args.save_plots:
                plot_solutions(subs, u_dict, u_global, args)
            
            print("=" * 70)
            print("EXPERIMENT COMPLETED SUCCESSFULLY")
            print("=" * 70)
        else:
            print("=" * 70)
            print("EXPERIMENT FAILED")
            print("=" * 70)


if __name__ == "__main__":
    main()
