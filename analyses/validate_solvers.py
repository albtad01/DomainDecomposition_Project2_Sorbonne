#!/usr/bin/env python3
"""
Unified validation script for domain decomposition solvers.

Compares two solution approaches:
1. Baseline direct GMRES on full global system (no domain decomposition)
2. Sequential domain decomposition (fixed-point and GMRES)

All solutions are compared to verify correctness.
"""

import numpy as np
import numpy.linalg as la
import json
from pathlib import Path
import argparse
import sys
import subprocess
import tempfile
import os

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from dd_operators import build_subdomains
from dd_solver import (
    baseline_gmres_solver, 
    fixed_point_solver,
    gmres_interface_solver,
    recover_local_solutions, 
    assemble_global_solution
)
from fem_core import mesh as build_mesh


def reconstruct_global_solution(Lx, Ly, Nx, Ny, subs, u_dict):
    """
    Reconstruct global solution on the true global mesh from local subdomain solutions.
    
    This properly handles duplicate vertices at interfaces by matching coordinates.
    """
    # Build global mesh
    vtx_global, _ = build_mesh(Nx, Ny, Lx, Ly)
    u_global = np.zeros(vtx_global.shape[0], dtype=np.complex128)
    
    # For each subdomain, map local solution to global mesh
    for j, sd in enumerate(subs):
        if j not in u_dict:
            continue
        
        vtxj = sd.vtxj
        uj = u_dict[j]
        
        # For each local vertex, find corresponding global vertex
        for local_idx, local_vtx in enumerate(vtxj):
            # Find closest global vertex (should be exact match)
            dists = np.linalg.norm(vtx_global - local_vtx, axis=1)
            global_idx = np.argmin(dists)
            
            # Verify it's an exact match
            if dists[global_idx] < 1e-10:
                u_global[global_idx] = uj[local_idx]
    
    return u_global


def run_direct_solver(Lx, Ly, Nx, Ny, kappa, ps, tol=1e-8, verbose=False):
    """
    Run the direct GMRES solver on the full global system (no domain decomposition).
    
    Returns:
        dict with u_global, iterations, solve_time, converged
    """
    if verbose:
        print("=" * 78)
        print("DIRECT GMRES SOLVER (No Domain Decomposition)")
        print("=" * 78)
    
    # Run baseline solver - it returns (u, info, history)
    u_global, info, history = baseline_gmres_solver(
        Lx, Ly, Nx, Ny, kappa, ps, tol=tol, verbose=verbose
    )
    
    result = {
        'u_global': u_global,
        'iterations': history['iterations'],
        'solve_time': history['solve_time'],
        'converged': (info == 0)
    }
    
    if verbose:
        print(f"\nDirect solver completed:")
        print(f"  Converged: {result['converged']}")
        print(f"  Solution norm: {la.norm(result['u_global']):.6e}")
        print(f"  Solution max: {np.max(np.abs(result['u_global'])):.6e}")
        print(f"  GMRES iterations: {result['iterations']}")
        print(f"  Solve time: {result['solve_time']:.3f}s")
    
    return result


def run_dd_fixed_point_solver(Lx, Ly, Nx, Ny, J, kappa, ps, 
                                tol=1e-8, maxiter=100, omega=1.0, verbose=False):
    """
    Run sequential domain decomposition fixed-point solver.
    
    Returns:
        dict with u_global, p_interface, iterations, solve_time, converged
    """
    if verbose:
        print("\n" + "=" * 78)
        print("DD FIXED-POINT SOLVER (Sequential)")
        print("=" * 78)
    
    # Build subdomains
    subs = build_subdomains(Lx, Ly, Nx, Ny, J, kappa, ps)
    
    # Solve interface problem
    p, history = fixed_point_solver(
        subs, tol=tol, maxiter=maxiter, omega=omega, verbose=verbose
    )
    
    # Recover local solutions
    u_dict = recover_local_solutions(subs, p)
    
    # Reconstruct on true global mesh
    u_global = reconstruct_global_solution(Lx, Ly, Nx, Ny, subs, u_dict)
    
    result = {
        'u_global': u_global,
        'p_interface': p,
        'iterations': history['iterations'],
        'solve_time': history.get('solve_time', 0.0),
        'converged': history['iterations'] < maxiter
    }
    
    if verbose:
        print(f"\nDD Fixed-point solver completed:")
        print(f"  Converged: {result['converged']}")
        print(f"  Solution norm: {la.norm(result['u_global']):.6e}")
        print(f"  Solution max: {np.max(np.abs(result['u_global'])):.6e}")
        print(f"  Iterations: {result['iterations']}")
        print(f"  Solve time: {result['solve_time']:.3f}s")
    
    return result


def run_dd_mpi_fixed_point_solver(Lx, Ly, Nx, Ny, J, kappa, ps, 
                                    nprocs=4, tol=1e-8, maxiter=100, omega=1.0, verbose=False):
    """
    Run parallel MPI domain decomposition fixed-point solver.
    
    This launches a separate MPI process using mpirun, saves the result to a temporary file,
    and loads it back for comparison.
    
    Args:
        nprocs: Number of MPI processes (should equal J for one subdomain per rank)
    
    Returns:
        dict with u_global, p_interface, iterations, solve_time, converged
    """
    if verbose:
        print("\n" + "=" * 78)
        print(f"DD MPI FIXED-POINT SOLVER (Parallel with {nprocs} processes)")
        print("=" * 78)
    
    # Check if mpirun is available
    try:
        subprocess.run(['which', 'mpirun'], check=True, capture_output=True)
    except subprocess.CalledProcessError:
        raise RuntimeError("mpirun not found. Please install MPI (e.g., 'brew install open-mpi')")
    
    # Create a temporary file for output
    with tempfile.NamedTemporaryFile(mode='w', suffix='.npz', delete=False) as tmp_out:
        output_file = tmp_out.name
    
    try:
        # Get the path to main.py
        main_script = SRC_DIR / "main.py"
        
        if not main_script.exists():
            raise FileNotFoundError(f"main.py not found at {main_script}")
        
        # Build command
        # Pass Nx and Ny directly to ensure consistent mesh across all solvers
        cmd = [
            'mpirun', '-n', str(nprocs),
            'python', str(main_script),
            '--algorithm', 'mpi-fixed-point',
            '--mpi',
            '--Nx', str(Nx),
            '--Ny', str(Ny),
            '--Lx', str(Lx),
            '--Ly', str(Ly),
            '--subdomains', str(J),
            '--wavenumber', str(kappa),
            '--sources', str(len(ps)),
            '--tolerance', str(tol),
            '--omega', str(omega),
            '--max-iterations', str(maxiter),
            '--output', output_file
        ]
        
        if verbose:
            cmd.append('--verbose')
        
        if verbose:
            print(f"Running MPI command: {' '.join(cmd)}")
        
        # Run MPI solver
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"MPI solver failed with return code {result.returncode}")
            print(f"STDOUT:\n{result.stdout}")
            print(f"STDERR:\n{result.stderr}")
            raise RuntimeError("MPI solver execution failed")
        
        if verbose:
            print("MPI solver output:")
            print(result.stdout)
        
        # Load results
        if not os.path.exists(output_file):
            raise FileNotFoundError(f"MPI solver did not create output file {output_file}")
        
        data = np.load(output_file)
        
        # Extract data
        u_global = data['u_global']
        p_interface = data['p_interface']
        iterations = int(data['iterations'])
        solve_time = float(data['solve_time'])
        
        result_dict = {
            'u_global': u_global,
            'p_interface': p_interface,
            'iterations': iterations,
            'solve_time': solve_time,
            'converged': iterations < maxiter,
            'nprocs': nprocs
        }
        
        if verbose:
            print(f"\nDD MPI Fixed-point solver completed:")
            print(f"  Converged: {result_dict['converged']}")
            print(f"  Solution norm: {la.norm(result_dict['u_global']):.6e}")
            print(f"  Solution max: {np.max(np.abs(result_dict['u_global'])):.6e}")
            print(f"  Iterations: {result_dict['iterations']}")
            print(f"  Solve time: {result_dict['solve_time']:.3f}s")
            print(f"  Processes: {result_dict['nprocs']}")
        
        return result_dict
        
    finally:
        # Clean up temporary file
        if os.path.exists(output_file):
            os.remove(output_file)


def run_dd_gmres_solver(Lx, Ly, Nx, Ny, J, kappa, ps, 
                         tol=1e-8, maxiter=500, verbose=False):
    """
    Run sequential domain decomposition GMRES solver.
    
    Returns:
        dict with u_global, p_interface, iterations, solve_time, converged
    """
    if verbose:
        print("\n" + "=" * 78)
        print("DD GMRES SOLVER (Sequential)")
        print("=" * 78)
    
    # Build subdomains
    subs = build_subdomains(Lx, Ly, Nx, Ny, J, kappa, ps)
    
    # Solve interface problem
    p, info, history = gmres_interface_solver(
        subs, tol=tol, maxiter=maxiter, verbose=verbose
    )
    
    # Recover local solutions
    u_dict = recover_local_solutions(subs, p)
    
    # Reconstruct on true global mesh
    u_global = reconstruct_global_solution(Lx, Ly, Nx, Ny, subs, u_dict)
    
    result = {
        'u_global': u_global,
        'p_interface': p,
        'iterations': history['iterations'],
        'solve_time': history.get('solve_time', 0.0),
        'converged': (info == 0)
    }
    
    if verbose:
        print(f"\nDD GMRES solver completed:")
        print(f"  Converged: {result['converged']}")
        print(f"  Solution norm: {la.norm(result['u_global']):.6e}")
        print(f"  Solution max: {np.max(np.abs(result['u_global'])):.6e}")
        print(f"  Iterations: {result['iterations']}")
        print(f"  Solve time: {result['solve_time']:.3f}s")
    
    return result


def compare_solutions(baseline_result, comparison_result, verbose=False):
    """
    Compare solutions from baseline and sequential solvers.
    
    Returns:
        dict: Comparison metrics
    """
    u_base = baseline_result['u_global']
    u_cmp = comparison_result['u_global']
    
    # Ensure same size
    if u_base.shape != u_cmp.shape:
        raise ValueError(f"Solution size mismatch: baseline={u_base.shape}, comparison={u_cmp.shape}")
    
    # Compute errors
    diff = u_base - u_cmp
    abs_error = np.linalg.norm(diff)
    rel_error = abs_error / np.linalg.norm(u_base)
    max_abs_error = np.max(np.abs(diff))
    max_rel_error = max_abs_error / np.max(np.abs(u_base))
    
    # Correlation (should be close to 1)
    correlation = np.corrcoef(u_base.real, u_cmp.real)[0, 1]
    
    metrics = {
        'absolute_error': abs_error,
        'relative_error': rel_error,
        'max_absolute_error': max_abs_error,
        'max_relative_error': max_rel_error,
        'correlation': correlation,
        'baseline_norm': np.linalg.norm(u_base),
        'comparison_norm': np.linalg.norm(u_cmp)
    }
    
    if verbose:
        print("\n" + "=" * 78)
        print("COMPARISON RESULTS")
        print("=" * 78)
        print(f"Baseline solution norm:     {metrics['baseline_norm']:.8e}")
        print(f"Comparison solution norm:   {metrics['comparison_norm']:.8e}")
        print(f"Absolute error (L2 norm):   {metrics['absolute_error']:.8e}")
        print(f"Relative error (L2 norm):   {metrics['relative_error']:.8e}")
        print(f"Max absolute error:         {metrics['max_absolute_error']:.8e}")
        print(f"Max relative error:         {metrics['max_relative_error']:.8e}")
        print(f"Correlation coefficient:    {metrics['correlation']:.10f}")
    
    return metrics


def print_pass_fail(name, metrics, tolerance=1e-4, verbose=False):
    """Print pass/fail status for a comparison."""
    if verbose:
        print("\n" + "=" * 78)
        print(f"VALIDATION: {name}")
        print("=" * 78)
        print(f"Tolerance: {tolerance:.0e}")
    
    passed = True
    
    if metrics['relative_error'] < tolerance:
        if verbose:
            print(f"✓ PASS: Relative error {metrics['relative_error']:.2e} < {tolerance:.0e}")
    else:
        if verbose:
            print(f"✗ FAIL: Relative error {metrics['relative_error']:.2e} >= {tolerance:.0e}")
        passed = False
    
    if metrics['correlation'] > 0.9999:
        if verbose:
            print(f"✓ PASS: Correlation {metrics['correlation']:.6f} > 0.9999")
    else:
        if verbose:
            print(f"✗ FAIL: Correlation {metrics['correlation']:.6f} <= 0.9999")
        passed = False
    
    return passed


def main():
    """Main validation routine."""
    parser = argparse.ArgumentParser(
        description="Validate domain decomposition solvers"
    )
    
    # Problem parameters
    parser.add_argument("--mesh-size", "-m", type=int, default=16,
                       help="Mesh refinement parameter (default: 16 for fast validation)")
    parser.add_argument("--Lx", type=float, default=1.0,
                       help="Domain length in x (default: 1.0)")
    parser.add_argument("--Ly", type=float, default=2.0,
                       help="Domain length in y (default: 2.0)")
    parser.add_argument("--subdomains", "-J", type=int, default=4,
                       help="Number of subdomains (default: 4)")
    parser.add_argument("--wavenumber", "-k", type=float, default=16.0,
                       help="Wavenumber (default: 16.0)")
    parser.add_argument("--sources", "-s", type=int, default=4,
                       help="Number of point sources (default: 4)")
    parser.add_argument("--tolerance", type=float, default=1e-8,
                       help="Solver tolerance (default: 1e-8)")
    parser.add_argument("--omega", type=float, default=1.0,
                       help="Fixed-point relaxation parameter (default: 1.0)")
    parser.add_argument("--max-iterations", type=int, default=200,
                       help="Maximum iterations (default: 100)")
    
    # Which solvers to test
    parser.add_argument("--test-fixed-point", action="store_true",default=False,
                       help="Test sequential fixed-point solver")
    parser.add_argument("--test-gmres", action="store_true", default=False,
                       help="Test sequential GMRES solver (default: True)")
    parser.add_argument("--test-mpi", action="store_true", default=False,
                       help="Test MPI parallel fixed-point solver")
    parser.add_argument("--nprocs", type=int, default=4,
                       help="Number of MPI processes for parallel solver (default: 4)")
    parser.add_argument("--all", action="store_true",
                       help="Test all solvers")
    
    # Execution parameters
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")
    parser.add_argument("--save-results", action="store_true",
                       help="Save comparison results to JSON")
    parser.add_argument("--comparison-tolerance", type=float, default=1e-4,
                       help="Tolerance for pass/fail validation (default: 1e-4)")
    
    args = parser.parse_args()
    
    # If --all specified, enable all tests
    if args.all:
        args.test_fixed_point = True
        args.test_gmres = True
        args.test_mpi = True
    
    # Mesh parameters - compute BEFORE running any solvers
    Nx = 1 + int(args.Lx * args.mesh_size)
    Ny_raw = 1 + int(args.Ly * args.mesh_size)
    
    # If testing domain decomposition methods, ensure (Ny-1) is divisible by J
    # This adjustment must be applied to ALL solvers for consistency
    if args.test_fixed_point or args.test_gmres or args.test_mpi:
        if (Ny_raw - 1) % args.subdomains != 0:
            # Round up to next compatible Ny
            Ny = 1 + ((Ny_raw - 1 + args.subdomains - 1) // args.subdomains) * args.subdomains
            print(f"Note: Adjusted Ny from {Ny_raw} to {Ny} for J={args.subdomains} compatibility")
        else:
            Ny = Ny_raw
    else:
        Ny = Ny_raw
    
    # Point sources (same format as main.py)
    np.random.seed(42)  # For reproducibility
    ps = [np.random.rand(3) * [args.Lx, args.Ly, 50.0] 
          for _ in range(args.sources)]
    
    print("=" * 78)
    print("DOMAIN DECOMPOSITION SOLVER VALIDATION")
    print("=" * 78)
    print(f"Configuration:")
    print(f"  Domain: [{0}, {args.Lx}] × [{0}, {args.Ly}]")
    print(f"  Mesh: Nx={Nx}, Ny={Ny} ({Nx*Ny} vertices)")
    print(f"  Subdomains: J={args.subdomains}")
    print(f"  Wavenumber: κ={args.wavenumber:.1f}")
    print(f"  Wavelength: λ={2*np.pi/args.wavenumber:.4f}")
    print(f"  Point sources: {args.sources}")
    print(f"  Tolerance: {args.tolerance:.0e}")
    
    tests_to_run = []
    if args.test_fixed_point:
        tests_to_run.append("Fixed-Point")
    if args.test_gmres:
        tests_to_run.append("GMRES")
    if args.test_mpi:
        tests_to_run.append(f"MPI Fixed-Point ({args.nprocs} procs)")
    print(f"  Tests: {', '.join(tests_to_run)}")
    print()
    
    results = {}
    all_passed = True
    
    # Run direct solver (baseline for comparison)
    try:
        print("Running baseline (direct) solver...")
        direct_result = run_direct_solver(
            args.Lx, args.Ly, Nx, Ny, args.wavenumber, ps,
            tol=args.tolerance, verbose=args.verbose
        )
        results['direct'] = direct_result
    except Exception as e:
        print(f"Direct solver failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Test sequential fixed-point
    if args.test_fixed_point:
        try:
            dd_fp_result = run_dd_fixed_point_solver(
                args.Lx, args.Ly, Nx, Ny, args.subdomains, args.wavenumber, ps,
                tol=args.tolerance, maxiter=args.max_iterations, omega=args.omega,
                verbose=args.verbose
            )
            results['dd_fixed_point'] = dd_fp_result
            
            # Compare
            metrics_fp = compare_solutions(direct_result, dd_fp_result, verbose=args.verbose)
            passed_fp = print_pass_fail("DD Fixed-Point vs Direct", metrics_fp, 
                                       tolerance=args.comparison_tolerance, verbose=True)
            all_passed = all_passed and passed_fp
            results['comparison_fixed_point'] = metrics_fp
        except Exception as e:
            print(f"DD Fixed-Point solver failed: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
    
    # Test sequential GMRES
    if args.test_gmres:
        try:
            dd_gmres_result = run_dd_gmres_solver(
                args.Lx, args.Ly, Nx, Ny, args.subdomains, args.wavenumber, ps,
                tol=args.tolerance, maxiter=args.max_iterations,
                verbose=args.verbose
            )
            results['dd_gmres'] = dd_gmres_result
            
            # Compare
            metrics_gmres = compare_solutions(direct_result, dd_gmres_result, verbose=args.verbose)
            passed_gmres = print_pass_fail("DD GMRES vs Direct", metrics_gmres, 
                                          tolerance=args.comparison_tolerance, verbose=True)
            all_passed = all_passed and passed_gmres
            results['comparison_gmres'] = metrics_gmres
        except Exception as e:
            print(f"DD GMRES solver failed: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
    
    # Test MPI parallel fixed-point
    if args.test_mpi:
        try:
            dd_mpi_result = run_dd_mpi_fixed_point_solver(
                args.Lx, args.Ly, Nx, Ny, args.subdomains, args.wavenumber, ps,
                nprocs=args.nprocs,
                tol=args.tolerance, maxiter=args.max_iterations, omega=args.omega,
                verbose=args.verbose
            )
            results['dd_mpi_fixed_point'] = dd_mpi_result
            
            # Compare with baseline
            metrics_mpi = compare_solutions(direct_result, dd_mpi_result, verbose=args.verbose)
            passed_mpi = print_pass_fail("DD MPI Fixed-Point vs Direct", metrics_mpi, 
                                        tolerance=args.comparison_tolerance, verbose=True)
            all_passed = all_passed and passed_mpi
            results['comparison_mpi'] = metrics_mpi
            
            # Also compare with sequential fixed-point if available
            if 'dd_fixed_point' in results:
                print("\n" + "=" * 78)
                print("COMPARING: MPI Fixed-Point vs Sequential Fixed-Point")
                print("=" * 78)
                metrics_mpi_vs_seq = compare_solutions(results['dd_fixed_point'], dd_mpi_result, verbose=True)
                passed_mpi_vs_seq = print_pass_fail("MPI vs Sequential Fixed-Point", metrics_mpi_vs_seq, 
                                                    tolerance=args.comparison_tolerance, verbose=True)
                all_passed = all_passed and passed_mpi_vs_seq
                results['comparison_mpi_vs_sequential'] = metrics_mpi_vs_seq
                
        except Exception as e:
            print(f"DD MPI Fixed-Point solver failed: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
    
    
    # Save results if requested
    if args.save_results:
        results_dict = {
            'configuration': {
                'Lx': args.Lx,
                'Ly': args.Ly,
                'Nx': Nx,
                'Ny': Ny,
                'J': args.subdomains,
                'kappa': args.wavenumber,
                'num_sources': args.sources,
                'tolerance': args.tolerance,
                'omega': args.omega
            }
        }
        
        # Add results (convert numpy types to python types)
        for key, value in results.items():
            if isinstance(value, dict):
                results_dict[key] = {
                    k: (float(v) if isinstance(v, (np.floating, float, np.integer, int)) else 
                        int(v) if isinstance(v, (np.integer, int)) else v)
                    for k, v in value.items()
                    if k != 'u_global' and k != 'p_interface'  # Skip large arrays
                }
        
        output_file = Path("validation_results.json")
        with open(output_file, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"\n✓ Results saved to {output_file}")
    
    # Final summary
    print("\n" + "=" * 78)
    print("FINAL VALIDATION SUMMARY")
    print("=" * 78)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
