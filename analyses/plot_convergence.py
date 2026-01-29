#!/usr/bin/env python3
"""
Plot convergence curves (residual history) for GMRES vs fixed-point.

This script finds matching results by experiment parameters and overlays
residual histories in a single plot. It does not modify any results.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


PARAM_KEYS = [
    "mesh_size",
    "Nx",
    "Ny",
    "Lx",
    "Ly",
    "subdomains",
    "wavenumber",
    "sources",
    "tolerance",
]


def load_metrics(metrics_path: Path) -> Dict[str, Any]:
    with open(metrics_path, "r") as f:
        return json.load(f)


def check_same_experiment(gmres: Dict[str, Any], fixed: Dict[str, Any]) -> List[str]:
    mismatches = []
    for key in PARAM_KEYS:
        if key in gmres and key in fixed:
            if gmres[key] != fixed[key]:
                mismatches.append(f"{key}: gmres={gmres[key]} vs fixed={fixed[key]}")
    return mismatches


def find_metrics(
    results_dir: Path,
    algorithm: str,
    m: int,
    J: int,
    kappa: float,
    omega: Optional[float] = None,
) -> Optional[Path]:
    candidates: List[Tuple[float, Path]] = []
    for metrics_path in results_dir.rglob("metrics.json"):
        try:
            data = load_metrics(metrics_path)
        except Exception:
            continue

        if data.get("algorithm") != algorithm:
            continue
        if data.get("mesh_size") != m:
            continue
        if data.get("subdomains") != J:
            continue
        if float(data.get("wavenumber")) != float(kappa):
            continue
        if omega is not None and data.get("omega") != omega:
            continue

        candidates.append((metrics_path.stat().st_mtime, metrics_path))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def find_fixed_metrics_by_omega(
    results_dir: Path,
    m: int,
    J: int,
    kappa: float,
) -> Dict[float, Path]:
    latest_by_omega: Dict[float, Tuple[float, Path]] = {}

    for metrics_path in results_dir.rglob("metrics.json"):
        try:
            data = load_metrics(metrics_path)
        except Exception:
            continue

        if data.get("algorithm") != "fixed-point":
            continue
        if data.get("mesh_size") != m:
            continue
        if data.get("subdomains") != J:
            continue
        if float(data.get("wavenumber")) != float(kappa):
            continue

        omega = data.get("omega")
        if omega is None:
            continue

        mtime = metrics_path.stat().st_mtime
        if omega not in latest_by_omega or mtime > latest_by_omega[omega][0]:
            latest_by_omega[omega] = (mtime, metrics_path)

    return {omega: path for omega, (_, path) in latest_by_omega.items()}


def find_metrics_by_mesh_size(
    results_dir: Path,
    algorithm: str,
    J: int,
    kappa: float,
    omega: Optional[float] = None,
) -> Dict[int, Path]:
    latest_by_m: Dict[int, Tuple[float, Path]] = {}

    for metrics_path in results_dir.rglob("metrics.json"):
        try:
            data = load_metrics(metrics_path)
        except Exception:
            continue

        if data.get("algorithm") != algorithm:
            continue
        if data.get("subdomains") != J:
            continue
        if float(data.get("wavenumber")) != float(kappa):
            continue
        if algorithm == "fixed-point" and omega is not None and data.get("omega") != omega:
            continue

        m = data.get("mesh_size")
        if m is None:
            continue

        mtime = metrics_path.stat().st_mtime
        if m not in latest_by_m or mtime > latest_by_m[m][0]:
            latest_by_m[m] = (mtime, metrics_path)

    return {m: path for m, (_, path) in latest_by_m.items()}


def find_metrics_by_subdomains_strong(
    results_dir: Path,
    algorithm: str,
    m: int,
    kappa: float,
    omega: Optional[float] = None,
) -> Dict[int, Path]:
    """Find metrics for strong scaling: fixed mesh size m, varying J."""
    latest_by_J: Dict[int, Tuple[float, Path]] = {}

    for metrics_path in results_dir.rglob("metrics.json"):
        try:
            data = load_metrics(metrics_path)
        except Exception:
            continue

        if data.get("algorithm") != algorithm:
            continue
        if data.get("mesh_size") != m:
            continue
        if float(data.get("wavenumber")) != float(kappa):
            continue
        if algorithm == "fixed-point" and omega is not None and data.get("omega") != omega:
            continue

        J = data.get("subdomains")
        if J is None:
            continue

        mtime = metrics_path.stat().st_mtime
        if J not in latest_by_J or mtime > latest_by_J[J][0]:
            latest_by_J[J] = (mtime, metrics_path)

    return {J: path for J, (_, path) in latest_by_J.items()}


def find_metrics_by_subdomains_weak(
    results_dir: Path,
    algorithm: str,
    kappa: float,
    omega: Optional[float] = None,
) -> Dict[int, Path]:
    """Find metrics for weak scaling: varying J with proportional m (m/J ~ constant)."""
    latest_by_J: Dict[int, Tuple[float, Path]] = {}

    for metrics_path in results_dir.rglob("metrics.json"):
        try:
            data = load_metrics(metrics_path)
        except Exception:
            continue

        if data.get("algorithm") != algorithm:
            continue
        if float(data.get("wavenumber")) != float(kappa):
            continue
        if algorithm == "fixed-point" and omega is not None and data.get("omega") != omega:
            continue

        J = data.get("subdomains")
        m = data.get("mesh_size")
        if J is None or m is None:
            continue

        mtime = metrics_path.stat().st_mtime
        if J not in latest_by_J or mtime > latest_by_J[J][0]:
            latest_by_J[J] = (mtime, metrics_path)

    return {J: path for J, (_, path) in latest_by_J.items()}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot convergence (residual) curves for GMRES and fixed-point"
    )
    parser.add_argument("-m", "--mesh-size", type=int, required=True, help="Mesh parameter m")
    parser.add_argument("-J", "--subdomains", type=int, required=True, help="Number of subdomains")
    parser.add_argument("-k", "--wavenumber", type=float, required=True, help="Wavenumber κ")
    parser.add_argument(
        "--omega",
        type=float,
        default=None,
        help="Fixed-point relaxation parameter ω to select a specific run",
    )
    parser.add_argument(
        "--results-dir",
        default="../results",
        help="Results directory (default: ../results)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output PNG file (if omitted, show interactively)",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional plot title override",
    )
    parser.add_argument(
        "--plot-omega-sweep",
        action="store_true",
        help="Also plot fixed-point convergence for all ω found",
    )
    parser.add_argument(
        "--plot-mesh-sweep",
        action="store_true",
        help="Also plot convergence for multiple mesh sizes",
    )
    parser.add_argument(
        "--mesh-sweep-algorithm",
        choices=["gmres", "fixed-point"],
        default="fixed-point",
        help="Algorithm for mesh sweep plot (default: fixed-point)",
    )
    parser.add_argument(
        "--plot-strong-scaling",
        action="store_true",
        help="Plot convergence vs number of subdomains (fixed domain size)",
    )
    parser.add_argument(
        "--plot-weak-scaling",
        action="store_true",
        help="Plot convergence vs number of subdomains (fixed DOF per subdomain)",
    )
    parser.add_argument(
        "--scaling-algorithm",
        choices=["gmres", "fixed-point"],
        default="gmres",
        help="Algorithm for scaling plots (default: gmres)",
    )
    parser.add_argument(
        "--skip-comparison",
        action="store_true",
        help="Skip GMRES vs fixed-point comparison and only run selected sweeps",
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not args.skip_comparison:
        gmres_path = find_metrics(
            results_dir, "gmres", args.mesh_size, args.subdomains, args.wavenumber
        )
        fixed_path = find_metrics(
            results_dir,
            "fixed-point",
            args.mesh_size,
            args.subdomains,
            args.wavenumber,
            omega=args.omega,
        )

        if gmres_path is None:
            print("Error: no gmres metrics.json found for given parameters.")
            return 1
        if fixed_path is None:
            print("Error: no fixed-point metrics.json found for given parameters.")
            return 1

        gmres = load_metrics(gmres_path)
        fixed = load_metrics(fixed_path)

        print(f"Using GMRES metrics: {gmres_path}")
        print(f"Using fixed-point metrics: {fixed_path}")

        if gmres.get("algorithm") != "gmres":
            print(f"Warning: gmres metrics file reports algorithm={gmres.get('algorithm')}")
        if fixed.get("algorithm") != "fixed-point":
            print(f"Warning: fixed metrics file reports algorithm={fixed.get('algorithm')}")

        mismatches = check_same_experiment(gmres, fixed)
        if mismatches:
            print("Warning: experiment parameters differ:")
            for item in mismatches:
                print(f"  - {item}")

        gmres_res = gmres.get("residual_history", [])
        fixed_res = fixed.get("residual_history", [])

        if not gmres_res:
            print("Error: GMRES residual_history is empty.")
            return 1
        if not fixed_res:
            print("Error: fixed-point residual_history is empty.")
            return 1

        gmres_iter = np.arange(1, len(gmres_res) + 1)
        fixed_iter = np.arange(1, len(fixed_res) + 1)

        plt.figure(figsize=(9, 6))
        plt.semilogy(gmres_iter, gmres_res, label="GMRES", linewidth=2)
        plt.semilogy(fixed_iter, fixed_res, label="Fixed-Point", linewidth=2)

        title = args.title
        if title is None:
            omega_text = fixed.get("omega")
            title = (
                f"Convergence Comparison (m={gmres.get('mesh_size')}, "
                f"J={gmres.get('subdomains')}, κ={gmres.get('wavenumber')}, ω={omega_text})"
            )

        plt.title(title)
        plt.xlabel("Iteration")
        plt.ylabel("Residual norm")
        plt.grid(True, which="both", linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()

        if args.output:
            output_path = Path(args.output)
            plt.savefig(output_path, dpi=150)
            print(f"Saved plot to {output_path}")
        else:
            plt.show()

    if args.plot_omega_sweep:
        fixed_by_omega = find_fixed_metrics_by_omega(
            results_dir,
            args.mesh_size,
            args.subdomains,
            args.wavenumber,
        )
        if not fixed_by_omega:
            print("Warning: no fixed-point runs found for omega sweep.")
            return 0

        plt.figure(figsize=(9, 6))
        for omega, path in sorted(fixed_by_omega.items(), key=lambda x: x[0]):
            data = load_metrics(path)
            res = data.get("residual_history", [])
            if not res:
                continue
            iters = np.arange(1, len(res) + 1)
            plt.semilogy(iters, res, label=f"ω={omega}", linewidth=2)

        plt.title(
            f"Fixed-Point Convergence vs ω (m={args.mesh_size}, J={args.subdomains}, κ={args.wavenumber})"
        )
        plt.xlabel("Iteration")
        plt.ylabel("Residual norm")
        plt.grid(True, which="both", linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()

        if args.output:
            output_path = Path(args.output)
            sweep_path = output_path.with_name(
                output_path.stem + "_omega_sweep" + output_path.suffix
            )
            plt.savefig(sweep_path, dpi=150)
            print(f"Saved omega sweep plot to {sweep_path}")
        else:
            plt.show()

    if args.plot_mesh_sweep:
        sweep_algo = args.mesh_sweep_algorithm
        mesh_runs = find_metrics_by_mesh_size(
            results_dir,
            sweep_algo,
            args.subdomains,
            args.wavenumber,
            omega=args.omega,
        )
        if not mesh_runs:
            print("Warning: no runs found for mesh sweep.")
            return 0

        plt.figure(figsize=(9, 6))
        for m, path in sorted(mesh_runs.items(), key=lambda x: x[0]):
            data = load_metrics(path)
            res = data.get("residual_history", [])
            if not res:
                continue
            iters = np.arange(1, len(res) + 1)
            label = f"m={m}"
            if sweep_algo == "fixed-point":
                label = f"m={m}, ω={data.get('omega')}"
            plt.semilogy(iters, res, label=label, linewidth=2)

        sweep_title = (
            f"{sweep_algo.upper()} Convergence vs Mesh (J={args.subdomains}, κ={args.wavenumber})"
        )
        plt.title(sweep_title)
        plt.xlabel("Iteration")
        plt.ylabel("Residual norm")
        plt.grid(True, which="both", linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()

        if args.output:
            output_path = Path(args.output)
            sweep_path = output_path.with_name(
                output_path.stem + f"_{sweep_algo}_mesh_sweep" + output_path.suffix
            )
            plt.savefig(sweep_path, dpi=150)
            print(f"Saved mesh sweep plot to {sweep_path}")
        else:
            plt.show()

    # Strong scaling plot (fixed domain size, varying J)
    if args.plot_strong_scaling:
        scaling_algo = args.scaling_algorithm
        strong_runs = find_metrics_by_subdomains_strong(
            results_dir,
            scaling_algo,
            args.mesh_size,
            args.wavenumber,
            omega=args.omega if scaling_algo == "fixed-point" else None,
        )
        if not strong_runs:
            print("Warning: no runs found for strong scaling.")
            return 0

        plt.figure(figsize=(9, 6))
        for J, path in sorted(strong_runs.items(), key=lambda x: x[0]):
            data = load_metrics(path)
            res = data.get("residual_history", [])
            if not res:
                continue
            iters = np.arange(1, len(res) + 1)
            label = f"J={J}"
            if scaling_algo == "fixed-point":
                label = f"J={J}, ω={data.get('omega')}"
            plt.semilogy(iters, res, label=label, linewidth=2)

        strong_title = (
            f"{scaling_algo.upper()} Strong Scaling (m={args.mesh_size}, κ={args.wavenumber})"
        )
        plt.title(strong_title)
        plt.xlabel("Iteration")
        plt.ylabel("Residual norm")
        plt.grid(True, which="both", linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()

        if args.output:
            output_path = Path(args.output)
            sweep_path = output_path.with_name(
                output_path.stem + f"_{scaling_algo}_strong_scaling" + output_path.suffix
            )
            plt.savefig(sweep_path, dpi=150)
            print(f"Saved strong scaling plot to {sweep_path}")
        else:
            plt.show()

    # Weak scaling plot (fixed DOF per subdomain, varying J)
    if args.plot_weak_scaling:
        scaling_algo = args.scaling_algorithm
        weak_runs = find_metrics_by_subdomains_weak(
            results_dir,
            scaling_algo,
            args.wavenumber,
            omega=args.omega if scaling_algo == "fixed-point" else None,
        )
        if not weak_runs:
            print("Warning: no runs found for weak scaling.")
            return 0

        plt.figure(figsize=(9, 6))
        for J, path in sorted(weak_runs.items(), key=lambda x: x[0]):
            data = load_metrics(path)
            res = data.get("residual_history", [])
            m = data.get("mesh_size")
            if not res:
                continue
            iters = np.arange(1, len(res) + 1)
            label = f"J={J}, m={m}"
            if scaling_algo == "fixed-point":
                label = f"J={J}, m={m}, ω={data.get('omega')}"
            plt.semilogy(iters, res, label=label, linewidth=2)

        weak_title = (
            f"{scaling_algo.upper()} Weak Scaling (κ={args.wavenumber})"
        )
        plt.title(weak_title)
        plt.xlabel("Iteration")
        plt.ylabel("Residual norm")
        plt.grid(True, which="both", linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()

        if args.output:
            output_path = Path(args.output)
            sweep_path = output_path.with_name(
                output_path.stem + f"_{scaling_algo}_weak_scaling" + output_path.suffix
            )
            plt.savefig(sweep_path, dpi=150)
            print(f"Saved weak scaling plot to {sweep_path}")
        else:
            plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
