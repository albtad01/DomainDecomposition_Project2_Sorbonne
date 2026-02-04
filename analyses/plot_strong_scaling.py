#!/usr/bin/env python3
"""
Plot strong scaling (Runtime & Iterations) for sequential DD runs.

Reads metrics.json files from results/strong_scaling.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, Any, List

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

LABEL_FONTSIZE = 16
TICK_FONTSIZE = 14
LEGEND_FONTSIZE = 14


def load_metrics(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def collect_latest_metrics(
    results_dir: Path, algorithm: str, mesh_size: int, wavenumber: float, omega: float | None
) -> Dict[int, Path]:
    """
    Returns latest metrics.json per J for the given configuration.
    """
    latest: Dict[int, Tuple[float, Path]] = {}

    for metrics_path in results_dir.rglob("metrics.json"):
        try:
            data = load_metrics(metrics_path)
        except Exception:
            continue

        if data.get("algorithm") != algorithm:
            continue
        if int(data.get("mesh_size")) != mesh_size:
            continue
        if abs(float(data.get("wavenumber")) - wavenumber) > 1e-9:
            continue
        if algorithm == "fixed-point" and omega is not None:
             if data.get("omega") is None or abs(float(data.get("omega")) - omega) > 1e-9:
                continue

        J = data.get("subdomains")
        if J is None:
            continue
        J = int(J)

        mtime = metrics_path.stat().st_mtime
        if J not in latest or mtime > latest[J][0]:
            latest[J] = (mtime, metrics_path)

    return {k: v for k, (_, v) in latest.items()}


def plot_scaling(
    metrics_map: Dict[int, Path],
    out_path: Path,
    algorithm: str,
    mesh_size: int,
    wavenumber: float,
    omega: float | None,
) -> None:
    if not metrics_map:
        print(f"No data found for {algorithm}")
        return

    js = sorted(metrics_map.keys())
    times = []
    iters = []
    
    for j in js:
        data = load_metrics(metrics_map[j])
        times.append(data.get("total_time", 0.0))
        iters.append(data.get("iterations", 0))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Runtime
    ax1.plot(js, times, 'o-', linewidth=2, label='Total Time')
    ax1.set_xlabel('Subdomains (J)', fontsize=LABEL_FONTSIZE)
    ax1.set_ylabel('Time (s)', fontsize=LABEL_FONTSIZE)
    ax1.set_title(f'Strong Scaling - Runtime ({algorithm})', fontsize=LABEL_FONTSIZE)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.tick_params(labelsize=TICK_FONTSIZE)
    
    # Plot 2: Iterations
    ax2.plot(js, iters, 's-', color='orange', linewidth=2, label='Iterations')
    ax2.set_xlabel('Subdomains (J)', fontsize=LABEL_FONTSIZE)
    ax2.set_ylabel('Iterations', fontsize=LABEL_FONTSIZE)
    ax2.set_title(f'Strong Scaling - Convergence ({algorithm})', fontsize=LABEL_FONTSIZE)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.tick_params(labelsize=TICK_FONTSIZE)

    # Metadata text
    info = f"m={mesh_size}, k={wavenumber}"
    if omega:
        info += f", ω={omega}"
    fig.suptitle(info, fontsize=14)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved scaling plot to {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot strong scaling metrics")
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--algorithm", type=str, required=True)
    parser.add_argument("--mesh-size", type=int, required=True)
    parser.add_argument("--wavenumber", type=float, required=True)
    parser.add_argument("--omega", type=float, default=None)

    args = parser.parse_args()
    
    metrics = collect_latest_metrics(
        Path(args.results_dir), 
        args.algorithm, 
        args.mesh_size, 
        args.wavenumber, 
        args.omega
    )
    
    plot_scaling(
        metrics, 
        Path(args.output), 
        args.algorithm, 
        args.mesh_size, 
        args.wavenumber, 
        args.omega
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
