#!/usr/bin/env python3
"""
Plot runtime comparison for full GMRES and DD-GMRES by subdomains.

Reads metrics.json files from results/runtime_gmres_comparison.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, Any, List

import matplotlib.pyplot as plt

LABEL_FONTSIZE = 24
TICK_FONTSIZE = 20
LEGEND_FONTSIZE = 24


def load_metrics(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def collect_latest(results_dir: Path) -> Dict[Tuple[str, int, int], Path]:
    """
    Latest metrics per (algorithm, mesh_size, subdomains).
    """
    latest: Dict[Tuple[str, int, int], Tuple[float, Path]] = {}
    for metrics_path in results_dir.rglob("metrics.json"):
        try:
            data = load_metrics(metrics_path)
        except Exception:
            continue

        algo = data.get("algorithm")
        if algo not in {"gmres", "baseline-gmres"}:
            continue

        m = data.get("mesh_size")
        J = data.get("subdomains")
        if m is None or J is None:
            continue

        key = (algo, int(m), int(J))
        mtime = metrics_path.stat().st_mtime
        if key not in latest or mtime > latest[key][0]:
            latest[key] = (mtime, metrics_path)

    return {k: v for k, (_, v) in latest.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot runtime vs mesh size for GMRES variants")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="../results/runtime_gmres_comparison",
        help="Results directory (default: ../results/runtime_gmres_comparison)",
    )
    parser.add_argument(
        "--figures-dir",
        type=str,
        default="../figures/runtime_gmres_comparison",
        help="Figures directory (default: ../figures/runtime_gmres_comparison)",
    )
    parser.add_argument(
        "--mesh-sizes",
        type=int,
        nargs="+",
        default=None,
        help="Mesh sizes to include (default: all found)",
    )
    parser.add_argument(
        "--subdomains",
        type=int,
        nargs="+",
        default=None,
        help="Subdomains for DD-GMRES lines (default: all found)",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="runtime_gmres_by_subdomains.png",
        help="Output filename (default: runtime_gmres_by_subdomains.png)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    latest = collect_latest(results_dir)
    if not latest:
        print("No metrics.json files found for gmres or baseline-gmres.")
        return 1

    if args.mesh_sizes is None:
        mesh_sizes = sorted({m for (algo, m, _) in latest.keys()})
    else:
        mesh_sizes = sorted(set(int(m) for m in args.mesh_sizes))

    if args.subdomains is None:
        subdomains = sorted({j for (algo, _, j) in latest.keys() if algo == "gmres"})
    else:
        subdomains = sorted(set(int(j) for j in args.subdomains))

    # Full GMRES (baseline-gmres) line
    baseline_times: List[float | None] = []
    for m in mesh_sizes:
        key = ("baseline-gmres", m, 1)
        if key in latest:
            data = load_metrics(latest[key])
            baseline_times.append(data.get("total_time"))
        else:
            baseline_times.append(None)

    fig, ax = plt.subplots(figsize=(10.5, 7.5))
    ax.plot(mesh_sizes, baseline_times, marker="o", linewidth=3.0, label="Full GMRES (J=1)")

    # DD-GMRES lines for each subdomain count
    for J in subdomains:
        times: List[float | None] = []
        for m in mesh_sizes:
            key = ("gmres", m, J)
            if key in latest:
                data = load_metrics(latest[key])
                times.append(data.get("total_time"))
            else:
                times.append(None)

        ax.plot(mesh_sizes, times, marker="o", linewidth=3.0, label=f"DD-GMRES (J={J})")

    ax.set_xlabel("Mesh size m", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Runtime (s)", fontsize=LABEL_FONTSIZE)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax.legend(fontsize=LEGEND_FONTSIZE)
    fig.tight_layout()

    out_path = figures_dir / args.output_name
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
