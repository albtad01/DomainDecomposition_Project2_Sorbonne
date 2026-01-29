#!/usr/bin/env python3
"""
Plot runtime comparison between DD-GMRES and baseline-GMRES.

Reads metrics.json files from results directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import json
import matplotlib.pyplot as plt


def load_metrics(path: Path) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot runtime comparison from results")
    parser.add_argument("--results-dir", type=str, default="../results/runtime_gmres_comparison")
    parser.add_argument("--figures-dir", type=str, default="../figures/runtime_gmres_comparison")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    latest: Dict[Tuple[str, int], Tuple[float, Path]] = {}

    for metrics_path in results_dir.rglob("metrics.json"):
        try:
            data = load_metrics(metrics_path)
        except Exception:
            continue

        algo = data.get("algorithm")
        if algo not in {"gmres", "baseline-gmres"}:
            continue

        m = data.get("mesh_size")
        if m is None:
            continue

        key = (algo, int(m))
        mtime = metrics_path.stat().st_mtime
        if key not in latest or mtime > latest[key][0]:
            latest[key] = (mtime, metrics_path)

    # Collect by mesh size
    mesh_sizes = sorted({m for (_, m) in latest.keys()})
    if not mesh_sizes:
        print("No metrics.json files found for gmres or baseline-gmres.")
        return 1

    gmres_times = []
    baseline_times = []

    for m in mesh_sizes:
        gm_key = ("gmres", m)
        bl_key = ("baseline-gmres", m)

        if gm_key in latest:
            gm_data = load_metrics(latest[gm_key][1])
            gmres_times.append(gm_data.get("total_time", None))
        else:
            gmres_times.append(None)

        if bl_key in latest:
            bl_data = load_metrics(latest[bl_key][1])
            baseline_times.append(bl_data.get("total_time", None))
        else:
            baseline_times.append(None)

    plt.figure(figsize=(9, 6))
    plt.plot(mesh_sizes, baseline_times, marker="o", linewidth=2, label="Full GMRES")
    plt.plot(mesh_sizes, gmres_times, marker="o", linewidth=2, label="DD-GMRES")
    plt.xlabel("Mesh size m")
    plt.ylabel("Runtime (s)")
    plt.title("Runtime Comparison: DD-GMRES vs Full GMRES")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    out_path = figures_dir / "runtime_comparison.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
