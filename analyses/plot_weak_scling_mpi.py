#!/usr/bin/env python3
"""
Plot weak scaling runtime and efficiency for MPI runs.

Reads metrics.json files from results/weak_scaling_mpi and uses J=2 as baseline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, Any, List

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

LABEL_FONTSIZE = 24
TICK_FONTSIZE = 20
LEGEND_FONTSIZE = 24


def load_metrics(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def collect_latest_metrics(results_dir: Path) -> Dict[Tuple[str, int], Path]:
    """
    Returns latest metrics.json per (base_label, J).
    base_label is the parent folder like base_32.
    """
    latest: Dict[Tuple[str, int], Tuple[float, Path]] = {}

    for metrics_path in results_dir.rglob("metrics.json"):
        try:
            data = load_metrics(metrics_path)
        except Exception:
            continue

        algo = data.get("algorithm")
        if algo not in {"mpi-fixed-point"}:
            continue

        J = data.get("subdomains")
        if J is None:
            continue

        if int(J) > 16:
            continue

        try:
            # Parent[2] is the experiment group folder (e.g. weak_scaling_mpi_32_XXXX)
            group_folder = metrics_path.parents[2].name
            # Try to extract the base size (assuming format weak_scaling_mpi_32_...)
            parts = group_folder.split('_')
            if len(parts) >= 4 and parts[3].isdigit():
                base_val = parts[3]
                base_label = f"Base m={base_val}"
            else:
                base_label = group_folder
        except Exception:
            base_label = "base"

        key = (base_label, int(J))
        mtime = metrics_path.stat().st_mtime
        if key not in latest or mtime > latest[key][0]:
            latest[key] = (mtime, metrics_path)

    return {k: v for k, (_, v) in latest.items()}


def build_series(
    latest: Dict[Tuple[str, int], Path], baseline_j: int
) -> Dict[str, Dict[str, List[Tuple[int, float]]]]:
    """
    Build series per base label: runtime, efficiency.
    Returns {base_label: {"time": [(J, t)], "eff": [(J, e)]}}
    """
    per_base: Dict[str, Dict[int, float]] = {}
    mesh_sizes: Dict[str, int] = {}

    for (base_label, J), path in latest.items():
        data = load_metrics(path)
        total_time = data.get("build_time")
        if total_time is None:
            continue

        per_base.setdefault(base_label, {})[int(J)] = float(total_time)
        if base_label not in mesh_sizes:
            m = data.get("mesh_size")
            if m is not None:
                mesh_sizes[base_label] = int(m)

    series: Dict[str, Dict[str, List[Tuple[int, float]]]] = {}
    for base_label, times_by_j in per_base.items():
        if baseline_j not in times_by_j:
            print(f"Warning: missing baseline J={baseline_j} for {base_label}. Skipping.")
            continue

        t_base = times_by_j[baseline_j]
        js_sorted = sorted(times_by_j.keys())
        efficiency = [(j, t_base / times_by_j[j]) for j in js_sorted]
        time_series = [(j, times_by_j[j]) for j in js_sorted]

        label = base_label
        # For weak scaling, mesh size changes, so we rely on the base_label we constructed
        # if base_label in mesh_sizes:
        #     label = f"m={mesh_sizes[base_label]}"

        series[label] = {
            "time": time_series,
            "eff": efficiency,
        }

    return series


def plot_weak_scaling(
    series: Dict[str, Dict[str, List[Tuple[int, float]]]],
    out_path: Path,
    baseline_j: int,
    wavenumber: float | None,
    omega: float | None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    runtime_path = out_path.with_name(out_path.stem + "_runtime" + out_path.suffix)
    efficiency_path = out_path.with_name(out_path.stem + "_efficiency" + out_path.suffix)

    fig_time = plt.figure(figsize=(9.0, 6.8), dpi=140)
    ax_time = fig_time.add_subplot(111)

    fig_eff = plt.figure(figsize=(9.0, 6.8), dpi=140)
    ax_eff = fig_eff.add_subplot(111)

    all_js: List[int] = []
    baseline_times: List[float] = []
    for label, data in series.items():
        js = [j for j, _ in data["time"]]
        times = [v for _, v in data["time"]]
        eff = [v for _, v in data["eff"]]

        all_js.extend(js)
        for j, t in data["time"]:
            if j == baseline_j:
                baseline_times.append(t)

        ax_time.plot(js, times, marker="o", linewidth=3.0, label=label)
        ax_eff.plot(js, eff, marker="o", linewidth=3.0, label=label)

    if all_js:
        js_sorted = sorted(set(all_js))
        if baseline_times:
            ideal_time = [baseline_times[0] for _ in js_sorted]
            ax_time.plot(js_sorted, ideal_time, linestyle=":", color="black", linewidth=3.0, label="Ideal")
        ideal_eff = [1.0 for _ in js_sorted]
        ax_eff.plot(js_sorted, ideal_eff, linestyle=":", color="black", linewidth=3.0, label="Ideal")

    ax_time.set_xlabel("Subdomains J", fontsize=LABEL_FONTSIZE)
    ax_time.set_ylabel("Runtime (s)", fontsize=LABEL_FONTSIZE)
    
    ax_time.set_xscale("log", base=2)
    ax_time.set_yscale("log")
    ax_time.grid(True, linestyle="--", alpha=0.4)
    ax_time.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    time_legend = ax_time.legend(fontsize=LEGEND_FONTSIZE, loc="upper right")
    time_param_parts = [f"baseline J={baseline_j}"]
    if wavenumber is not None:
        time_param_parts.append(f"k={wavenumber}")
    if omega is not None:
        time_param_parts.append(f"ω={omega}")
    time_param = Line2D([], [], color="none", label="\n".join(time_param_parts))
    ax_time.add_artist(time_legend)
    ax_time.legend(
        handles=[time_param],
        fontsize=LEGEND_FONTSIZE,
        loc="lower right",
        handlelength=0,
        handletextpad=0,
        borderpad=0.3,
        labelspacing=0.3,
    )

    ax_eff.set_xlabel("Subdomains J", fontsize=LABEL_FONTSIZE)
    ax_eff.set_ylabel("Weak efficiency", fontsize=LABEL_FONTSIZE)
    
    ax_eff.set_xscale("log", base=2)
    ax_eff.set_yscale("log")
    ax_eff.grid(True, linestyle="--", alpha=0.4)
    ax_eff.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    eff_legend = ax_eff.legend(fontsize=LEGEND_FONTSIZE, loc="lower left")
    eff_param_parts = [f"baseline J={baseline_j}"]
    if wavenumber is not None:
        eff_param_parts.append(f"k={wavenumber}")
    if omega is not None:
        eff_param_parts.append(f"ω={omega}")
    eff_param = Line2D([], [], color="none", label="\n".join(eff_param_parts))
    ax_eff.add_artist(eff_legend)
    ax_eff.legend(
        handles=[eff_param],
        fontsize=LEGEND_FONTSIZE,
        loc="lower right",
        handlelength=0,
        handletextpad=0,
        borderpad=0.3,
        labelspacing=0.3,
    )

    fig_time.tight_layout()
    fig_eff.tight_layout()

    fig_time.savefig(runtime_path, dpi=150)
    fig_eff.savefig(efficiency_path, dpi=150)
    print(f"Saved plot to {runtime_path}")
    print(f"Saved plot to {efficiency_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot weak scaling for MPI runs")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="../results/weak_scaling_mpi",
        help="Results directory (default: ../results/weak_scaling_mpi)",
    )
    parser.add_argument(
        "--figures-dir",
        type=str,
        default="../figures/weak_scaling_mpi",
        help="Figures directory (default: ../figures/weak_scaling_mpi)",
    )
    parser.add_argument(
        "--baseline-j",
        type=int,
        default=1,
        help="Baseline subdomains J (default: 1)",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="weak_scaling_mpi.png",
        help="Output figure filename (default: weak_scaling_mpi.png)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    latest = collect_latest_metrics(results_dir)
    if not latest:
        print("No metrics.json files found for mpi-fixed-point in weak scaling results.")
        return 1

    series = build_series(latest, baseline_j=args.baseline_j)
    if not series:
        print("No valid series to plot (missing baselines or data).")
        return 1

    out_path = figures_dir / args.output_name
    wavenumber = None
    omega = None
    if latest:
        sample_path = next(iter(latest.values()))
        try:
            sample_data = load_metrics(sample_path)
            wavenumber = sample_data.get("wavenumber")
            omega = sample_data.get("omega")
        except Exception:
            pass

    plot_weak_scaling(
        series,
        out_path,
        baseline_j=args.baseline_j,
        wavenumber=wavenumber,
        omega=omega,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
