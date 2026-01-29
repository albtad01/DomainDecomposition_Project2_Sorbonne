#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _ensure_outdir(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)


def load_and_filter(csv_path: Path, require_nx_eq_ny: bool = False) -> pd.DataFrame:
    """
    Load CSV and return a clean dataframe with at least columns:
      m, iters, nx, ny
    Optionally keep only rows where nx == ny (if you have such runs).
    """
    df = pd.read_csv(csv_path)

    required = {"m", "iters", "nx", "ny"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")

    # keep numeric, drop NaNs
    for c in ["m", "iters", "nx", "ny"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["m", "iters", "nx", "ny"]).copy()

    # filter out failed runs (gmres_info sometimes nonzero). If column exists, keep only successful by default.
    if "gmres_info" in df.columns:
        df["gmres_info"] = pd.to_numeric(df["gmres_info"], errors="coerce")
        df = df[(df["gmres_info"].fillna(0) == 0)].copy()

    if require_nx_eq_ny:
        df = df[df["nx"] == df["ny"]].copy()

    # aggregate duplicates (same m may appear multiple seeds): take median iters
    df_agg = (
        df.groupby("m", as_index=False)
          .agg(iters=("iters", "median"),
               nx=("nx", "median"),
               ny=("ny", "median"),
               n=("n", "median") if "n" in df.columns else ("iters", "size"))
    )

    df_agg = df_agg.sort_values("m").reset_index(drop=True)
    return df_agg


def plot_iters_vs_m(df: pd.DataFrame, outpath: Path | None, title: str | None = None, annotate: bool = False) -> None:
    m = df["m"].to_numpy(dtype=float)
    iters = df["iters"].to_numpy(dtype=float)

    fig = plt.figure(figsize=(9.2, 5.2), dpi=140)
    ax = fig.add_subplot(111)

    ax.plot(m, iters, marker="o", linewidth=2.0, markersize=5.5)

    ax.set_xlabel("m (discretization parameter)")
    ax.set_ylabel("GMRES iterations to reach tolerance")
    if title:
        ax.set_title(title)

    # Make it clean & readable
    ax.grid(True, which="major", linewidth=0.6, alpha=0.35)
    ax.grid(True, which="minor", linewidth=0.4, alpha=0.20)
    ax.minorticks_on()

    # Nice margins
    ax.margins(x=0.03, y=0.08)

    # Optional: annotate each point with (nx,ny)
    if annotate:
        for mi, iti, nxi, nyi in zip(m, iters, df["nx"], df["ny"]):
            ax.annotate(f"({int(nxi)},{int(nyi)})",
                        xy=(mi, iti),
                        xytext=(6, 6),
                        textcoords="offset points",
                        fontsize=8,
                        alpha=0.85)

    # Tight layout
    fig.tight_layout()

    if outpath is not None:
        outpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath, bbox_inches="tight")
        print(f"[OK] Saved figure to: {outpath}")

    plt.show()


def main() -> None:
    p = argparse.ArgumentParser(description="Plot GMRES iterations vs discretization parameter m from CSV.")
    p.add_argument("--csv", type=str, required=True, help="Path to CSV results file")
    p.add_argument("--out", type=str, default="figures/iters_vs_m.png", help="Output figure path (png/pdf/svg ok)")
    p.add_argument("--title", type=str, default=None, help="Plot title")
    p.add_argument("--annotate", action="store_true", help="Annotate points with (nx,ny)")
    p.add_argument("--nx_eq_ny", action="store_true", help="Keep only runs with nx == ny")
    args = p.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = load_and_filter(csv_path, require_nx_eq_ny=args.nx_eq_ny)

    if df.empty:
        raise RuntimeError("No valid rows after filtering. Check CSV or flags.")

    outpath = Path(args.out) if args.out else None

    title = args.title
    if title is None:
        title = "Mesh refinement: iterations vs m"

    plot_iters_vs_m(df, outpath=outpath, title=title, annotate=args.annotate)


if __name__ == "__main__":
    main()
