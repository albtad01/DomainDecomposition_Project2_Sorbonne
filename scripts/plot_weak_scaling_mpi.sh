#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ANALYSIS_DIR="$ROOT_DIR/analyses"
RESULTS_DIR="$ROOT_DIR/results/weak_scaling_mpi"
FIGURES_DIR="$ROOT_DIR/figures/weak_scaling_mpi"

mkdir -p "$FIGURES_DIR"

(cd "$ANALYSIS_DIR" && python plot_weak_scling_mpi.py \
  --results-dir "$RESULTS_DIR" \
  --figures-dir "$FIGURES_DIR")

echo "Weak scaling MPI plot complete."
echo "Figures: $FIGURES_DIR"
