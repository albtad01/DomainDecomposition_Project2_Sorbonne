#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ANALYSIS_DIR="$ROOT_DIR/analyses"
RESULTS_DIR="$ROOT_DIR/results/strong_scaling_mpi"
FIGURES_DIR="$ROOT_DIR/figures/strong_scaling_mpi"

mkdir -p "$FIGURES_DIR"

(cd "$ANALYSIS_DIR" && python plot_strong_scaling_mpi.py \
  --results-dir "$RESULTS_DIR" \
  --figures-dir "$FIGURES_DIR")

echo "Strong scaling MPI plot complete."
echo "Figures: $FIGURES_DIR"
