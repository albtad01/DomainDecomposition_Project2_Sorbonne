#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ANALYSIS_DIR="$ROOT_DIR/analyses"
RESULTS_DIR="$ROOT_DIR/results/runtime_gmres_comparison"
FIGURES_DIR="$ROOT_DIR/figures/runtime_gmres_comparison"

mkdir -p "$FIGURES_DIR"

(cd "$ANALYSIS_DIR" && python plot_runtime_gmres_subdomains.py \
  --results-dir "$RESULTS_DIR" \
  --figures-dir "$FIGURES_DIR")

echo "Runtime GMRES subdomain plot complete."
echo "Figures: $FIGURES_DIR"
