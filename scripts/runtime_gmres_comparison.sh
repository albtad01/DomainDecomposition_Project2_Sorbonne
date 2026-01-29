#!/usr/bin/env bash
set -euo pipefail

# Runtime comparison: run main.py for DD-GMRES and baseline-GMRES, then plot.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="$ROOT_DIR/src"
ANALYSIS_DIR="$ROOT_DIR/analyses"
RESULTS_DIR="$ROOT_DIR/results/runtime_gmres_comparison"
FIGURES_DIR="$ROOT_DIR/figures/runtime_gmres_comparison"

# Parameters
MESH_SIZES=(32 64 128)
SUBDOMAINS=4
WAVENUMBER=16
SOURCES=8
TOLERANCE=1e-8
SEED=42
Lx=1.0
Ly=2.0

mkdir -p "$RESULTS_DIR" "$FIGURES_DIR"

for M in "${MESH_SIZES[@]}"; do
  echo "Running DD-GMRES (m=${M})"
  (cd "$SRC_DIR" && python main.py \
    --algorithm gmres \
    --mesh-size "$M" \
    --subdomains "$SUBDOMAINS" \
    --wavenumber "$WAVENUMBER" \
    --sources "$SOURCES" \
    --tolerance "$TOLERANCE" \
    --seed "$SEED" \
    --Lx "$Lx" \
    --Ly "$Ly" \
    --output-dir "$RESULTS_DIR")

  echo "Running baseline GMRES (m=${M})"
  (cd "$SRC_DIR" && python main.py \
    --algorithm baseline-gmres \
    --mesh-size "$M" \
    --subdomains 1 \
    --wavenumber "$WAVENUMBER" \
    --sources "$SOURCES" \
    --tolerance "$TOLERANCE" \
    --seed "$SEED" \
    --Lx "$Lx" \
    --Ly "$Ly" \
    --output-dir "$RESULTS_DIR")
done

(cd "$ANALYSIS_DIR" && python plot_runtime.py \
  --results-dir "$RESULTS_DIR" \
  --figures-dir "$FIGURES_DIR")

echo "Runtime comparison complete."
echo "Results: $RESULTS_DIR"
echo "Figures: $FIGURES_DIR"
