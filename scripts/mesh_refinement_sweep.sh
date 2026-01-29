#!/usr/bin/env bash
set -euo pipefail

# Mesh refinement sweep for GMRES and fixed-point, then plot mesh-sweep convergence.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="$ROOT_DIR/src"
ANALYSIS_DIR="$ROOT_DIR/analyses"
EXP_RESULTS_DIR="$ROOT_DIR/results/mesh_refinement"
FIGURES_DIR="$ROOT_DIR/figures/mesh_refinement"
mkdir -p "$FIGURES_DIR"

# Fixed parameters
SUBDOMAINS=4
WAVENUMBER=16
SOURCES=8
TOLERANCE=1e-8
MAX_ITER=500
OMEGA=0.1

# Mesh sizes (doubling): 32 -> 256
MESH_SIZES=(32 64 128)

echo "Running GMRES mesh refinement sweep..."
for M in "${MESH_SIZES[@]}"; do
  echo "  m=${M}"
  (cd "$SRC_DIR" && python main.py \
    -a gmres \
    -m "$M" \
    -J "$SUBDOMAINS" \
    -k "$WAVENUMBER" \
    -s "$SOURCES" \
    --tolerance "$TOLERANCE" \
    --max-iterations "$MAX_ITER" \
    --output-dir "${EXP_RESULTS_DIR}" \
    --save-plots)

done

echo "Running fixed-point mesh refinement sweep (omega=${OMEGA})..."
for M in "${MESH_SIZES[@]}"; do
  echo "  m=${M}"
  (cd "$SRC_DIR" && python main.py \
    -a fixed-point \
    -m "$M" \
    -J "$SUBDOMAINS" \
    -k "$WAVENUMBER" \
    -s "$SOURCES" \
    --tolerance "$TOLERANCE" \
    --max-iterations "$MAX_ITER" \
    --omega "$OMEGA" \
    --output-dir "${EXP_RESULTS_DIR}" \
    --save-plots)

done

echo "Generating mesh-sweep convergence plots..."
(cd "$ANALYSIS_DIR" && python plot_convergence.py \
  -m 32 \
  -J "$SUBDOMAINS" \
  -k "$WAVENUMBER" \
  --omega "$OMEGA" \
  --results-dir "${EXP_RESULTS_DIR}" \
  --plot-mesh-sweep \
  --mesh-sweep-algorithm gmres \
  --output "$FIGURES_DIR/gmres_mesh_refinement.png")

(cd "$ANALYSIS_DIR" && python plot_convergence.py \
  -m 32 \
  -J "$SUBDOMAINS" \
  -k "$WAVENUMBER" \
  --omega "$OMEGA" \
  --results-dir "${EXP_RESULTS_DIR}" \
  --plot-mesh-sweep \
  --mesh-sweep-algorithm fixed-point \
  --output "$FIGURES_DIR/fixed_point_mesh_refinement.png")

echo "Done."
