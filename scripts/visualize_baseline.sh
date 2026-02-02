#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="$ROOT_DIR/src"
RESULTS_DIR="$ROOT_DIR/results/visualization_baseline_gmres"

# Parameters for a clear visualization
MESH_SIZE=32
SUBDOMAINS=1
WAVENUMBER=16
SOURCES=8
TOLERANCE=1e-8
SEED=42
Lx=1.0
Ly=2.0

mkdir -p "$RESULTS_DIR"

(cd "$SRC_DIR" && python main.py \
  --algorithm baseline-gmres \
  --mesh-size "$MESH_SIZE" \
  --subdomains "$SUBDOMAINS" \
  --wavenumber "$WAVENUMBER" \
  --sources "$SOURCES" \
  --tolerance "$TOLERANCE" \
  --seed "$SEED" \
  --Lx "$Lx" \
  --Ly "$Ly" \
  --plot-global \
  --plot-local \
  --plot-mesh \
  --save-plots \
  --output-dir "$RESULTS_DIR")

echo "GMRES visualization complete."
echo "Results and plots saved under: $RESULTS_DIR"
