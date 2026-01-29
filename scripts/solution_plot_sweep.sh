#!/usr/bin/env bash
set -euo pipefail

# Sweep mesh sizes and domain sizes for GMRES and Fixed-Point.
# Saves solution plots into results/solution_plot_sweep/<exp>/<timestamp>/plots

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="$ROOT_DIR/src"
RESULTS_DIR="$ROOT_DIR/results/solution_plot_sweep"

mkdir -p "$RESULTS_DIR"

# Parameters
MESH_SIZES=(32)
SUBDOMAINS=4
WAVENUMBER=16
SOURCES=8
TOLERANCE=1e-8
MAX_ITER=200
OMEGA=1.0
SEED=42

# Domain sizes (Lx,Ly pairs)
DOMAIN_SIZES=("4.0 6.0")

for M in "${MESH_SIZES[@]}"; do
  for DOMAIN in "${DOMAIN_SIZES[@]}"; do
    Lx=$(echo "$DOMAIN" | awk '{print $1}')
    Ly=$(echo "$DOMAIN" | awk '{print $2}')
    DOMAIN_TAG="Lx${Lx}_Ly${Ly}"
    DOMAIN_RESULTS_DIR="$RESULTS_DIR/$DOMAIN_TAG"
    mkdir -p "$DOMAIN_RESULTS_DIR"

    echo "Running GMRES: m=${M}, Lx=${Lx}, Ly=${Ly}"
    (cd "$SRC_DIR" && python main.py \
      --algorithm gmres \
      --mesh-size "$M" \
      --subdomains "$SUBDOMAINS" \
      --wavenumber "$WAVENUMBER" \
      --sources "$SOURCES" \
      --tolerance "$TOLERANCE" \
      --max-iterations "$MAX_ITER" \
      --seed "$SEED" \
      --Lx "$Lx" \
      --Ly "$Ly" \
      --output-dir "$DOMAIN_RESULTS_DIR" \
      --plot-global \
      --save-plots)

    echo "Running Fixed-Point: m=${M}, Lx=${Lx}, Ly=${Ly}"
    (cd "$SRC_DIR" && python main.py \
      --algorithm fixed-point \
      --mesh-size "$M" \
      --subdomains "$SUBDOMAINS" \
      --wavenumber "$WAVENUMBER" \
      --sources "$SOURCES" \
      --tolerance "$TOLERANCE" \
      --max-iterations "$MAX_ITER" \
      --omega "$OMEGA" \
      --seed "$SEED" \
      --Lx "$Lx" \
      --Ly "$Ly" \
      --output-dir "$DOMAIN_RESULTS_DIR" \
      --plot-global \
      --save-plots)
  done
done

echo "Done. Plots saved under: $RESULTS_DIR"
