#!/bin/bash
# Strong Scaling Study: Fixed domain size (m=128), varying number of subdomains J
# Studies how convergence is affected by domain decomposition with constant problem size

set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="$ROOT_DIR/src"
ANALYSIS_DIR="$ROOT_DIR/analyses"
EXP_RESULTS_DIR="$ROOT_DIR/results/strong_scaling"
FIGURES_DIR="$ROOT_DIR/figures/strong_scaling"
mkdir -p "$FIGURES_DIR"
# Fixed parameters
MESH_SIZE=64
WAVENUMBER=5.0
TOLERANCE=1e-4
OMEGA=1.0  # For fixed-point

# Subdomain counts (powers of 2)
SUBDOMAINS=(2 4 8 16 32)

echo "=== Strong Scaling Study ==="
echo "Fixed mesh size: m=${MESH_SIZE}"
echo "Varying subdomains J: ${SUBDOMAINS[@]}"
echo "Wavenumber κ: ${WAVENUMBER}"
echo ""

# Run GMRES for all subdomain counts
echo "Running GMRES experiments..."
for J in "${SUBDOMAINS[@]}"; do
    echo "  J=${J}..."
     (cd "$SRC_DIR" && python main.py \
        --mesh-size ${MESH_SIZE} \
        --subdomains ${J} \
        --wavenumber ${WAVENUMBER} \
        --algorithm gmres \
        --tolerance ${TOLERANCE} \
        --output-dir "${EXP_RESULTS_DIR}" \
        --save-plots)
done

echo ""
echo "Running Fixed-Point experiments..."
for J in "${SUBDOMAINS[@]}"; do
    echo "  J=${J}..."
     (cd "$SRC_DIR" && python main.py \
        --mesh-size ${MESH_SIZE} \
        --subdomains ${J} \
        --wavenumber ${WAVENUMBER} \
        --algorithm fixed-point \
        --omega ${OMEGA} \
        --tolerance ${TOLERANCE} \
        --output-dir "${EXP_RESULTS_DIR}" \
        --save-plots)
done

echo ""
echo "=== Generating Plots ==="

# Generate strong scaling plot for GMRES
echo "Generating GMRES strong scaling plot..."
(cd "$ANALYSIS_DIR" && python plot_convergence.py \
    --mesh-size ${MESH_SIZE} \
    --subdomains ${SUBDOMAINS[0]} \
    --wavenumber ${WAVENUMBER} \
    --results-dir "${EXP_RESULTS_DIR}" \
    --plot-strong-scaling \
    --scaling-algorithm gmres \
    --output "$FIGURES_DIR/gmres_strong_scaling.png")

# Generate strong scaling plot for Fixed-Point
echo "Generating Fixed-Point strong scaling plot..."
(cd "$ANALYSIS_DIR" && python plot_convergence.py \
    --mesh-size ${MESH_SIZE} \
    --subdomains ${SUBDOMAINS[0]} \
    --wavenumber ${WAVENUMBER} \
    --omega ${OMEGA} \
    --results-dir "${EXP_RESULTS_DIR}" \
    --plot-strong-scaling \
    --scaling-algorithm fixed-point \
    --output "$FIGURES_DIR/fixed_point_strong_scaling.png")

echo ""
echo "=== Strong Scaling Study Complete ==="
echo "Plots saved to:"
echo "  - figures/strong_scaling/gmres_strong_scaling.png"
echo "  - figures/strong_scaling/fixed_point_strong_scaling.png"
