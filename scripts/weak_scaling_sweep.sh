#!/bin/bash
# Weak Scaling Study: Fixed DOF per subdomain, varying number of subdomains J
# As J increases, total mesh size increases proportionally to maintain constant work per subdomain

set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXP_RESULTS_DIR="$ROOT_DIR/results/weak_scaling"
FIGURES_DIR="$ROOT_DIR/figures/weak_scaling"
ANALYSIS_DIR="$ROOT_DIR/analyses"
mkdir -p "$FIGURES_DIR"

# Base parameters
BASE_MESH_SIZE=16  # Mesh size for J=1
WAVENUMBER=5.0
TOLERANCE=1e-4
OMEGA=1.0  # For fixed-point

# Subdomain counts (powers of 2) with corresponding mesh sizes
# To maintain constant DOF per subdomain: m scales with J
# J=1: m=32 (baseline)
# J=2: m=64 (2x mesh for 2x subdomains)
# J=4: m=128 (4x mesh for 4x subdomains)
# J=8: m=256 (8x mesh for 8x subdomains)
# J=16: m=512

# Function to get mesh size for given J (portable bash 3.x compatible)
get_mesh_size() {
    local J=$1
    echo $((BASE_MESH_SIZE * J))
}

echo "=== Weak Scaling Study ==="
echo "Fixed DOF per subdomain (m/J ~ constant)"
echo "Wavenumber κ: ${WAVENUMBER}"
echo ""
echo "Configurations:"
for J in 2 4 8; do
    m=$(get_mesh_size $J)
    echo "  J=${J}, m=${m}"
done
echo ""

# Run GMRES for all configurations
echo "Running GMRES experiments..."
SRC_DIR="$ROOT_DIR/src"
for J in 2 4 8; do
    m=$(get_mesh_size $J)
    echo "  J=${J}, m=${m}..."
    (cd "$SRC_DIR" && python main.py \
        --mesh-size ${m} \
        --subdomains ${J} \
        --wavenumber ${WAVENUMBER} \
        --algorithm gmres \
        --tolerance ${TOLERANCE} \
        --output-dir "${EXP_RESULTS_DIR}" \
        --save-plots)
done

echo ""
echo "Running Fixed-Point experiments..."
for J in 2 4 8; do
    m=$(get_mesh_size $J)
    echo "  J=${J}, m=${m}..."
    (cd "$SRC_DIR" && python main.py \
        --mesh-size ${m} \
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

# Generate weak scaling plot for GMRES
echo "Generating GMRES weak scaling plot..."
SRC_DIR="$ROOT_DIR/src"
(cd "$ANALYSIS_DIR" && python plot_convergence.py \
    --mesh-size ${BASE_MESH_SIZE} \
    --subdomains 1 \
    --wavenumber ${WAVENUMBER} \
    --results-dir "${EXP_RESULTS_DIR}" \
    --skip-comparison \
    --plot-weak-scaling \
    --scaling-algorithm gmres \
    --output "$FIGURES_DIR/gmres_weak_scaling.png")

# Generate weak scaling plot for Fixed-Point
echo "Generating Fixed-Point weak scaling plot..."
(cd "$ANALYSIS_DIR" && python plot_convergence.py \
    --mesh-size ${BASE_MESH_SIZE} \
    --subdomains 1 \
    --wavenumber ${WAVENUMBER} \
    --omega ${OMEGA} \
    --results-dir "${EXP_RESULTS_DIR}" \
    --skip-comparison \
    --plot-weak-scaling \
    --scaling-algorithm fixed-point \
    --output "$FIGURES_DIR/fixed_point_weak_scaling.png")

echo ""
echo "=== Weak Scaling Study Complete ==="
echo "Plots saved to:"
echo "  - figures/weak_scaling/gmres_weak_scaling.png"
echo "  - figures/weak_scaling/fixed_point_weak_scaling.png"
