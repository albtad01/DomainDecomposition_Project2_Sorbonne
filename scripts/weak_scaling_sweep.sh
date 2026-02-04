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
BASE_MESH_SIZE=128  # Fixed mesh size
BASE_LY=2.0        # Base domain height for J=1
WAVENUMBER=16
TOLERANCE=1e-4
OMEGA=0.1  # For fixed-point

# Subdomain counts (powers of 2)
# To maintain constant DOF per subdomain: Ly scales with J
# J=1: Ly=2.0 (baseline)
# J=2: Ly=4.0
# ...

# Calculate base nodes in Y (minus 1) for J=1
# BASE_LY * BASE_MESH_SIZE should be integer for clean scaling
# For Ly=2.0, m=32 -> 64 intervals
BASE_NY_MO=$(python3 -c "print(int($BASE_LY * $BASE_MESH_SIZE))")

get_Ly() {
    local J=$1
    python3 -c "print(float($BASE_LY * $J))"
}

get_Ny() {
    local J=$1
    echo $((BASE_NY_MO * J + 1))
}

echo "=== Weak Scaling Study ==="
echo "Fixed DOF per subdomain (Ly ~ J, Ny ~ J, m constant)"
echo "Wavenumber κ: ${WAVENUMBER}"
echo "Mesh Size m: ${BASE_MESH_SIZE}"
echo ""
echo "Configurations:"
for J in 2 4 8 16 32; do
    Ly=$(get_Ly $J)
    Ny=$(get_Ny $J)
    echo "  J=${J}, Ly=${Ly}, Ny=${Ny}"
done
echo ""

echo "Running GMRES experiments..."
SRC_DIR="$ROOT_DIR/src"
for J in 2 4 8 16 32; do
    #Ly=$(get_Ly $J)
    Ny=$(get_Ny $J)
    echo "  J=${J}, Ly=${Ly}, Ny=${Ny}..."
    (cd "$SRC_DIR" && python main.py \
        --mesh-size ${BASE_MESH_SIZE} \
        --Ly 2 \
        --Ny ${Ny} \
        --subdomains ${J} \
        --wavenumber ${WAVENUMBER} \
        --algorithm gmres \
        --tolerance ${TOLERANCE} \
        --output-dir "${EXP_RESULTS_DIR}" \
        --no-solution)
done

echo ""
echo "Running Fixed-Point experiments..."
for J in 2 4 8 16 32; do
    Ly=$(get_Ly $J)
    Ny=$(get_Ny $J)
    echo "  J=${J}, Ly=${Ly}, Ny=${Ny}..."
    (cd "$SRC_DIR" && python main.py \
        --mesh-size ${BASE_MESH_SIZE} \
        --Ly 2 \
        --Ny ${Ny} \
        --subdomains ${J} \
        --wavenumber ${WAVENUMBER} \
        --algorithm fixed-point \
        --omega ${OMEGA} \
        --max-iterations 1000 \
        --tolerance ${TOLERANCE} \
        --output-dir "${EXP_RESULTS_DIR}" \
        --no-solution)
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
