#!/usr/bin/env bash
set -euo pipefail

# Fixed-point omega study runner + convergence analysis

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="$ROOT_DIR/src"
ANALYSIS_DIR="$ROOT_DIR/analyses"
EXP_RESULTS_DIR="$ROOT_DIR/results/fixed-point_omage"
FIGURES_DIR="$ROOT_DIR/figures/fixed-point_omage"
mkdir -p "$FIGURES_DIR"

# Fixed parameters
MESH_SIZE=32
SUBDOMAINS=4
WAVENUMBER=16
SOURCES=8
TOLERANCE=1e-8
MAX_ITER=100

# Ten omega values
OMEGAS=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

echo "Running fixed-point omega study..."
for OMEGA in "${OMEGAS[@]}"; do
	echo "  omega=${OMEGA}"
	(cd "$SRC_DIR" && python main.py \
		-a fixed-point \
		-m "$MESH_SIZE" \
		-J "$SUBDOMAINS" \
		-k "$WAVENUMBER" \
		-s "$SOURCES" \
		--tolerance "$TOLERANCE" \
		--max-iterations "$MAX_ITER" \
		--omega "$OMEGA" \
		--output-dir "$EXP_RESULTS_DIR" \
		--save-plots)
done

echo "Generating omega sweep plot..."
(cd "$ANALYSIS_DIR" && python plot_convergence.py \
	-m "$MESH_SIZE" \
	-J "$SUBDOMAINS" \
	-k "$WAVENUMBER" \
	--results-dir "$EXP_RESULTS_DIR" \
	--skip-comparison \
	--plot-omega-sweep \
	--output "$FIGURES_DIR/convergence_omega_study_m${MESH_SIZE}_J${SUBDOMAINS}_k${WAVENUMBER}.png")

echo "Done."
