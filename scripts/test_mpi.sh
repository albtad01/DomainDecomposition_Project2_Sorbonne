#!/bin/bash

# Test script for MPI parallel solver
# Usage: bash scripts/test_mpi.sh

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SRC_DIR="$ROOT_DIR/src"

echo "========================================"
echo "Testing MPI Parallel Solver"
echo "========================================"
echo ""

# Test parameters
MESH_SIZE=64
SUBDOMAINS=4
LX=1.0
LY=2.0
OMEGA=0.1

echo "Test configuration:"
echo "  Mesh size:    $MESH_SIZE"
echo "  Subdomains:   $SUBDOMAINS"
echo "  Domain:       [0, $LX] x [0, $LY]"
echo "  Omega:        $OMEGA"
echo ""

# Test 1: Check if mpi4py is installed
echo "Test 1: Checking mpi4py installation..."
python3 -c "import mpi4py; print('  ✓ mpi4py is installed')" || {
    echo "  ✗ mpi4py not found. Install with: pip install mpi4py"
    exit 1
}
echo ""

# Test 2: Check if mpirun is available
echo "Test 2: Checking mpirun availability..."
which mpirun > /dev/null 2>&1 && echo "  ✓ mpirun found: $(which mpirun)" || {
    echo "  ✗ mpirun not found. Install MPI (e.g., brew install mpich or brew install open-mpi)"
    exit 1
}
echo ""

# Test 3: Run a small MPI test
echo "Test 3: Running MPI solver with 4 processes..."
echo ""

cd "$ROOT_DIR"

mpirun -np $SUBDOMAINS python "$SRC_DIR/main.py" \
    --mpi \
    --mesh-size $MESH_SIZE \
    --subdomains $SUBDOMAINS \
    --Lx $LX \
    --Ly $LY \
    --omega $OMEGA \
    --tolerance 1e-6 \
    --max-iterations 100 \
    --output-dir "./results/test_mpi" \
    --plot-global \
    --save-plots

echo ""
echo "========================================"
echo "MPI Test Complete!"
echo "========================================"
