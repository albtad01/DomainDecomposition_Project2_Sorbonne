#!/bin/bash
# Test the validation script with MPI solver comparison

set -e  # Exit on error

echo "========================================================================"
echo "Testing Domain Decomposition Solver Validation with MPI"
echo "========================================================================"
echo

# Change to analyses directory
cd "$(dirname "$0")/../analyses" || exit 1

echo "Running validation with all solvers (including MPI)..."
echo "Using small mesh (m=16) for fast validation"
echo

python validate_solvers.py \
    --mesh-size 16 \
    --subdomains 4 \
    --wavenumber 16.0 \
    --sources 2 \
    --tolerance 1e-6 \
    --max-iterations 300 \
    --nprocs 4 \
    --all \
    --save-results

exit_code=$?

echo
echo "========================================================================"
if [ $exit_code -eq 0 ]; then
    echo "✓ VALIDATION PASSED - All solvers agree!"
else
    echo "✗ VALIDATION FAILED - Check output above"
fi
echo "========================================================================"

exit $exit_code
