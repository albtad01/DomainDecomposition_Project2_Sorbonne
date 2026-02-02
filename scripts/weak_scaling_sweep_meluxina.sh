#!/bin/bash
#SBATCH --job-name=DD_StrongScaling_MPI
#SBATCH --account=p200981
#SBATCH --partition=cpu
#SBATCH --qos=default
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --time=08:00:00
#SBATCH --output=logs/strong_scaling_mpi_%j.out

mkdir -p logs

cd ..

SRC_DIR="./src"
EXP_RESULTS_DIR="./results/strong_scaling_mpi_${SLURM_JOB_ID:-local}"
echo "${EXP_RESULTS_DIR}"
mkdir -p "$EXP_RESULTS_DIR"

# --- Initialize module system on compute nodes ---
if [ -f /etc/profile ]; then
    source /etc/profile
fi
if ! command -v module >/dev/null 2>&1; then
    [ -f /etc/profile.d/lmod.sh ] && source /etc/profile.d/lmod.sh
fi
if ! command -v module >/dev/null 2>&1; then
    [ -f /etc/profile.d/modules.sh ] && source /etc/profile.d/modules.sh
fi

if ! command -v module >/dev/null 2>&1; then
    echo "ERROR: 'module' command not found on this node."
    exit 1
fi

echo "=== JOB INFO ==="
echo "Host: $(hostname)"
echo "JobID: ${SLURM_JOB_ID:-}"
echo "Requested ntasks: ${SLURM_NTASKS:-}"
echo "==============="

module --force purge
module load env/release/2024.1
module load Python/3.12.3-GCCcore-13.3.0
module load SciPy-bundle/2024.05-gfbf-2024a
module load OpenMPI/5.0.3-GCC-13.3.0

echo "=== MODULE PYTHON ==="
which python
python -V
python -c "import sys; print(sys.executable)"
echo "====================="

#python3 -m venv venv
#source venv/bin/activate
#pip install -r requirements.txt

#if [ ! -f "$ROOT_DIR/venv/bin/activate" ]; then
#    echo "ERROR: venv/bin/activate not found in ${ROOT_DIR}."
#    exit 1
#fi
source venv/bin/activate

#pip install mpi4py

echo "=== VENV PYTHON ==="
which python
python -V
python -c "import sys; print(sys.executable)"
echo "==================="

if [ -z "${EBROOTPYTHON:-}" ] || [ ! -d "${EBROOTPYTHON}/lib" ]; then
    echo "ERROR: EBROOTPYTHON is not set correctly (modules not loaded?)."
    echo "EBROOTPYTHON='${EBROOTPYTHON:-}'"
    exit 1
fi
export LD_LIBRARY_PATH="${EBROOTPYTHON}/lib:${LD_LIBRARY_PATH:-}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1



set -e

FIGURES_DIR="./figures/weak_scaling"
ANALYSIS_DIR="./analyses"
mkdir -p "$FIGURES_DIR"
mkdir -p "$ANALYSIS_DIR"

# Base parameters
BASE_MESH_SIZE=32  # Mesh size for J=1
WAVENUMBER=16
TOLERANCE=1e-4
OMEGA=0.1  # For fixed-point

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
for J in 2 4 8 16 32; do
    m=$(get_mesh_size $J)
    echo "  J=${J}, m=${m}"
done
echo ""

# Run GMRES for all configurations
echo "Running GMRES experiments..."
SRC_DIR="./src"
for J in 2 4 8 16 32; do
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
for J in 2 4 8 16 32; do
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
SRC_DIR="./src"
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
