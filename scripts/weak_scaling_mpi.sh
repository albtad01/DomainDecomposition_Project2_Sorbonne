#!/bin/bash
#SBATCH --job-name=DD_WeakScaling_MPI
#SBATCH --account=p200981
#SBATCH --partition=cpu
#SBATCH --qos=default
#SBATCH --nodes=64
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --time=08:00:00
#SBATCH --output=logs/weak_scaling_mpi_%j.out

set -euo pipefail
mkdir -p logs

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="$ROOT_DIR/src"
EXP_RESULTS_DIR="$ROOT_DIR/results/weak_scaling_mpi_${SLURM_JOB_ID:-local}"
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

if [ ! -f "$ROOT_DIR/venv/bin/activate" ]; then
    echo "ERROR: venv/bin/activate not found in ${ROOT_DIR}."
    exit 1
fi
source "$ROOT_DIR/venv/bin/activate"

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

# Base parameters
BASE_MESH_SIZE=32  # Mesh size for J=1
WAVENUMBER=16
TOLERANCE=1e-4
OMEGA=0.1

# Subdomain counts / MPI ranks
SUBDOMAINS=(2 4 8 16 32 64)

# Function to get mesh size for given J (portable bash 3.x compatible)
get_mesh_size() {
    local J=$1
    echo $((BASE_MESH_SIZE * J))
}

echo "=== Weak Scaling Study (MPI Fixed-Point) ==="
echo "Fixed DOF per subdomain (m/J ~ constant)"
echo "Wavenumber κ: ${WAVENUMBER}"
echo "Output: ${EXP_RESULTS_DIR}"
echo ""
echo "Configurations:"
for J in "${SUBDOMAINS[@]}"; do
    m=$(get_mesh_size $J)
    echo "  J=${J}, m=${m}"
done
echo ""

for J in "${SUBDOMAINS[@]}"; do
    m=$(get_mesh_size $J)
    echo "  J=${J}, m=${m} (srun -n ${J})..."
    (cd "$SRC_DIR" && srun --ntasks="${J}" --kill-on-bad-exit=1 --export=ALL,LD_LIBRARY_PATH \
        python main.py \
            --mesh-size ${m} \
            --subdomains ${J} \
            --wavenumber ${WAVENUMBER} \
            --algorithm mpi-fixed-point \
            --mpi \
            --omega ${OMEGA} \
            --tolerance ${TOLERANCE} \
            --output-dir "${EXP_RESULTS_DIR}" \
            --save-plots)
done

echo ""
echo "=== Weak Scaling (MPI) Complete ==="
echo "Results saved to: ${EXP_RESULTS_DIR}"
