#!/bin/bash
#SBATCH --job-name=strong_512
#SBATCH --partition=az4-mixed
#SBATCH --qos=default
#SBATCH --nodes=4
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --time=00:02:00
#SBATCH --output=logs/strong_scaling_mpi_512_%j.out

mkdir -p logs

cd ..
SRC_DIR="./src"

echo "=== JOB INFO ==="
echo "Host: $(hostname)"
echo "JobID: ${SLURM_JOB_ID:-}"
echo "Requested ntasks: ${SLURM_NTASKS:-}"
echo "==============="

# load python module
# load OpenMPI module


source "venv/bin/activate"

echo "=== VENV PYTHON ==="
which python
python -V
python -c "import sys; print(sys.executable)"
echo "==================="


export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Base parameters
LIST_BASE_MESH_SIZE=(512)  # Mesh size for J=1
WAVENUMBER=16
TOLERANCE=1e-4
OMEGA=0.1

# Subdomain counts / MPI ranks
SUBDOMAINS=(1 2 4 8 16 32 64)

echo "=== Strong Scaling Study (MPI Fixed-Point) ==="
echo "Fixed DOF per subdomain (m/J ~ constant)"
echo "Wavenumber κ: ${WAVENUMBER}"
echo "Output: ${EXP_RESULTS_DIR}"
echo ""
echo "Configurations:"
for J in "${SUBDOMAINS[@]}"; do
    m=${LIST_BASE_MESH_SIZE[0]}
    echo "  J=${J}, m=${m}"
done

for BASE_MESH_SIZE in "${LIST_BASE_MESH_SIZE[@]}"; do
    echo "Running strong scaling with base mesh size: ${BASE_MESH_SIZE}"
    EXP_RESULTS_DIR="./results/strong_scaling_mpi_${BASE_MESH_SIZE}_${SLURM_JOB_ID:-local}"
    mkdir -p "$EXP_RESULTS_DIR"
    for J in "${SUBDOMAINS[@]}"; do
        m=$BASE_MESH_SIZE
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
                --no-solution)
    done
    echo "done for base mesh size: ${BASE_MESH_SIZE}"
done


echo ""
echo "=== Strong Scaling (MPI) Complete ==="
echo "Results saved to: ${EXP_RESULTS_DIR}"