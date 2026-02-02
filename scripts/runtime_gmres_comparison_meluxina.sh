#!/bin/bash
#SBATCH --job-name=runtime_gmres_comparison
#SBATCH --account=p200981
#SBATCH --partition=cpu
#SBATCH --qos=default
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --time=08:00:00
#SBATCH --output=logs/runtime_gmres_comparison_%j.out

mkdir -p logs

cd ..

SRC_DIR="./src"
EXP_RESULTS_DIR="./results/runtime_gmres_comparison_${SLURM_JOB_ID:-local}"
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

FIGURES_DIR="./figures/runtime_gmres_comparison"
ANALYSIS_DIR="./analyses"
mkdir -p "$FIGURES_DIR"
mkdir -p "$ANALYSIS_DIR"


# Parameters
MESH_SIZES=(32 64 128)

SUBDOMAINS=(2 4 8 16)
WAVENUMBER=16
SOURCES=8
TOLERANCE=1e-8
SEED=42
Lx=1.0
Ly=2.0


for M in "${MESH_SIZES[@]}"; do

  echo "Running DD-GMRES (m=${M})"
  for J in "${SUBDOMAINS[@]}"; do
    (cd "$SRC_DIR" && python main.py \
      --algorithm gmres \
      --mesh-size "$M" \
      --subdomains "$J" \
      --wavenumber "$WAVENUMBER" \
      --sources "$SOURCES" \
      --tolerance "$TOLERANCE" \
      --seed "$SEED" \
      --Lx "$Lx" \
      --Ly "$Ly" \
      --output-dir "$EXP_RESULTS_DIR")
  done

  echo "Running baseline GMRES (m=${M})"
  (cd "$SRC_DIR" && python main.py \
    --algorithm baseline-gmres \
    --mesh-size "$M" \
    --subdomains 1 \
    --wavenumber "$WAVENUMBER" \
    --sources "$SOURCES" \
    --tolerance "$TOLERANCE" \
    --seed "$SEED" \
    --Lx "$Lx" \
    --Ly "$Ly" \
    --output-dir "$EXP_RESULTS_DIR")
done

(cd "$ANALYSIS_DIR" && python plot_runtime.py \
  --results-dir "$EXP_RESULTS_DIR" \
  --figures-dir "$FIGURES_DIR")

echo "Runtime comparison complete."
echo "Results: $EXP_RESULTS_DIR"
echo "Figures: $FIGURES_DIR"
