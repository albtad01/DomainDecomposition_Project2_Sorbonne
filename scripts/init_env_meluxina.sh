#!/bin/bash
# Initialize environment on cluster nodes (Meluxina-style modules + venv)

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

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

module --force purge
module load env/release/2024.1
module load Python/3.12.3-GCCcore-13.3.0
module load SciPy-bundle/2024.05-gfbf-2024a
module load OpenMPI/5.0.3-GCC-13.3.0

# Create venv if missing and install requirements
if [ ! -f "$ROOT_DIR/venv/bin/activate" ]; then
  echo "Creating venv in ${ROOT_DIR}/venv..."
  python -m venv "$ROOT_DIR/venv"
fi

source "$ROOT_DIR/venv/bin/activate"

if [ -f "$ROOT_DIR/requirements.txt" ]; then
  echo "Installing Python requirements..."
  pip install --upgrade pip
  pip install -r "$ROOT_DIR/requirements.txt"
else
  echo "WARNING: requirements.txt not found in ${ROOT_DIR}."
fi

if [ -z "${EBROOTPYTHON:-}" ] || [ ! -d "${EBROOTPYTHON}/lib" ]; then
  echo "ERROR: EBROOTPYTHON is not set correctly (modules not loaded?)."
  echo "EBROOTPYTHON='${EBROOTPYTHON:-}'"
  exit 1
fi
export LD_LIBRARY_PATH="${EBROOTPYTHON}/lib:${LD_LIBRARY_PATH:-}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "Environment initialized."
which python
python -V
