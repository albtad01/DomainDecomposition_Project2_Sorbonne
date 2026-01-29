# Domain Decomposition Method for Helmholtz Equation

A high-performance parallel solver for the 2D Helmholtz equation using domain decomposition methods (DDM) with slab decomposition.

## Problem Description

We solve the following boundary value problem on $\Omega = (0, L_x) \times (0, L_y)$:

$$
\begin{cases}
(-\Delta - \kappa^2) u = f & \text{in } \Omega \\
(\partial_\mathbf{n} - i\kappa) u = g & \text{on } \Gamma
\end{cases}
$$

where:
- $\kappa > 0$ is the **wavenumber** (related to wavelength by $\lambda = 2\pi/\kappa$)
- $f(\mathbf{x})$ is a weighted sum of Gaussian point sources
- $g = 0$ (crude radiation boundary condition)
- The domain is discretized using $\mathbb{P}_1$ Lagrange finite elements on a uniform triangular mesh

## Solution Approach

The solver uses a **non-overlapping slab domain decomposition** with:

1. **Interface problem** (skeleton formulation):
   $$(\mathbb{I} + \boldsymbol{\Pi}\mathbb{S})\mathbb{p} = \mathbb{g}$$
   where $\mathbb{S}$ is the scattering operator and $\boldsymbol{\Pi}$ exchanges interface data between subdomains.

2. **Two solution algorithms:**
  - **Fixed-point iteration** (Richardson method, sequential): $\mathbb{p}_{n+1} = [(1-\omega)\mathbb{I} - \omega\boldsymbol{\Pi}\mathbb{S}]\mathbb{p}_n + \omega\mathbb{g}$
  - **GMRES** (Krylov subspace method, sequential): More robust and typically faster

3. **Local recovery**: Once the interface problem converges, solve local subdomain problems in parallel to recover the volume solution.

## Repository Structure

```
.
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── Project 2 Notes.pdf            # Detailed mathematical formulation
├── figures/                       # Output plots and visualizations
├── results/                       # Convergence studies and benchmark results
│   └── discretization/
├── analyses/                      # Plotting + validation scripts
│   ├── validate_solvers.py         # Validation script
│   ├── plot_convergence.py         # Convergence/strong/weak scaling plots
│   ├── plot_runtime.py             # Runtime comparison plots
│   ├── plot_discretization.py      # Discretization/mesh refinement plots
│   └── README.md                   # Analysis workflow notes
├── src/
│   ├── main.py                   # Main experiment runner
│   ├── dd_solver.py              # Sequential interface problem solvers
│   ├── dd_operators.py           # S operator, Π permutation, g vector
│   ├── dd_restrictions.py        # B_j, C_j restriction matrices
│   ├── dd_mesh.py                # Local mesh extraction and boundaries
│   ├── dd_local_problems.py      # Local system assembly (A_j, T_j, etc.)
│   ├── fem_core.py               # FEM assembly (mass, stiffness, etc.)
│   ├── discretization.py         # Global mesh generation
│   ├── code.py                   # Baseline direct solver example
│   └── sanity_checks_1.py        # Unit tests and validation
└── .git/                         # Version control
```

## Installation

### Prerequisites

- Python 3.8+
- pip or conda

### Setup (Virtual Environment)

```bash
# Clone/navigate to repository
cd DomainDecomposition_Project2_Sorbonne

# Create virtual environment
python3 -m venv hpc
source hpc/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

- **numpy** — Numerical arrays and linear algebra
- **scipy** — Sparse matrices, direct/iterative solvers (GMRES), sparse LU factorization
- **matplotlib** — Visualization and plotting

All dependencies are specified in `requirements.txt`.

## Quick Start

### Run Default Experiment

```bash
cd src
python main.py
```

This solves the problem with:
- GMRES solver
- $32 \times 64$ mesh
- 4 subdomains
- Wavenumber $\kappa = 16$
- 8 point sources

### Specify Algorithm

```bash
# Use fixed-point iteration (sequential)
python main.py --algorithm fixed-point

# Use GMRES (sequential, default)
python main.py --algorithm gmres

```
```

### Mesh Refinement

```bash
# Finer mesh (mesh_size=64)
python main.py -m 64

# Coarser mesh
python main.py -m 16
```

### Domain Decomposition Control

```bash
# Increase number of subdomains (default: 4)
python main.py --subdomains 8

# Shorter: -J flag
python main.py -J 8
```

### Problem Parameters

```bash
# Higher wavenumber (higher frequency)
python main.py -k 32

# Larger domain
python main.py --Lx 2.0 --Ly 3.0

# More sources
python main.py -s 16
```

### Solver Control

```bash
# Tighter tolerance
python main.py --tolerance 1e-10

# Different relaxation (fixed-point only)
python main.py -a fixed-point --omega 0.8

# More iterations allowed
python main.py --max-iterations 1000
```

### Visualization

```bash
# Plot global solution
python main.py --plot-global

# Plot local subdomain solutions
python main.py --plot-local

# Plot mesh structure
python main.py --plot-mesh

# All visualizations + verbose solver output
python main.py --plot-global --plot-local -v
```

### Complete Example

```bash
python main.py \
  --algorithm gmres \
  --mesh-size 48 \
  --subdomains 8 \
  --wavenumber 24 \
  --sources 12 \
  --tolerance 1e-9 \
  --plot-local \
  --verbose
```

## Key Classes and Modules

### `dd_operators.py`

- **`SubdomainData`** — Dataclass storing all precomputed data for one subdomain
  - Local mesh vertices/elements, local FEM matrices (A_j, T_j), factorization (lu), RHS (b_j)
  - Restriction/prolongation matrices (B_j, C_j)

- **`build_subdomains(Lx, Ly, Nx, Ny, J, kappa, ps)`** — Precomputes all subdomain data
  - Returns list of `SubdomainData` objects

- **`S_operator(p, subs)`** — Applies scattering operator $\mathbb{S}$ to interface vector $\mathbb{p}$
  - Solves local systems in parallel: $\mathbb{S}_j\mathbb{v}_j = \mathbb{B}_j^*\mathbb{T}_j\mathbb{p}_j$
  - Outputs: $\mathbb{p}_j + 2i\mathbb{B}_j\mathbb{v}_j$

- **`Pi_operator(x, Nx, J)`** — Permutation operator exchanging interface DOFs between neighbors
  - Swaps: top($\Omega_j$) ↔ bottom($\Omega_{j+1}$)

- **`g_vector(subs)`** — Computes RHS of interface problem: $\mathbb{g} = -2i\boldsymbol{\Pi}\mathbb{B}(\mathbb{A} - i\mathbb{B}^*\mathbb{T}\mathbb{B})^{-1}\mathbb{b}$

### `dd_solver.py` (Sequential Solvers)

- **`fixed_point_solver(subs, tol, maxiter, omega, verbose)`** — Richardson fixed-point iteration
  - Returns: converged interface solution $\mathbb{p}$ and convergence history

- **`gmres_interface_solver(subs, tol, maxiter, verbose)`** — GMRES Krylov solver
  - Uses `LinearOperator` to avoid assembling full interface system matrix
  - Returns: converged solution, convergence info, and residual history

- **`baseline_gmres_solver(subs, tol, verbose)`** — Full system GMRES (for comparison)
  - Solves global system directly without domain decomposition
  - Returns: global solution $\mathbb{u}$, iterations, solve time

- **`recover_local_solutions(subs, p)`** — Solves local subdomain problems given interface solution
  - Returns: dict mapping subdomain index $j$ to local solution $\mathbb{u}_j$

- **`assemble_global_solution(subs, u_dict)`** — Concatenates local solutions into global vector

### `analyses/validate_solvers.py` (Validation Script)

- **`run_direct_solver(...)`** — Execute baseline GMRES solver
- **`run_dd_fixed_point_solver(...)`** — Sequential fixed-point validation
- **`run_dd_gmres_solver(...)`** — Sequential GMRES validation
- **`compare_solutions(...)`** — Compute error metrics between solutions

### `dd_mesh.py`, `dd_restrictions.py`, `dd_local_problems.py`

Lower-level utilities:
- **`dd_mesh.py`** — Extract local mesh for each subdomain and identify boundary edges
- **`dd_restrictions.py`** — Build restriction/prolongation matrices ($\mathbb{B}_j$, $\mathbb{C}_j$)
- **`dd_local_problems.py`** — Assemble local FEM matrices ($\mathbb{A}_j$, $\mathbb{T}_j$) and factorize

### `fem_core.py`

Finite element assembly (mesh-independent):
- **`mesh(nx, ny, Lx, Ly)`** — Create uniform triangular mesh
- **`stiffness(vtx, elt)`** — Assemble stiffness matrix $K$
- **`mass(vtx, elt)`** — Assemble mass matrix $M$
- **`point_source(sp, kappa)`** — Gaussian point source term
- **`plot_mesh(vtx, elt, [u])`** — Visualization helper

## Output and Statistics

The solver reports:

```
HELMHOLTZ DOMAIN DECOMPOSITION SOLVER
==============================================================================

Configuration:
  Algorithm:       gmres
  Mesh:            Nx=33, Ny=65
  Domain:          [0, 1.0] × [0, 2.0]
  Subdomains:      4
  Wavenumber κ:    16.0
  Wavelength λ:    0.3927

Building subdomain data...
  Done in 0.245s
  Total DOFs (all subdomains): 2145
  Interface DOFs: 128
  DOFs per subdomain: [513, 650, 650, 332]

Solving interface problem with gmres...
  Converged in 47 iterations
  Final residual: 8.3456e-09
  Solve time: 1.234s

Recovering local solutions...
  Done in 0.089s

Solution statistics:
  Global solution norm: 1.2345e-02
  Global solution max:  5.6789e-03
  Global solution min:  1.2345e-05
```

## Performance Characteristics

### Scaling

- **DOFs per subdomain** grows with mesh refinement: $O(N_x \cdot N_y/J)$
- **Interface DOFs** fixed per interface: $N_x$ per boundary
- **Total interface system size**: $O(J \cdot N_x)$ (small compared to global system $O(N_x \cdot N_y)$)

### Solution Methods

| Method | Pros | Cons |
|--------|------|------|
| **Fixed-Point (Sequential)** | Simple implementation | Can stagnate; slow convergence for ill-conditioned problems |
| **GMRES (Sequential)** | Faster, robust, theoretically guaranteed convergence | Slightly more overhead; needs Krylov space storage |

### Parallelization

The code is structured for parallelization:
- Each `S_operator` application solves $J$ **independent** local systems
- Each local solve (SuperLU factorization) can run in parallel
- `Pi_operator` is a pure data permutation

**Implementation:**
- **Sequential** (`dd_solver.py`): Fixed-point and GMRES on single process

**Future parallelization options:**
- GPU acceleration via `CuPy` for large-scale problems

## Validation and Testing

### Validate Sequential Solvers Against Baseline GMRES

```bash
cd analyses

# Run validation with default parameters
python validate_solvers.py

# Custom problem size
python validate_solvers.py --mesh-size 48 --subdomains 8

# Save validation results
python validate_solvers.py --save-results --verbose

# Quick test with coarse mesh
python validate_solvers.py -m 16 -J 2 -v
```

The validation script:
1. Runs baseline GMRES on full system (sequential)
2. Runs sequential fixed-point and/or GMRES DD solvers
3. Compares solutions with detailed error metrics
4. Reports PASS/FAIL based on relative error and correlation

### Unit Tests

```bash
cd src
python sanity_checks_1.py
```

### Baseline Direct Solver

For small problems, compare with direct solver:

```bash
python code.py
```

## References

- **Project specification**: `Project 2 Notes.pdf`
- **Mathematical foundation**: Sorbonne University HPC/Linear Algebra course materials
- **Domain decomposition background**: Smith, Bjørstad, Gropp (2004) "Domain Decomposition"

## Troubleshooting

### GMRES does not converge

- Increase `--max-iterations`
- Lower `--tolerance` (less strict)
- Check problem is well-posed: try lower wavenumber `--wavenumber 8`
- Try fixed-point: `--algorithm fixed-point`

### Memory issues with large meshes

- Reduce `--mesh-size`
- Increase `--subdomains` (more parallelism-friendly)
- Use sparse matrix format (already done internally)

### Slow fixed-point convergence

- Try `--omega 0.5` (damping)
- Switch to GMRES (typically 5-10× faster)

## MPI Parallel Solver Guide

### Overview

The MPI parallel solver distributes the domain decomposition computation across multiple processes, with each MPI rank owning ONE subdomain. This enables parallel execution on multi-core machines or HPC clusters.

### Installation

#### Prerequisites

1. **Python packages**:
  ```bash
  pip install numpy scipy matplotlib mpi4py
  ```

2. **MPI implementation**:
  - **macOS**: `brew install mpich` or `brew install open-mpi`
  - **Linux**: `sudo apt-get install mpich` or `sudo apt-get install openmpi-bin`
  - **Windows**: Use Microsoft MPI or install via WSL

### Usage

#### Basic Usage

Run with MPI using the `--mpi` flag:

```bash
mpirun -np 4 python src/main.py --mpi --mesh-size 64 --subdomains 4
```

Or use the algorithm selection:

```bash
mpirun -np 4 python src/main.py --algorithm mpi-fixed-point --subdomains 4
```

**IMPORTANT**: The number of MPI processes (`-np`) MUST equal the number of subdomains (`--subdomains`).

#### Command-Line Options

All standard options from `main.py` are supported:

```bash
mpirun -np 8 python src/main.py \
   --mpi \
   --mesh-size 128 \
   --subdomains 8 \
   --Lx 2.0 \
   --Ly 2.0 \
   --wavenumber 16.0 \
   --omega 0.8 \
   --tolerance 1e-6 \
   --max-iterations 200 \
   --verbose \
   --plot-global \
   --output-dir ./results/mpi_run
```

#### Test Script

```bash
bash scripts/test_mpi.sh
```

#### Scaling Scripts (MPI)

- `scripts/strong_scaling_mpi.sh`
- `scripts/weak_scaling_mpi.sh`

### How It Works

#### Algorithm Structure

```
Initialization (once):
├─ All ranks: Build all subdomain data with same seed (deterministic)
├─ Each rank: Extract its own subdomain (rank j gets subdomain j)
└─ Each rank: Compute local g_j with one MPI exchange

Fixed-point iteration (loop):
├─ STEP 1 (LOCAL):  Each rank computes S_j(p_j)        [No communication]
├─ STEP 2 (MPI):    Exchange interface data with neighbors via Π
├─ STEP 3 (LOCAL):  Each rank updates p_j^{n+1}       [No communication]
└─ STEP 4 (MPI):    Check global convergence via MPI.Allreduce

Finalization:
├─ Each rank: Reconstruct local solution u_j from p_j
├─ Gather: Rank 0 collects all solutions
└─ Rank 0: Save results and plots
```

**Note on Data Distribution**: All ranks build the complete subdomain data independently (using the same random seed for deterministic results). This avoids MPI serialization issues with the SuperLU factorization objects. Since `build_subdomains()` is deterministic, all ranks get identical data. Each rank then extracts only its subdomain for computation.

#### Communication Pattern

For slab decomposition in y-direction with J subdomains:

- **Rank 0** (bottom): Has only TOP interface → exchanges with Rank 1
- **Rank j** (internal): Has BOTTOM and TOP interfaces → exchanges with Ranks j-1 and j+1
- **Rank J-1** (top): Has only BOTTOM interface → exchanges with Rank J-2

### Troubleshooting (MPI)

#### Error: "MPI size must equal number of subdomains"

**Problem**: The number of MPI processes doesn't match `--subdomains`.

**Solution**: Make sure `-np` matches `-J`:
```bash
mpirun -np 4 python src/main.py --mpi -J 4  # Correct
mpirun -np 8 python src/main.py --mpi -J 4  # Wrong
```

#### Error: "mpi4py not available"

**Problem**: The `mpi4py` package is not installed.

**Solution**:
```bash
pip install mpi4py
```

#### Error: "mpirun: command not found"

**Problem**: No MPI implementation is installed.

**Solution**:
- macOS: `brew install mpich`
- Linux: `sudo apt-get install mpich`

## Contributing

Improvements welcome! Areas for enhancement:

- [ ] Overlapping domain decomposition
- [ ] Adaptive mesh refinement
- [ ] Preconditioning strategies
- [ ] Higher-order finite elements
- [ ] 3D extension
- [ ] GPU acceleration

## License

Academic use (Sorbonne University)

## Authors

- Leon Ackermann (Implementation)
- Based on course materials from HPC/Linear Algebra 2024-2025
