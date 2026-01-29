# Analysis README

This document explains how to perform convergence analysis and relaxation-parameter studies for the domain decomposition solvers using the existing results and analysis script.

## What is analyzed

1. **Convergence comparison (GMRES vs Fixed-Point)**
   - Uses the residual history recorded in each run.
   - Compares both methods on the **same experiment parameters** (m, J, κ).

2. **Relaxation parameter study (Fixed-Point only)**
   - Overlays convergence curves for multiple ω values.
   - Shows how ω impacts convergence speed and stagnation.

## Where the data comes from

Each run saves a `metrics.json` file under the results directory (default: `../results/`).
The analysis script searches these files and selects the **latest** matching run for the given parameters.

Relevant fields in `metrics.json`:
- `algorithm`: `gmres` or `fixed-point`
- `mesh_size`: m
- `subdomains`: J
- `wavenumber`: κ
- `omega`: ω (only for fixed-point)
- `residual_history`: list of residual norms per iteration

## Convergence plotting (GMRES vs Fixed-Point)

The script automatically finds the two matching runs (same m, J, κ) and plots the residual histories on a semilog scale.

Run (from this folder):
- `python plot_convergence.py -m 32 -J 4 -k 16`
- Optionally select a specific ω for fixed-point:
  - `python plot_convergence.py -m 32 -J 4 -k 16 --omega 0.8`
- Save output:
  - `python plot_convergence.py -m 32 -J 4 -k 16 --output convergence.png`

The plot title includes the ω value for the fixed-point run.

## Relaxation-parameter sweep (Fixed-Point only)

To compare multiple ω values for the same experiment:
- `python plot_convergence.py -m 32 -J 4 -k 16 --plot-omega-sweep`

If an output is specified, a second file is saved automatically:
- `convergence.png`
- `convergence_omega_sweep.png`

## Notes on expected behavior

- **GMRES** typically converges much faster and to lower residuals than fixed-point.
- **Fixed-point** often converges slowly and can stagnate, especially for larger κ or suboptimal ω.
- Large initial residuals for fixed-point are normal due to initialization and scaling.

## Recommended workflow

1. Run GMRES and fixed-point with identical parameters (m, J, κ).
2. Run fixed-point with multiple ω values (e.g., 0.5, 0.8, 1.0, 1.2).
3. Use `plot_convergence.py` to generate:
   - GMRES vs fixed-point comparison
   - Fixed-point ω sweep

## Example

- `python ../src/main.py -a gmres -m 32 -J 4 -k 16`
- `python ../src/main.py -a fixed-point -m 32 -J 4 -k 16 --omega 0.8`
- `python ../src/main.py -a fixed-point -m 32 -J 4 -k 16 --omega 1.0`
- `python plot_convergence.py -m 32 -J 4 -k 16 --plot-omega-sweep --output convergence.png`
