#! /usr/bin/python3
from math import pi

import numpy as np
na = np.newaxis
import numpy.linalg as la
import scipy.sparse.linalg as spla
from scipy.sparse import csr_matrix, csc_matrix
from fem_core import mesh, boundary, mass, stiffness, point_source, plot_mesh

from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

## Example resolution of model problem
Lx = 1           # Length in x direction
Ly = 2           # Length in y direction
nx = 1 + Lx * 32 # Number of points in x direction
ny = 1 + Ly * 32 # Number of points in y direction
k = 16           # Wavenumber of the problem
ns = 8           # Number of point sources + random position and weight below
sp = [np.random.rand(3) * [Lx, Ly, 50.0] for _ in np.arange(ns)]
vtx, elt = mesh(nx, ny, Lx, Ly)
belt = boundary(nx, ny)
M = mass(vtx, elt)
Mb = mass(vtx, belt)
K = stiffness(vtx, elt)
A = K - k**2 * M - 1j*k*Mb      # matrix of linear system 
b = M @ point_source(sp,k)(vtx) # linear system RHS (source term)
x = spla.spsolve(A, b)          # solution of linear system via direct solver

# GMRES
residuals = [] # storage of GMRES residual history
def callback(x):
    residuals.append(x)
y, _ = spla.gmres(A, b, rtol=1e-12, callback=callback, callback_type='pr_norm', maxiter=200)
print("Total number of GMRES iterations = ", len(residuals))
print("Direct vs GMRES error            = ", la.norm(y - x))

# Plots
plot_mesh(vtx, elt) # slow for fine meshes
plt.show()
plot_mesh(vtx, elt, np.real(x))
plt.colorbar()
plt.show()
plot_mesh(vtx, elt, np.abs(x))
plt.colorbar()
plt.show()
plt.semilogy(residuals)
plt.show()