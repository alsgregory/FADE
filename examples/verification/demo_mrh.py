""" demo for the multidimensional rank histogram in Firedrake """

from __future__ import division
from __future__ import absolute_import

from firedrake import *

from fade import *

import numpy as np


# single cell
mesh = UnitIntervalMesh(1)

# function space
fs = FunctionSpace(mesh, 'DG', 0)

# obs and coords
ny = 1
nty = 1000

coords = []
observations = []

for i in range(nty):
    coords.append([])
    observations.append([])
    f = Function(fs).assign(np.random.normal(0, 1))
    for j in range(ny):
        coords[i].append(np.array([np.random.uniform(0, 1)]))
        observations[i].append(f.at(coords[i][j]))

# ensemble functions
n = 40
ensemble = []

for i in range(n):
    f = Function(fs).assign(np.random.normal(0, 1))
    ensemble.append(f)

# set-up, compute and plot rank histogram
R = rank_histogram(fs, n)

for i in range(nty):
    R.compute_rank(ensemble, coords[i], observations[i])

R.plot_histogram()
