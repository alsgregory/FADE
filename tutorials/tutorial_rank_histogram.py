# Tutorial of the multidimensional rank histogram in FADE
# =======================================================
#
# This is a tutorial on how to verify spatial dependencies
# and the calibration of an ensemble forecast given by an
# ensemble of functions in FADE. We use observations, taken
# from a reference function to be able to do this.
#
# Each function in an ensemble, :math:`f \in V`, where :math:`V`
# is a function space, needs to remain in the same 'position'
# in the ensemble during the computation of the histogram.
#
# This means that it can't be used alongside an Ensemble Transform
# update that exists in FADE, rather it should be used for the post
# processing of assimilated ensembles forecasts or ensemble
# forecasts without data assimilation.
#
# For this tutorial, we will just use an ensemble that isn't
# altered throughout time via a data assimilation process, and
# assume that observations are only used for the verification
# process.
#
# Firstly, let's generate a mesh and a function space.

from __future__ import division
from __future__ import absolute_import

from fade import *

import numpy as np


mesh = UnitIntervalMesh(3)
V = FunctionSpace(mesh, 'DG', 0)

# Next we need some data. Here, lists of observations, taken
# at `ny` coordinates in the domain from a reference function
# are generated at `nty` times. At each of these time-steps,
# the reference function is just assigned a scalar value drawn
# randomly from a standard normal distribution.

ny = 2
nty = 200

coords = []
observations = []

ref = Function(V)

for i in range(nty):
    coords.append([])
    observations.append([])
    ref.assign(np.random.normal(0, 1))
    for j in range(ny):
        coords[i].append(np.array([np.random.uniform(0, 1)]))
        observations[i].append(ref.at(coords[i][j]))

# To make an ensemble, a list or tuple of `n` functions can be
# generated.

n = 40
ensemble = []
for j in range(n):
    f = Function(V)
    ensemble.append(f)

# Now we can initialize the rank histogram class. This will store
# the ranks (normalized by :math:`n + 1`) in, and can be indexed
# directly; `R[i]` will return the `i'th` rank for example. It can also
# be used to plot the histogram as we will do in a few steps later.

R = rank_histogram(V, n)

# For each time-step we assign a normally distributed random increment
# to each function in the ensemble and compute the rank (normalized)
# by using the instance `compute_rank` from the initialized class.
#
# We can then plot the histogram as done below.

for i in range(nty):
    for j in range(n):
        ensemble[j].assign(np.random.normal(0, 1))
    R.compute_rank(ensemble, coords[i], observations[i])

R.plot_histogram()
