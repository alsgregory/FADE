""" Demo showing the runtime of the ensemble transform algorithm in FADE over increasing ensemble sizes.
Two different radii of localisation are considered, to see the additional cost of localisation.
Functions are DG0 functions on an interval mesh.
Each function takes a normally distributed scalar value. """

from __future__ import division

from __future__ import absolute_import

from fade import *

import numpy as np

import matplotlib.pyplot as plot

import time


# generate a mesh
mesh = UnitIntervalMesh(5)

# generate function space
V = FunctionSpace(mesh, 'DG', 0)

# the coordinates of observation
ny = 3
coords = []
obs = []
for i in range(ny):
    coords.append(np.random.uniform(0, 1, 1))
    obs.append(np.random.uniform(0, 1, 1))

coords = tuple(coords)
obs = tuple(obs)

# measurement error variance
R = 2.0

# initialize observation operator
observation_operator = Observations(V, R)

# range of sample sizes
ns = 4 * (2 ** np.linspace(0, 6, 7))

# preallocate runtime arrays
time_rloc_0 = np.zeros(len(ns))
time_rloc_1 = np.zeros(len(ns))


# define the ensemble transform update step
def ensemble_transform_step(V, n, observation_operator, coords, obs, r_loc):

    # generate ensemble
    ensemble = []
    weights = []
    for i in range(n):
        f = Function(V).assign(np.random.normal(1, 1))
        g = Function(V).assign(1.0 / n)
        ensemble.append(f)
        weights.append(g)

    # generate posterior
    observation_operator.update_observation_operator(coords, obs)
    weights = weight_update(ensemble, weights, observation_operator, r_loc)
    ensemble_transform_update(ensemble, weights, r_loc)


# iterate over ensemble sizes
for i in range(len(ns)):

    a0 = time.time()
    ensemble_transform_step(V, int(ns[i]), observation_operator, coords, obs, 0)
    b0 = time.time()

    a1 = time.time()
    ensemble_transform_step(V, int(ns[i]), observation_operator, coords, obs, 1)
    b1 = time.time()

    time_rloc_0[i] = b0 - a0
    time_rloc_1[i] = b1 - a1

    print 'completed n = ', ns[i], ' sample size iteration'

# plot results
plot.loglog(ns, time_rloc_0, 'r*-')
plot.loglog(ns, time_rloc_1, 'bo-')
plot.loglog(ns, 1e-1 * ns, 'k--')
plot.loglog(ns, 1e-1 * ns ** 1.5, 'k-')
plot.legend(['radius 0 localisation', 'radius 1 localisation', 'O(N)', 'O(N ^ 1.5)'])
plot.xlabel('sample size')
plot.ylabel('runtime (seconds)')
plot.show()
