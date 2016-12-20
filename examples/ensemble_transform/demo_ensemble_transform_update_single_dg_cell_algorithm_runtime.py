""" demo for the algorithm runtime growth of a single ensemble transform update step
on a single cell """

from __future__ import division

from __future__ import absolute_import

from firedrake_da import *

import numpy as np

import matplotlib.pyplot as plot

import time


# create the mesh hierarchy (needed for coarsening localisation)
mesh = UnitIntervalMesh(5)
mesh_hierarchy = MeshHierarchy(mesh, 1)

# set used mesh to the bottom mesh
mesh = mesh_hierarchy[-1]

V = FunctionSpace(mesh, 'DG', 0)
fs = FunctionSpace(mesh, 'DG', 0)

# the coordinates of observation (only cell)
ny = 3
coords = []
obs = []
for i in range(ny):
    coords.append(np.random.uniform(0, 1, 1))
    obs.append(np.random.uniform(0, 1, 1))

coords = tuple(coords)
obs = tuple(obs)

observation_operator = Observations(V)

# range of sample sizes
ns = 4 * (2 ** np.linspace(0, 7, 8))

# preallocate rmse array
time_rloc_0 = np.zeros(len(ns))
time_rloc_1 = np.zeros(len(ns))


# define the ensemble transform update step
def ensemble_transform_step(V, n, observation_operator, coords, obs, sigma, lf):

    # generate ensemble
    ensemble = []
    weights = []
    for i in range(n):
        f = Function(V).assign(np.random.normal(1, 1, 1)[0])
        g = Function(V).assign(1.0 / n)
        ensemble.append(f)
        weights.append(g)

    # generate posterior
    r_loc = 0
    weights = weight_update(ensemble, weights, observation_operator, coords, obs, sigma, r_loc)
    X = ensemble_transform_update(ensemble, weights, lf)

    # generate mean
    M = 0
    for i in range(n):
        M += (1 / float(n)) * X[i].dat.data[0]
    return M


sigma = 2.0

niter = 1

# generate localisation radii
lf0 = 0
lf1 = 1

for i in range(len(ns)):

    temp_time_rloc_0 = np.zeros(niter)
    temp_time_rloc_1 = np.zeros(niter)

    for j in range(niter):

        a0 = time.time()
        k_rloc_0 = ensemble_transform_step(V, int(ns[i]), observation_operator, coords, obs,
                                           sigma, lf0)
        b0 = time.time()
        a1 = time.time()
        k_rloc_1 = ensemble_transform_step(V, int(ns[i]), observation_operator, coords, obs,
                                           sigma, lf1)
        b1 = time.time()

        temp_time_rloc_0[j] = b0 - a0
        temp_time_rloc_1[j] = b1 - a1

    time_rloc_0[i] = np.mean(temp_time_rloc_0)
    time_rloc_1[i] = np.mean(temp_time_rloc_1)

    print 'completed n = ', ns[i], ' sample size iteration'

plot.loglog(ns, time_rloc_0, 'r*-')
plot.loglog(ns, time_rloc_1, 'bo-')
plot.loglog(ns, 1e-1 * ns, 'k--')
plot.loglog(ns, 1e-1 * ns ** 1.5, 'k-')
plot.legend(['radius 0 localisation', 'radius 1 localisation', 'O(N)', 'O(N ^ 1.5)'])
plot.xlabel('sample size')
plot.ylabel('runtime (seconds)')
plot.show()
