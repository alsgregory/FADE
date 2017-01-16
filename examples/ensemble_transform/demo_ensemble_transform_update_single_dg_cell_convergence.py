""" demo for the posterior convergence of a single ensemble transform update step on a single cell """

from __future__ import division

from __future__ import absolute_import

from fade import *

import numpy as np

import matplotlib.pyplot as plot


# design the fade type of mesh
mesh = FadeMesh("UnitIntervalMesh", 1)

# generate function space
V = FunctionSpace(mesh, 'DG', 0)

# the coordinates of observation (only cell)
coords = tuple([np.array([0.5])])
obs = tuple([0.1])

sigma = 2.0

observation_operator = Observations(V, sigma)

# denote the true mean of the posterior
TrueMean = 0.7

# range of sample sizes
ns = 4 * (2 ** np.linspace(0, 5, 6))

# preallocate rmse array
rmse = np.zeros(len(ns))


# define the ensemble transform update step
def ensemble_transform_step(V, n, observation_operator, coords, obs):

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
    observation_operator.update_observation_operator(coords, obs)
    weights = weight_update(ensemble, weights, observation_operator)
    X = ensemble_transform_update(ensemble, weights, r_loc)

    # generate mean
    M = 0
    for i in range(n):
        M += (1 / float(n)) * X[i].dat.data[0]
    return M


niter = 10

for i in range(len(ns)):

    temp_mse = np.zeros(niter)

    for j in range(niter):

        k = ensemble_transform_step(V, int(ns[i]), observation_operator, coords, obs)

        temp_mse[j] = np.square(k - TrueMean)

    rmse[i] = np.sqrt(np.mean(temp_mse))

    print 'completed n = ', ns[i], ' sample size iteration'

plot.loglog(ns, rmse, 'r*-')
plot.loglog(ns, 8e-1 * ns ** (- 1.0 / 2.0), 'k--')
plot.legend(['rmse', 'sqrt decay'])
plot.xlabel('sample size')
plot.ylabel('rmse')
plot.show()
