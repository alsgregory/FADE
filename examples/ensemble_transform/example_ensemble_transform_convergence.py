""" Demo showing the convergence of posterior functions after an ensemble transform update.
Functions are DG0 functions on an interval mesh with a single cell.
Each function takes a normally distributed scalar value. """

from __future__ import division

from __future__ import absolute_import

from fade import *

import numpy as np

import matplotlib.pyplot as plot


# generate a mesh
mesh = UnitIntervalMesh(1)

# generate function space
V = FunctionSpace(mesh, 'DG', 0)

# the coordinates of observation (only cell)
coords = tuple([np.array([0.5])])
obs = tuple([0.1])

# measurement error variance
R = 2.0

# initialize observation operator
observation_operator = Observations(V, R)

# denote the true mean of the posterior
TrueMean = 0.7

# range of sample sizes
ns = 4 * (2 ** np.linspace(0, 4, 5))

# preallocate rmse array
rmse = np.zeros(len(ns))


# define the ensemble transform update step
def ensemble_transform_step(V, n, observation_operator, coords, obs):

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
    weights = weight_update(ensemble, weights, observation_operator)
    X = ensemble_transform_update(ensemble, weights)

    # generate mean at cell containing coordinate
    M = 0
    index = X[i].ufl_domain().locate_cell(coords[0])
    for i in range(n):
        M += (1 / float(n)) * X[i].dat.data[index]
    return M


# define number of iterations for each posterior estimate
niter = 5

# iterate over ensemble sizes
for i in range(len(ns)):

    temp_mse = np.zeros(niter)

    for j in range(niter):

        k = ensemble_transform_step(V, int(ns[i]), observation_operator, coords, obs)

        temp_mse[j] = np.square(k - TrueMean)

    rmse[i] = np.sqrt(np.mean(temp_mse))

    print 'completed n = ', ns[i], ' sample size iteration'

# plot results
plot.loglog(ns, rmse, 'r*-')
plot.loglog(ns, 8e-1 * ns ** (- 1.0 / 2.0), 'k--')
plot.legend(['rmse', 'sqrt decay'])
plot.xlabel('sample size')
plot.ylabel('rmse')
plot.show()
