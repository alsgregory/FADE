""" demo for the posterior convergence of a single kalman update step on a single cell """

from __future__ import division

from __future__ import absolute_import

from firedrake_da import *

import numpy as np

import matplotlib.pyplot as plot


mesh = UnitIntervalMesh(1)

V = FunctionSpace(mesh, 'DG', 0)
fs = FunctionSpace(mesh, 'DG', 0)

# the coordinates of observation (only cell)
coords = tuple([np.array([0.5, 0.5])])
obs = tuple([0.1])

# denote the true mean of the posterior
TrueMean = 0.7

# range of sample sizes
ns = 4 * (2 ** np.linspace(0, 5, 6))

# preallocate rmse array
rmse = np.zeros(len(ns))


# define the kalman update step
def kalman_step(V, fs, n, coords, obs, sigma):

    # generate ensemble
    ensemble = []
    for i in range(n):
        f = Function(V).assign(np.random.normal(1, 1, 1)[0])
        ensemble.append(f)

    # generate posterior
    X = Kalman_update(ensemble, coords, obs, sigma, fs)

    # generate mean
    M = 0
    for i in range(n):
        M += (1 / float(n)) * X[i].dat.data[0]

    return M


sigma = 2.0

niter = 10

for i in range(len(ns)):

    temp_mse = np.zeros(niter)

    for j in range(niter):

        k = kalman_step(V, fs, int(ns[i]), coords, obs, sigma)

        temp_mse[j] = np.square(k - TrueMean)

    rmse[i] = np.sqrt(np.mean(temp_mse))

    print 'completed n = ', ns[i], ' sample size iteration'

plot.loglog(ns, rmse, 'r*-')
plot.loglog(ns, 8e-1 * ns ** (- 1.0 / 2.0), 'k--')
plot.legend(['rmse', 'sqrt decay'])
plot.xlabel('sample size')
plot.ylabel('rmse')
plot.show()
