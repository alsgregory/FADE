""" demo for sequential importance sampling on an ensemble of functions with single cell - scalar
OU process """

from __future__ import division

from __future__ import absolute_import

from fade import *

import numpy as np

import matplotlib.pyplot as plot


# generate a mesh with a single cell from a mesh hierarchy
mesh_hierarchy = MeshHierarchy(UnitIntervalMesh(1), 0)
mesh = mesh_hierarchy[-1]

# generate function space
V = FunctionSpace(mesh, 'DG', 0)

# generate a reference function and observation function (with measurement error)
ref = Function(V)
obs = Function(V)

# generate an ensemble, sized 50
N = 50
ensemble = []
for i in range(N):
    f = Function(V)
    ensemble.append(f)

# define the initial prior weights
weights = []
for i in range(N):
    w = Function(V).assign(1.0 / N)
    weights.append(w)

# define an observation operator with independent measurement error variance R
R = 2e-1
observation_operator = Observations(V, R)

# define the number of assimilation steps and time-intervals of assimilation
ny = 50
delta_t = 1

# define OU process parameters
alpha = -0.1
sigma = 0.2
dt = 5e-1
nt = int((ny * delta_t) / dt)

# make lists for observations and the coordinate each was taken from
coordinates = []
observations = []

# preallocate reference functions
references = []

# propagate reference solution over time and take observations
c = 0
for k in range(nt):
    ref.dat.data[:] += ((alpha * dt * ref.dat.data[:]) +
                        (np.sqrt(dt) * sigma * np.random.normal(0, 1, len(ref.dat.data[:]))))
    if ((k + 1) * dt) % delta_t == 0:

        # store reference solution
        f = Function(V).assign(ref)
        references.append(f)

        # take observations (these have to be inside lists for each assimilation step)
        obs.dat.data[:] = (ref.dat.data[:] +
                           np.random.normal(0, np.sqrt(R), len(ref.dat.data[:])))
        coordinates.append([np.array([np.random.uniform(0, 1)])])
        observations.append([obs.at(coordinates[c])])

        c += 1

# preallocate effective sample size
eff = np.zeros(ny)

# preallocate error of weighted mean of ensemble
error = np.zeros(ny)

# propagate ensemble over time and update weights at each assimilation step
c = 0
for k in range(nt):
    for j in range(N):
        ensemble[j].dat.data[:] += ((alpha * dt * ensemble[j].dat.data[:]) +
                                    (np.sqrt(dt) * sigma *
                                     np.random.normal(0, 1, len(ensemble[j].dat.data[:]))))
    if ((k + 1) * dt) % delta_t == 0:

        # update observation operator
        observation_operator.update_observation_operator(coordinates[c], observations[c])

        # update weights
        weights = weight_update(ensemble, weights, observation_operator)

        # compute weighted mean
        mean = Function(V)
        sums = Function(V)
        for j in range(N):
            mean += assemble(weights[j] * ensemble[j])
            sums += weights[j]

        # check that all weights sum to 1
        assert norm(assemble(sums)) == 1.0

        # compute weighted mean error
        error[c] = norm(assemble(mean - references[c]))

        # compute effective sample size
        for j in range(N):
            eff[c] += norm(assemble(weights[j] ** 2))
        eff[c] = 1.0 / eff[c]

        c += 1

# plot error over time
plot.plot(np.linspace(1, ny, ny), error, 'r*-', linewidth=3)
plot.xlabel('assimilation step')
plot.ylabel('normalized error')
plot.show()

# plot effective sample size over time
plot.plot(np.linspace(1, ny, ny), eff, 'bo-', linewidth=3)
plot.xlabel('assimilation step')
plot.ylabel('effective sample size')
plot.show()
