""" demo of assimilating via ensemble transform into quasi-geostrophic model with random ic
and random forcing"""

from __future__ import division

from __future__ import absolute_import

from fade import *

# requires the use of quasi_geostrophic_model package
from quasi_geostrophic_model import *

import matplotlib.pyplot as plot

import numpy as np


# generate a mesh from a mesh hierarchy
mesh_hierarchy = MeshHierarchy(RectangleMesh(10, 2, 5, 1), 3)
mesh = mesh_hierarchy[-1]

# define function spaces
dg_fs = FunctionSpace(mesh, 'DG', 1)
cg_fs = FunctionSpace(mesh, 'CG', 1)


def get_observations(dg_fs, cg_fs, dt, T, var, R, ny):

    mesh = dg_fs.mesh()

    x = SpatialCoordinate(mesh)
    ufl_expression = (conditional(x[1] > 0.5 + (0.25 * sin(4 * pi * (x[0]))), 1.0, 0.0) +
                      conditional(x[1] < 0.5 + (0.25 * sin(4 * pi * (x[0]))), -1.0, 0.0))

    # set-up observations array
    coords = []
    observations = []
    refFile = File("reference_q.pvd")
    QG = quasi_geostrophic(dg_fs, cg_fs, var)
    QG.initial_condition(ufl_expression, 1.0)
    refFile.write(QG.q_)

    refFunctions = []

    ref_psi = Function(dg_fs)
    ref_psiProjector = Projector(QG.psi_, ref_psi)

    nt = int(T / dt)
    for i in range(nt):
        QG.timestepper((i + 1) * dt)

        # get observations
        coords.append([])
        observations.append([])
        ref_psi.assign(0)
        ref_psiProjector.project()
        ref_psi.dat.data[:] += np.random.normal(0, np.sqrt(R), len(ref_psi.dat.data))
        for j in range(ny):
            coords[i].append(np.array([np.random.uniform(0, 5, 1)[0],
                                       np.random.uniform(0, 1, 1)[0]]))
            observations[i].append(ref_psi.at(coords[i][j]))
        refFile.write(QG.q_)

        # keep reference functions for error
        f = Function(dg_fs).assign(QG.q_)
        refFunctions.append(f)

    return coords, observations, refFunctions


def quasi_geostrophic_ensemble_transform(dg_fs, cg_fs, N,
                                         dt, T, var,
                                         observation_operator, coords, observations):

    mesh = dg_fs.mesh()

    x = SpatialCoordinate(mesh)
    ufl_expression = (conditional(x[1] > 0.5 + (0.25 * sin(4 * pi * (x[0]))), 1.0, 0.0) +
                      conditional(x[1] < 0.5 + (0.25 * sin(4 * pi * (x[0]))), -1.0, 0.0))

    # radius of localisation
    r_loc = 3

    # mean file
    meanqFile = File("et_mean_q.pvd")
    M = Function(dg_fs)
    meanFunctions = []

    # generate ensemble
    psi_ensemble = []
    q_ensemble = []
    psi_dg_ensemble = []

    # projector from observation space to ensemble space
    psi_to_dgProjectors = []

    object_ensemble = []
    weights = []
    for i in range(N):
        object_ensemble.append(quasi_geostrophic(dg_fs, cg_fs, var))
        object_ensemble[i].initial_condition(ufl_expression, 1.0)
        psi_ensemble.append(object_ensemble[i].psi_)
        q_ensemble.append(object_ensemble[i].q_)
        psi_dg_ensemble.append(Function(dg_fs))

        # setup projectors
        psi_to_dgProjectors.append(Projector(object_ensemble[i].psi_, psi_dg_ensemble[i]))

        f = Function(dg_fs).assign(1.0 / N)
        weights.append(f)

    # write mean to file
    M.assign(0)
    for i in range(N):
        M += object_ensemble[i].q_ * (1.0 / N)
    meanqFile.write(M)

    # begin timestepping
    nt = int(T / dt)
    for k in range(nt):
        for i in range(N):
            object_ensemble[i].timestepper((k + 1) * dt)
            psi_ensemble[i] = object_ensemble[i].psi_
            q_ensemble[i] = object_ensemble[i].q_

        # projecting psi to dg
        for i in range(N):
            psi_to_dgProjectors[i].project()

        # update weights and ensmeble transform
        observation_operator.update_observation_operator(coords[k], observations[k])
        weights = weight_update(psi_dg_ensemble, weights,
                                observation_operator, r_loc)

        X = ensemble_transform_update(q_ensemble, weights, r_loc)

        # rewrite ensemble
        for i in range(N):
            object_ensemble[i].q_.assign(X[i])
            q_ensemble[i].assign(object_ensemble[i].q_)

        # write mean to file
        M.assign(0)
        for i in range(N):
            M += q_ensemble[i] * (1.0 / N)
        meanqFile.write(M)

        # keep mean functions for error
        f = Function(dg_fs).assign(M)
        meanFunctions.append(f)

    return meanFunctions, q_ensemble


# variance of random forcing
var = 2.0

# assimilation parameters (every 1.0 time we have an assimilation step)
dt = 1.0
T = 10.0
ny = 150
N = 75
R = 5e-3

# observation operator
observation_operator = Observations(dg_fs, R)


""" Data Assimilation Simulation """

# run simulation
print "finding observations..."
coords, observations, refFunctions = get_observations(dg_fs, cg_fs, dt, T, var, R, ny)

print "simulating ensemble transform mean of potential vorticity..."
meanFunctions, q_ensemble = quasi_geostrophic_ensemble_transform(dg_fs, cg_fs, N,
                                                                 dt, T, var,
                                                                 observation_operator, coords, observations)


""" Root Mean Square Error """

# find error
error = np.zeros(len(meanFunctions))
for i in range(len(meanFunctions)):
    error[i] = norm(assemble(meanFunctions[i] - refFunctions[i]))

# plot error
obs_error = np.sqrt(R) * np.ones(len(meanFunctions))

ts = np.linspace(dt, T, int(T / dt))

plot.plot(ts, error, 'r*-')
plot.plot(ts, obs_error, 'bo-')
plot.xlabel('time')
plot.ylabel('normalized error')
plot.legend(['etpf mean', 'observations'])
plot.show()


""" Cumulative Distribution Function """

# find empirical CDF at a single coordinate
coordinate = np.array([0.6, 0.5])

ensemble = []
for i in range(N):
    ensemble.append(q_ensemble[i].at(coordinate))

weights = np.linspace(1, N, N) / N

cumulative_ensemble = np.sort(ensemble)

plot.plot(cumulative_ensemble, weights, 'r-')
plot.plot(refFunctions[-1].at(coordinate) * np.ones(N), weights, 'k--')
plot.xlabel('CDF')
plot.ylabel('State at x=' + str(coordinate[0]) + ', y=' + str(coordinate[1]))
plot.legend(['CDF of Ensemble', 'Reference'])
plot.show()
