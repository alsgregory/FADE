""" demo of assimilating via ensemble transform into quasi-geostrophic model with random ic
and random forcing"""

from __future__ import division

from __future__ import absolute_import

from firedrake_da import *

# requires the use of quasi_geostrophic_model package
from quasi_geostrophic_model import *

import numpy as np


# create the mesh hierarchy (needed for coarsening localisation)
mesh = RectangleMesh(30, 6, 5, 1)
mesh_hierarchy = MeshHierarchy(mesh, 3)

# define function spaces
dg_fs = FunctionSpace(mesh_hierarchy[3], 'DG', 1)
cg_fs = FunctionSpace(mesh_hierarchy[3], 'CG', 1)


# define initial condition ufl expression
def ic(mesh):
    x = SpatialCoordinate(mesh)
    xp = np.random.uniform(0.1, 0.25)
    ufl_expression = (conditional(x[1] > (0.5 + xp) + (0.25 * sin(4 * pi * (x[0]))), 1.0, 0.0) +
                      conditional(x[1] < (0.5 - xp) + (0.25 * sin(4 * pi * (x[0]))), -1.0, 0.0))
    return ufl_expression


def get_observations(dg_fs, cg_fs, dt, T, var, R, ny):

    mesh = dg_fs.mesh()

    # set-up observations array
    coords = []
    observations = []
    refFile = File("reference_q.pvd")
    QG = quasi_geostrophic(dg_fs, cg_fs, var)
    QG.initial_condition(ic(mesh))
    refFile.write(QG.q_)

    nt = int(T / dt)
    for i in range(nt):
        QG.timestepper((i + 1) * dt)

        # get observations
        coords.append([])
        observations.append([])
        for j in range(ny):
            coords[i].append(np.array([np.random.uniform(0, 5, 1)[0],
                                       np.random.uniform(0, 1, 1)[0]]))
            observations[i].append(QG.psi_.at(coords[i][j]) +
                                   np.random.normal(0, np.sqrt(R), 1)[0])
        refFile.write(QG.q_)

    return coords, observations


def quasi_geostrophic_ensemble_transform(dg_fs, cg_fs, N,
                                         dt, T, var, R,
                                         observation_operator, coords, observations):

    mesh = dg_fs.mesh()

    # radius of localisation
    r_loc = 3

    # mean file
    meanqFile = File("et_mean_q.pvd")
    M = Function(dg_fs)

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
        object_ensemble[i].initial_condition(ic(mesh))
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
        weights = weight_update(psi_dg_ensemble, weights,
                                observation_operator, coords[k], observations[k], R, r_loc)

        # effective sample size
        eff = 0
        for i in range(N):
            eff += (weights[i].at(np.array([2.5, 0.75])) ** 2)

        print "effective sample size at time ", (k + 1) * dt, " is: ", 1.0 / eff

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


# variance of random forcing
var = 2.0

# observation operator
observation_operator = Observations(dg_fs)

# assimilation parameters (every 1.0 time we have an assimilation step)
dt = 1.0
T = 50.0
ny = 100
N = 20
R = 2.5e-2

# run simulation
print "finding observations..."
coords, observations = get_observations(dg_fs, cg_fs, dt, T, var, R, ny)

print "simulating ensemble transform mean of potential vorticity..."
quasi_geostrophic_ensemble_transform(dg_fs, cg_fs, N,
                                     dt, T, var, R,
                                     observation_operator, coords, observations)
