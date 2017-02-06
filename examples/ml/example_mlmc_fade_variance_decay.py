""" Demo showing the decay in sample variance between fine and coarse ensembles of functions as l increases.
Functions are on interval meshes with values given by interpolating a randomly shifted sin curve. """

from __future__ import division

from __future__ import absolute_import

from fade import *
from fade.ml import *

import numpy as np

import matplotlib.pyplot as plot


# create the mesh hierarchy
mesh = UnitIntervalMesh(1)
L = 5
mesh_hierarchy = MeshHierarchy(mesh, L)

# generate function space hierarchy
fs_hierarchy = tuple([FunctionSpace(m, 'CG', 1) for m in mesh_hierarchy])

# the coordinates of observation and generate the observation
ny = 1
R = 0.4
coords = []
obs = []
x = SpatialCoordinate(mesh_hierarchy[-1])
ref = Function(fs_hierarchy[-1]).interpolate(sin((x[0] + np.random.normal(0, 1)) * 2 * pi))
ref_perturbed = Function(fs_hierarchy[-1])
ref_perturbed.dat.data[:] = np.random.normal(0, 1, len(ref.dat.data)) * np.sqrt(R)
for i in range(ny):
    coords.append(np.random.uniform(0, 1, 1))
    obs.append(ref_perturbed.at(coords[i][0]))

coords = tuple(coords)
obs = tuple(obs)

# generate observation operator hierarchy
ooh = tuple([Observations(fs, R) for fs in fs_hierarchy])

# sample size
n = 50

# preallocate variance decay array
var = np.zeros(len(mesh_hierarchy) - 1)


# define the seamless coupling update step
def seamless_coupling_step(fs_hierarchy, lvlc, lvlf, n, ooh,
                           coords, obs):

    Vc = fs_hierarchy[lvlc]
    Vf = fs_hierarchy[lvlf]

    # generate ensemble of functions and weights
    ensemble_c = []
    ensemble_f = []
    weights_c = []
    weights_f = []
    for i in range(n):
        xc = SpatialCoordinate(Vc.mesh())
        d = np.random.normal(0, 0.1, 1)[0]
        f = Function(Vc).interpolate(sin(2 * pi * (d + xc[0])))
        ensemble_c.append(f)
        xf = SpatialCoordinate(Vf.mesh())
        g = Function(Vf).interpolate(sin(2 * pi * (d + xf[0])))
        ensemble_f.append(g)
        hc = Function(Vc).assign(1.0 / n)
        weights_c.append(hc)
        hf = Function(Vf).assign(1.0 / n)
        weights_f.append(hf)

    # generate coarse and fine posterior - increase radius of coarsening localisation with level!
    r_loc_c = lvlc
    r_loc_f = lvlf
    ooh[lvlc].update_observation_operator(coords, obs)
    ooh[lvlf].update_observation_operator(coords, obs)
    weights_c = weight_update(ensemble_c, weights_c, ooh[lvlc], r_loc_c)
    weights_f = weight_update(ensemble_f, weights_f, ooh[lvlf], r_loc_f)
    Xc, Xf = seamless_coupling_update(ensemble_c, ensemble_f, weights_c,
                                      weights_f, r_loc_c, r_loc_f)

    # generate sample variance
    Msq = Function(fs_hierarchy[-1])
    Mm = Function(fs_hierarchy[-1])
    comp_f = Function(fs_hierarchy[-1])
    comp_c = Function(fs_hierarchy[-1])
    for i in range(n):
        if lvlf == len(fs_hierarchy) - 1:
            comp_f.assign(Xf[i])
            prolong(Xc[i], comp_c)
        else:
            prolong(Xf[i], comp_f)
            prolong(Xc[i], comp_c)
        Msq += assemble(((comp_f - comp_c) ** 2) * (1.0 / n))
        Mm += assemble((comp_f - comp_c) * (1.0 / n))
    f = Function(fs_hierarchy[-1]).assign(Msq - (Mm ** 2))
    return f


# define number of iterations for each posterior estimate
niter = 3

# iterate over level of resolution
for i in range(len(mesh_hierarchy) - 1):

    temp_var = np.zeros(niter)

    for j in range(niter):

        k = seamless_coupling_step(fs_hierarchy, i, i + 1, int(n), ooh, coords, obs)

        temp_var[j] = assemble(k * dx)

    var[i] = np.mean(temp_var)

    print 'completed lvlc = ', i, ' level iteration'

# find number of degrees of freedom for each level of resolution
degs = 4 * (2 ** np.linspace(0, len(mesh_hierarchy) - 2, len(mesh_hierarchy) - 1))

# plot results
plot.loglog(degs, var, 'r*-')
plot.loglog(degs, 8e-2 * degs ** (- 1.0 / 2.0), 'k--')
plot.legend(['variance decay', 'sqrt decay'])
plot.xlabel('sample size')
plot.ylabel('rmse')
plot.show()
