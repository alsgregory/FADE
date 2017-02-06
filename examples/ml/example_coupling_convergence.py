""" Demo showing the convergence of the coarse and fine posterior functions after a seamless coupling update.
Coarse functions are DG0 functions on interval meshes with a single cell.
Each function takes a normally distributed scalar value. """

from __future__ import division

from __future__ import absolute_import

from fade import *
from fade.ml import *

import numpy as np

import matplotlib.pyplot as plot


# create the mesh hierarchy
mesh = UnitIntervalMesh(1)
mesh_hierarchy = MeshHierarchy(mesh, 1)

# define coarse and fine function spaces
Vc = FunctionSpace(mesh_hierarchy[0], 'DG', 0)
Vf = FunctionSpace(mesh_hierarchy[1], 'DG', 0)

# the coordinates of observation (only one cell)
coords = tuple([np.array([0.5])])
obs = tuple([0.1])

# variance of measurement error
R = 2.0

# initialize coarse and fine observation operators
observation_operator_c = Observations(Vc, R)
observation_operator_f = Observations(Vf, R)

# denote the true mean of both the coarse and fine posterior in the single cell
TrueMean = 0.7

# range of sample sizes
ns = 4 * (2 ** np.linspace(0, 4, 5))

# preallocate rmse array
rmse_c = np.zeros(len(ns))
rmse_f = np.zeros(len(ns))


# define the seamless coupling update step
def seamless_coupling_step(Vc, Vf, n, oo_c, oo_f, coords, obs):

    # generate ensemble
    ensemble_c = []
    ensemble_f = []
    weights_c = []
    weights_f = []
    for i in range(n):
        f = Function(Vc).assign(np.random.normal(1, 1))
        ensemble_c.append(f)
        g = Function(Vf).assign(np.random.normal(1, 1))
        ensemble_f.append(g)
        hc = Function(Vc).assign(1.0 / n)
        weights_c.append(hc)
        hf = Function(Vf).assign(1.0 / n)
        weights_f.append(hf)

    # generate coarse and fine posterior
    oo_c.update_observation_operator(coords, obs)
    oo_f.update_observation_operator(coords, obs)
    weights_c = weight_update(ensemble_c, weights_c, oo_c)
    weights_f = weight_update(ensemble_f, weights_f, oo_f)
    Xc, Xf = seamless_coupling_update(ensemble_c, ensemble_f, weights_c, weights_f)

    # generate coarse / fine mean at the cell which contains coordinate of observation
    mesh_c = Vc.mesh()
    mesh_f = Vf.mesh()
    Mc = 0
    for i in range(n):
        Mc += (1 / float(n)) * Xc[i].dat.data[mesh_c.locate_cell(coords[0])]
    Mf = 0
    for i in range(n):
        Mf += (1 / float(n)) * Xf[i].dat.data[mesh_f.locate_cell(coords[0])]
    return Mc, Mf


# define number of iterations for each posterior estimate
niter = 5

# iterate over ensemble sizes
for i in range(len(ns)):

    temp_mse_c = np.zeros(niter)
    temp_mse_f = np.zeros(niter)

    for j in range(niter):

        kc, kf = seamless_coupling_step(Vc, Vf, int(ns[i]), observation_operator_c,
                                        observation_operator_f, coords, obs)

        temp_mse_c[j] = np.square(kc - TrueMean)
        temp_mse_f[j] = np.square(kf - TrueMean)

    rmse_c[i] = np.sqrt(np.mean(temp_mse_c))
    rmse_f[i] = np.sqrt(np.mean(temp_mse_f))

    print 'completed n = ', ns[i], ' sample size iteration'

# plot results
plot.loglog(ns, rmse_c, 'r*-')
plot.loglog(ns, rmse_f, 'bo-')
plot.loglog(ns, 8e-1 * ns ** (- 1.0 / 2.0), 'k--')
plot.legend(['coarse posterior rmse', 'fine posterior rmse', 'sqrt decay'])
plot.xlabel('sample size')
plot.ylabel('rmse')
plot.show()
