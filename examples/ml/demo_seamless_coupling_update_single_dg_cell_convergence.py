""" demo for the coarse and fine posterior convergence of a single seamless coupling update step on a
single cell """

from __future__ import division

from __future__ import absolute_import

from firedrake_da import *
from firedrake_da.ml import *

import numpy as np

import matplotlib.pyplot as plot


# create the mesh hierarchy
mesh = UnitIntervalMesh(1)
mesh_hierarchy = MeshHierarchy(mesh, 1)

Vc = FunctionSpace(mesh_hierarchy[0], 'DG', 0)
Vf = FunctionSpace(mesh_hierarchy[1], 'DG', 0)

# the coordinates of observation (only one cell)
coords = tuple([np.array([0.5])])
obs = tuple([0.1])

observation_operator_c = Observations(Vc)
observation_operator_f = Observations(Vf)

# denote the true mean of both the coarse and fine posterior in the single cell
TrueMean = 0.7

# range of sample sizes
ns = 4 * (2 ** np.linspace(0, 5, 6))

# preallocate rmse array
rmse_c = np.zeros(len(ns))
rmse_f = np.zeros(len(ns))


# define the seamless coupling update step
def seamless_coupling_step(Vc, Vf, n, oo_c, oo_f, coords, obs, sigma):

    # generate ensemble
    ensemble_c = []
    ensemble_f = []
    for i in range(n):
        f = Function(Vc).assign(np.random.normal(1, 1, 1)[0])
        ensemble_c.append(f)
        g = Function(Vf).assign(np.random.normal(1, 1, 1)[0])
        ensemble_f.append(g)

    # generate coarse and fine posterior
    r_loc = 0
    weights_c = weight_update(ensemble_c, oo_c, coords, obs, sigma, r_loc)
    weights_f = weight_update(ensemble_f, oo_f, coords, obs, sigma, r_loc)
    Xc, Xf = seamless_coupling_update(ensemble_c, ensemble_f, weights_c, weights_f, r_loc, r_loc)

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


sigma = 2.0

niter = 20

for i in range(len(ns)):

    temp_mse_c = np.zeros(niter)
    temp_mse_f = np.zeros(niter)

    for j in range(niter):

        kc, kf = seamless_coupling_step(Vc, Vf, int(ns[i]), observation_operator_c,
                                        observation_operator_f, coords, obs, sigma)

        temp_mse_c[j] = np.square(kc - TrueMean)
        temp_mse_f[j] = np.square(kf - TrueMean)

    rmse_c[i] = np.sqrt(np.mean(temp_mse_c))
    rmse_f[i] = np.sqrt(np.mean(temp_mse_f))

    print 'completed n = ', ns[i], ' sample size iteration'

plot.loglog(ns, rmse_c, 'r*-')
plot.loglog(ns, rmse_f, 'bo-')
plot.loglog(ns, 8e-1 * ns ** (- 1.0 / 2.0), 'k--')
plot.legend(['coarse posterior rmse', 'fine posterior rmse', 'sqrt decay'])
plot.xlabel('sample size')
plot.ylabel('rmse')
plot.show()
