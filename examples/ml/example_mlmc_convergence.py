""" Demo showing the convergence (as eps -> 0) of the multilevel Monte Carlo (computed by firedrake-mlmc)
mean of posterior distribution.
Coarsest functions are DG0 functions on an interval mesh with a single cell.
Each function takes a normally distributed scalar value, with given mean for each level. """

from __future__ import division

from firedrake import *
from firedrake_mlmc import *
from fade import *

import matplotlib.pyplot as plot

import numpy as np


# define the mesh hierarchy
L = 5
mesh_hierarchy = MeshHierarchy(UnitIntervalMesh(1), L)

# define means of each level's prior normal distribution
means = (2 ** (-np.linspace(0, L, L + 1))) + 1

# the coordinates of observation (only one cell)
coords = tuple([np.array([0.5])])
obs = tuple([0.1])

# denote the true mean of finest posterior in the single cell
TrueMean = 0.7

# measurement error variance
R = 2.0

# define the function space hierarchy
fs_hierarchy = tuple([FunctionSpace(m, 'DG', 0) for m in mesh_hierarchy])

# observation operator hierarchy
oo_hierarchy = tuple([Observations(fs, R) for fs in fs_hierarchy])


# define the function to calculate the multilevel monte carlo estimate of
# posterior mean
def mlmc_estimate(eps, fs_hierarchy, means, oo_hierarchy, coords, obs):

    eh = EnsembleHierarchy(fs_hierarchy)
    L = int(np.ceil((-1 * np.log(eps)) / np.log(2)))
    if L > len(fs_hierarchy) - 1:
        print 'eps too low for function space hierarchy'
    N0 = int(eps ** (-2))
    ns = np.zeros(L)

    for i in range(L):
        n = np.max([2, int(N0 / (2 ** i))])
        ns[i] = n
        coarse = []
        fine = []
        weights_c = []
        weights_f = []
        for k in range(n):
            d = np.random.normal(0, 1)
            xc = d + means[i]
            xf = d + means[i + 1]
            coarse.append(Function(fs_hierarchy[i]).assign(xc))
            fine.append(Function(fs_hierarchy[i + 1]).assign(xf))
            hc = Function(fs_hierarchy[i]).assign(1.0 / n)
            weights_c.append(hc)
            hf = Function(fs_hierarchy[i + 1]).assign(1.0 / n)
            weights_f.append(hf)

        # weight calculation
        oo_hierarchy[i].update_observation_operator(coords, obs)
        oo_hierarchy[i + 1].update_observation_operator(coords, obs)
        weights_c = weight_update(coarse, weights_c, oo_hierarchy[i])
        weights_f = weight_update(fine, weights_f, oo_hierarchy[i + 1])

        # transform / couple
        new_coarse, new_fine = seamless_coupling_update(coarse,
                                                        fine,
                                                        weights_c,
                                                        weights_f)

        # put into ensemble
        for k in range(n):
            s = State(new_coarse[k], new_fine[k])
            eh.AppendToEnsemble(s)

    # update statistics using ensemble hierarchy
    eh.UpdateStatistics()

    # find rmse at coordinate of mesh in which observation is taken in
    index = eh.MultilevelExpectation.ufl_domain().locate_cell(coords[0])
    mean = eh.MultilevelExpectation.dat.data[index]

    return mean, ns, L


# define epsilon values
epsl = np.array([8e-1, 4e-1, 2e-1, 1e-1])

# define number of epsilon values and number of iterations for each posterior estimate
num_eps = len(epsl)
niter = 2

rmse_func = np.zeros(num_eps)

# iterate over epsilon values
for i in range(num_eps):

    temp_rmse_func = np.zeros(niter)

    for j in range(niter):

        m_func, ns, L = mlmc_estimate(epsl[i], fs_hierarchy, means,
                                      oo_hierarchy, coords, obs)

        temp_rmse_func[j] = np.square(m_func - TrueMean)

    rmse_func[i] = np.sqrt(np.mean(temp_rmse_func))

    print 'completed estimate with L =', L, ' and Nl =', ns

# plot results
plot.loglog(epsl, rmse_func, 'r*-')
plot.loglog(epsl, 1e1 * epsl.astype(float) ** (1), 'k--')
plot.legend(['rmse (function)', 'linear decay',
             'sqrt decay'])
plot.xlabel('epsilon')
plot.ylabel('RMSE')
plot.show()
