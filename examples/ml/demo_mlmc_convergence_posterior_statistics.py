""" Convergence demo of multilevel monte carlo estimates of posterior statistics using a hierarchy of transformed / coupled ensembles """

from __future__ import division

from firedrake import *
from firedrake_mlmc import *
from fade import *

import matplotlib.pyplot as plot

import numpy as np


# define the mesh hierarchy
L = 2
mesh_hierarchy = MeshHierarchy(UnitSquareMesh(1, 1), L)

# define each level means
means = np.ones(L + 1)
means[0:L - 1] = 2 ** (-np.linspace(0, L - 2, L - 1))
means[-1] = 0

# the coordinates of observation (only one cell)
coords = tuple([np.array([0.5])])
obs = tuple([0.1])

# denote the true mean of finest posterior in the single cell
TrueMean = 0.7

# observation variance
sigma = 2.0

# define the function space hierarchy
fs_hierarchy = tuple([FunctionSpace(m, 'DG', 0) for m in mesh_hierarchy])

# observation operator hierarchy
oo_hierarchy = tuple([Observations(fs) for fs in fs_hierarchy])


# define the function to calculate the multilevel monte carlo estimate of
# posterior mean
def mlmc_estimate(N0, fs_hierarchy, means, oo_hierarchy, coords, obs, sigma):

    eh = EnsembleHierarchy(fs_hierarchy)
    L = len(fs_hierarchy) - 1
    ns = np.zeros(L)

    for i in range(L):
        n = int(N0 / (2 ** i))
        ns[i] = n
        coarse = []
        fine = []
        weights_c = []
        weights_f = []
        for k in range(n):
            xc = np.random.normal(0, 1, 1)[0] + means[i]
            xf = np.random.normal(0, 1, 1)[0] + means[i + 1]
            coarse.append(Function(fs_hierarchy[i]).assign(xc))
            fine.append(Function(fs_hierarchy[i + 1]).assign(xf))
            hc = Function(fs_hierarchy[i]).assign(1.0 / n)
            weights_c.append(hc)
            hf = Function(fs_hierarchy[i + 1]).assign(1.0 / n)
            weights_f.append(hf)

        # weight calculation
        r_loc = 0
        weights_c = weight_update(coarse, weights_c, oo_hierarchy[i], coords, obs, sigma, r_loc)
        weights_f = weight_update(fine, weights_f, oo_hierarchy[i + 1], coords, obs, sigma, r_loc)

        # transform / couple
        new_coarse, new_fine = seamless_coupling_update(coarse,
                                                        fine,
                                                        weights_c,
                                                        weights_f,
                                                        r_loc,
                                                        r_loc)

        # put into ensemble
        for k in range(n):
            s = State(new_coarse[k], new_fine[k])
            eh.AppendToEnsemble(s)

    eh.UpdateStatistics()

    # find variance
    ind_mean = eh.MultilevelExpectation.ufl_domain().locate_cell(coords[0])
    var = 0
    for i in range(L):
        ind_var = eh.Variance[i].ufl_domain().locate_cell(coords[0])
        var += eh.Variance[i].dat.data[ind_var] / ns[i]
    mean = eh.MultilevelExpectation.dat.data[ind_mean]

    return mean, var


# run the convergence loops with increasing n

s = 5
niter = 5

N0s = (4 * (2 ** np.linspace(0, s - 1, s))).astype(int)

rmse_func = np.zeros(s)
var_func = np.zeros(s)

for i in range(s):

    temp_rmse_func = np.zeros(niter)
    temp_var_func = np.zeros(niter)

    for j in range(niter):

        m_func, v_func = mlmc_estimate(N0s[i], fs_hierarchy, means,
                                       oo_hierarchy, coords, obs, sigma)

        temp_rmse_func[j] = np.square(m_func)

        temp_var_func[j] = v_func

    rmse_func[i] = np.sqrt(np.mean(temp_rmse_func))

    var_func[i] = np.mean(temp_var_func)

    print 'done sample size: ', N0s[i]

# plot results

plot.loglog(N0s, rmse_func, 'r*-')
plot.loglog(N0s, var_func, 'r*--')

plot.loglog(N0s, 1e1 * N0s.astype(float) ** (-1), 'k--')
plot.loglog(N0s, 1e1 * N0s.astype(float) ** (-0.5), 'k-')

plot.legend(['rmse (function)', 'variance (function)', 'linear decay',
             'sqrt decay'])

plot.show()
