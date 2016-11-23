""" ensemble transform update """

from __future__ import division

from __future__ import absolute_import

from firedrake import *

import numpy as np

from firedrake_da.localisation import *
from firedrake_da.emd import *


def ensemble_transform_update(ensemble, weights, r_loc_func):

    """ Computes the ensemble transform update (with localisation) in the etpf of a weighted ensemble

        :arg ensemble: list of :class:`Function`s in the ensemble
        :type ensemble: tuple / list

        :arg weights: list of :class:`Function`s representing the importance weights
        :type weights: tuple / list

        :arg r_loc_func: radius of localisation function
        :type r_loc_func: int

    """

    if len(ensemble) < 1:
        raise ValueError('ensemble cannot be indexed')

    # function space
    fs = ensemble[0].function_space()

    n = len(ensemble)

    # check that weights have same length
    assert len(weights) == n

    # check that weights add up to one
    nc = len(ensemble[0].dat.data)
    for i in range(nc):

        c = 0
        for k in range(n):
            c += weights[k].dat.data[i]

        if np.abs(c - 1) > 1e-3:
            raise ValueError('Weights dont add up to 1')

    # design localisation functions (NB: make this in script at the start / class for it, rather than
    # doing it at each assimilation step!)
    C = []
    for i in range(nc):
        C.append(Localisation(fs, r_loc_func, i))

    # preallocate new ensemble
    new_ensemble = []
    for i in range(n):
        f = Function(fs)
        new_ensemble.append(f)

    # find particle and weights matrcies
    particles = np.zeros((nc, n))
    w = np.zeros((nc, n))
    for j in range(nc):
        for k in range(n):
            particles[j, k] = ensemble[k].dat.data[j]
            w[j, k] = weights[k].dat.data[j]

    # for each component carry out emd
    for j in range(nc):

        # design cost matrix, using localisation functions
        Cost = np.zeros((n, n))
        for i in range(nc):
            p = np.reshape(particles[i, :], ((1, n)))
            Cost += C[j].dat.data[i] * CostMatrix(p, p)

        # transform
        P = np.reshape(particles[j, :], ((1, n)))
        # if 1D use cheap algorithm
        if nc == 1:
            ens = np.reshape(onedtransform(w[j, :],
                             np.ones(n) * (1.0 / n), particles[j, :]), ((1, n)))
        else:
            ens = transform(P, P, w[j, :], np.ones(n) * (1.0 / n), Cost)

        # into new ensemble
        for k in range(n):
            new_ensemble[k].dat.data[j] = ens[0, k]

    # check that components have the same mean
    for j in range(nc):
        mn = 0
        m = 0
        for k in range(n):
            mn += new_ensemble[k].dat.data[j] * (1.0 / n)
            m += ensemble[k].dat.data[j] * weights[k].dat.data[j]
        assert np.abs(mn - m) < 1e-5

    return new_ensemble
