""" ensemble transform update """

from __future__ import division

from __future__ import absolute_import

from firedrake import *

import numpy as np

from firedrake_da.localisation import *
from firedrake_da.localisation_functions import *
from firedrake_da.emd import *

from pyop2.profiling import timed_stage


def ensemble_transform_update(ensemble, weights, localisation_functions):

    """ Computes the ensemble transform update (with localisation) in the etpf of a weighted ensemble

        :arg ensemble: list of :class:`Function`s in the ensemble
        :type ensemble: tuple / list

        :arg weights: list of :class:`Function`s representing the importance weights
        :type weights: tuple / list

        :arg r_loc_func: radius of localisation function
        :type r_loc_func: int

        :arg localisation_functions: The :class:`LocalisationFunctions` for the given function space
        :type localisation_functions: :class:`LocalisationFunctions`

    """

    if len(ensemble) < 1:
        raise ValueError('ensemble cannot be indexed')

    # function space
    fs = ensemble[0].function_space()

    n = len(ensemble)

    # check that weights have same length
    assert len(weights) == n

    # check that weights add up to one
    with timed_stage("Checking weights are normalized"):
        nc = len(ensemble[0].dat.data)
        c = np.zeros(nc)
        for k in range(n):
            c += weights[k].dat.data[:]

        if np.max(np.abs(c - 1)) > 1e-3:
            raise ValueError('Weights dont add up to 1')

    # check if the localisation functions are of that type
    if not isinstance(localisation_functions, LocalisationFunctions):
        raise ValueError('localisation_functions needs to be the object LocalisationFunctions. ' +
                         'See help(LocalisationFunctions) for details')

    # check that the function spaces of :class:`LocalisationFunctions` are the same
    assert localisation_functions.function_space == fs

    # preallocate new ensemble
    with timed_stage("Preallocating functions"):
        new_ensemble = []
        for i in range(n):
            f = Function(fs)
            new_ensemble.append(f)

    # find particle and weights matrcies
    with timed_stage("Assigning basis coefficient arrays"):
        particles = np.zeros((nc, n))
        w = np.zeros((nc, n))
        for k in range(n):
            particles[:, k] = ensemble[k].dat.data[:]
            w[:, k] = weights[k].dat.data[:]

    # for each component carry out emd
    with timed_stage("Ensemble transform"):
        for j in range(nc):

            # design cost matrix, using localisation functions
            Cost = np.zeros((n, n))
            for i in range(nc):
                p = np.reshape(particles[i, :], ((1, n)))
                Cost += localisation_functions[j].dat.data[i] * CostMatrix(p, p)

            # transform
            P = np.reshape(particles[j, :], ((1, n)))
            # if 1D or r_loc_func = 0 use cheap algorithm
            if nc == 1 or localisation_functions.r_loc_func == 0:
                a = np.argsort(particles[j, :])
                ens = np.reshape(onedtransform(w[j, a.astype(int)],
                                 np.ones(n) * (1.0 / n), particles[j, a.astype(int)]), ((1, n)))
            else:
                ens = transform(P, P, w[j, :], np.ones(n) * (1.0 / n), Cost)

            # into new ensemble
            for k in range(n):
                new_ensemble[k].dat.data[j] = ens[0, k]

    # check that components have the same mean
    with timed_stage("Checking posterior mean consistency"):
        mn = np.zeros(nc)
        m = np.zeros(nc)
        for k in range(n):
            mn += new_ensemble[k].dat.data[:] * (1.0 / n)
            m += np.multiply(ensemble[k].dat.data[:], weights[k].dat.data[:])
        assert np.max(np.abs(mn - m)) < 1e-5

    return new_ensemble
