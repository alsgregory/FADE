""" ensemble transform update with kernels looping over cells """

from __future__ import division

from __future__ import absolute_import

from firedrake import *

import numpy as np

from firedrake_da.EMD.emd_kernel import *

from pyop2.profiling import timed_stage


def ensemble_transform_update(ensemble, weights, r_loc):

    """ Computes the (localised) ensemble transform update in the etpf of a weighted ensemble

        :arg ensemble: list of :class:`Function`s in the ensemble
        :type ensemble: tuple / list

        :arg weights: list of :class:`Function`s representing the importance weights
        :type weights: tuple / list

        :arg r_loc: Radius of coarsening localisation for the cost functions
        :type r_loc: int

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

    # preallocate new ensemble
    with timed_stage("Preallocating functions"):
        new_ensemble = []
        for i in range(n):
            f = Function(fs)
            new_ensemble.append(f)

    # define even weights
    weights2 = []
    for k in range(n):
        f = Function(fs).assign(1.0 / n)
        weights2.append(f)

    # ensemble transform implementation
    kernel_transform(ensemble, ensemble, weights, weights2,
                     new_ensemble, r_loc)

    # check that components have the same mean
    with timed_stage("Checking posterior mean consistency"):
        mn = np.zeros(nc)
        m = np.zeros(nc)
        for k in range(n):
            mn += new_ensemble[k].dat.data[:] * (1.0 / n)
            m += np.multiply(ensemble[k].dat.data[:], weights[k].dat.data[:])
        assert np.max(np.abs(mn - m)) < 1e-5

    return new_ensemble
