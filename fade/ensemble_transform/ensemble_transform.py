""" ensemble transform update with kernels looping over cells """

from __future__ import division

from __future__ import absolute_import

from firedrake import *

import numpy as np

from fade.emd.emd_kernel import *

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
    if len(weights) < 1:
        raise ValueError('weights cannot be indexed')

    n = len(ensemble)

    # function space and vector function space
    fs = ensemble[0].function_space()
    deg = fs.ufl_element().degree()
    fam = fs.ufl_element().family()
    vfs = VectorFunctionSpace(fs.mesh(), fam, deg, dim=n)

    # check that weights have same length
    assert len(weights) == n

    # check that weights add up to one
    with timed_stage("Checking weights are normalized"):
        c = Function(fs)
        for k in range(n):
            c.dat.data[:] += weights[k].dat.data[:]

        if np.max(np.abs(c.dat.data[:] - 1)) > 1e-3:
            raise ValueError('Weights dont add up to 1')

    # preallocate new ensemble and assign basis coeffs to new vector function
    with timed_stage("Preallocating functions"):
        ensemble_f = Function(vfs)
        new_ensemble_f = Function(vfs)
        if n == 1:
            ensemble_f.dat.data[:] = ensemble[0].dat.data[:]
        else:
            for i in range(n):
                ensemble_f.dat.data[:, i] = ensemble[i].dat.data[:]

    # define even weights
    with timed_stage("Preallocating functions"):
        weights2 = []
        f = Function(fs).assign(1.0 / n)
        for k in range(n):
            weights2.append(f)

    # ensemble transform implementation
    kernel_transform(ensemble_f, ensemble_f, weights, weights2,
                     new_ensemble_f, r_loc)

    # check that components have the same mean
    with timed_stage("Checking posterior mean consistency"):
        m = Function(fs)
        for k in range(n):
            m.dat.data[:] += np.multiply(ensemble[k].dat.data[:], weights[k].dat.data[:])

    # override ensemble
    if n == 1:
        ensemble[0].dat.data[:] = new_ensemble_f.dat.data[:]
    else:
        for i in range(n):
            ensemble[i].dat.data[:] = new_ensemble_f.dat.data[:, i]

    # reset weights
    for i in range(n):
        weights[i].assign(1.0 / n)

    # check that components have the same mean
    with timed_stage("Checking posterior mean consistency"):
        mn = Function(fs)
        for k in range(n):
            mn.dat.data[:] += np.multiply(ensemble[k].dat.data[:], weights[k].dat.data[:])

        assert np.max(np.abs(mn.dat.data[:] - m.dat.data[:])) < 1e-5

    return ensemble
