""" covariance of ensemble of Firedrake functions using a vector function """

from __future__ import division

from __future__ import absolute_import

from firedrake import *

import numpy as np

from pyop2.profiling import timed_stage


def covariance(ensemble_f):

    """ finds the covariance vector function, for basis coeffs of a
        certain :class:`FunctionSpace`,
        from an ensemble given by a vector :class:`Function`.

        :arg ensemble: vector :class:`Function` representing ensemble (length n)
        :type ensemble: :class:`Function`

    """

    mesh = ensemble_f.function_space().mesh()

    # original space
    fs = ensemble_f.function_space()
    deg = fs.ufl_element().degree()
    fam = fs.ufl_element().family()

    nc = np.shape(ensemble_f.dat.data)[0]
    if len(np.shape(ensemble_f.dat.data)) > 1:
        n = np.shape(ensemble_f.dat.data)[1]
    else:
        n = 1

    vfsm = VectorFunctionSpace(mesh, fam, deg, dim=nc)

    with timed_stage("Preallocating functions"):
        cov = Function(vfsm)

    # generate covariance
    cov.dat.data[:] = np.cov(np.reshape(ensemble_f.dat.data[:],
                                        ((nc, n))))

    return cov
