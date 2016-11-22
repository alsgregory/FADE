""" covariance of ensemble of Firedrake functions """

from __future__ import division

from __future__ import absolute_import

from firedrake import *

import numpy as np


def Covariance(ensemble, fs_to_project_to="default"):

    """ finds the covariance matrix, for basis coeffs of a certain projecting :class:`FunctionSpace`,
        from an ensemble of :class:`Function`s

        :arg ensemble: list of :class:`Function`s in the ensemble
        :type ensemble: tuple / list

        :arg fs_to_project_to: optional argument allowing user to calculate kalman gain in
                               another :class:`FunctionSpace` then simply DG0
        :type fs_to_project_to: :class:`FunctionSpace`

    """

    if len(ensemble) < 1:
        raise ValueError('ensemble cannot be indexed')
    mesh = ensemble[0].function_space().mesh()

    n = len(ensemble)

    if fs_to_project_to is "default":
        fs_to_project_to = FunctionSpace(mesh, 'DG', 0)

    # project to function space for covariance estimation
    in_ensemble = []
    for f in ensemble:
        in_func = Function(fs_to_project_to).project(f)
        in_ensemble.append(in_func)

    """" derive covariance using basis coeffs """

    dim = [s for s in np.shape(in_ensemble[0].dat.data)]
    dim.append(n)
    n_shape = tuple(dim)
    in_ensemble_dat = np.zeros(n_shape)
    for i in range(n):
        in_ensemble_dat[..., i] = in_ensemble[i].dat.data

    # functionality only exists for scalar functionals
    if len(n_shape) > 2:
        raise ValueError('only scalar functionals currently supported')

    est_cov = np.cov(in_ensemble_dat)

    return est_cov, in_ensemble_dat
