""" kalman update calculation for an ensemble and observations - NB: Independent observation error! """

from __future__ import division

from __future__ import absolute_import

from firedrake import *

import numpy as np

from firedrake_da.kalman.cov import *
from firedrake_da.observations import *


def Kalman_update(ensemble, observation_operator, observation_coords, observations,
                  sigma, fs_to_project_to="default"):

    """

        :arg ensemble: list of :class:`Function`s in the ensemble
        :type ensemble: tuple / list

        :arg observation_operator: the :class:`Observations` for the assimilation problem
        :type observation_operator: :class:`Observations`

        :arg observation_coords: tuple / list defining the coords of observations
        :type observation_coords: tuple / list

        :arg observations: tuple / list of observation state values
        :type observations: tuple / list

        :arg sigma: variance of independent observation error
        :type sigma: float

        :arg fs_to_project_to: optional argument allowing user to calculate kalman gain in
                               another :class:`FunctionSpace` then simply DG0
        :type fs_to_project_to: :class:`FunctionSpace`

    """

    if len(ensemble) < 1:
        raise ValueError('ensemble cannot be indexed')
    mesh = ensemble[0].function_space().mesh()

    if fs_to_project_to is "default":
        fs_to_project_to = FunctionSpace(mesh, 'DG', 0)

    # original space
    original_space = ensemble[0].function_space()

    n = len(ensemble)

    """ we make the kalman gain in the 'observation space', given a simple fs_to_project_to
    :class:`FunctionSpace`, and then build the new ensemble in this space, projecting back to
    the original space """

    # generate covariance
    cov, in_ensemble_dat = Covariance(ensemble, fs_to_project_to)

    # difference in the observation space
    p = 1
    observation_operator.update_observation_operator(observation_coords, observations)
    D = []
    for i in range(n):
        f = Function(fs_to_project_to)
        # have to project that difference functions back into the fs_to_project_to
        f.project(observation_operator.difference(ensemble[i], p))
        D.append(f)

    # now put this difference in matrix of data
    d = np.zeros(np.shape(in_ensemble_dat))
    for i in range(n):
        d[:, i] = D[i].dat.data

    """ kalman gain and update """
    # make R matrix on observation space - independent observation error!
    r = np.identity(len(d[:, 0])) * sigma

    # kalman gain - NB: because all components of K are in observation space, operator H becomes id
    K = np.dot(cov, np.linalg.inv(cov + r))
    X = in_ensemble_dat + np.dot(K, d)

    # formula for kalman gain
    new_ensemble = []
    for i in range(n):
        f = Function(fs_to_project_to)
        f.dat.data[:] = X[:, i]
        new_ensemble.append(f)

    """ project new ensemble back to original space """
    transformed_ensemble = []

    for i in range(n):
        out_func = Function(original_space).project(new_ensemble[i])
        transformed_ensemble.append(out_func)

    return transformed_ensemble
