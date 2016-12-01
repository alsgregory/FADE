""" weight update calculation for an ensemble and observations - NB: Independent gaussian observation error! """

from __future__ import division

from __future__ import absolute_import

from firedrake import *

from firedrake.mg.utils import get_level

import numpy as np

from firedrake_da.observations import *
from firedrake_da.localisation import *

from pyop2.profiling import timed_stage


def weight_update(ensemble, observation_operator, observation_coords,
                  observations, sigma, r_loc):

    """ Calculates the importance weights of ensemble members around assumed gaussian observations

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

        :arg r_loc: radius of coarsening localisation for importance weight update
        :type r_loc: int

    """

    if len(ensemble) < 1:
        raise ValueError('ensemble cannot be indexed')
    mesh = ensemble[0].function_space().mesh()

    # check that part of a hierarchy - so that one can coarsen localise
    hierarchy, lvl = get_level(mesh)
    if lvl is None:
        raise ValueError('mesh for ensemble members needs to be part of hierarchy for coarsening loc')

    # function space
    fs = ensemble[0].function_space()

    n = len(ensemble)

    """ we make the importance weights in the observation space """

    # difference in the observation space
    p = 2
    with timed_stage("Initial observation instance"):
        observation_operator.update_observation_operator(observation_coords,
                                                         observations)
    D = []
    for i in range(n):
        with timed_stage("Preallocating functions"):
            f = Function(fs)
        with timed_stage("Calculating observation squared differences"):
            # have to project that difference functions back into the fs_to_project_to
            f.assign(observation_operator.difference(ensemble[i], p))
        D.append(f)

    # now conduct coarsening localisation and make weights
    with timed_stage("Coarsening localisation"):
        WLoc = []
        for i in range(n):
            WLoc.append(CoarseningLocalisation(D[i], r_loc))

    # find number of basis coefficients and preallocate weight functions
    nc = len(WLoc[0].dat.data)
    with timed_stage("Preallocating functions"):
        W = []
        for i in range(n):
            f = Function(fs)
            W.append(f)

    # gaussian likelihood
    with timed_stage("Likelihood calculation"):
        for j in range(n):
            WLoc[j].dat.data[:] = (1 / np.sqrt(2 * np.pi * sigma)) * np.exp(-(1 / (2 * sigma)) *
                                                                            WLoc[j].dat.data[:])

    # normalize and check weights
    t = np.zeros(nc)
    c = np.zeros(nc)
    for j in range(n):
        t += WLoc[j].dat.data[:]

    with timed_stage("Checking weights are normalized"):
        for k in range(n):
            W[k].dat.data[:] = np.divide(WLoc[k].dat.data[:], t)
            c += W[k].dat.data[:]

        if np.max(np.abs(c - 1)) > 1e-3:
            raise ValueError('Weights dont add up to 1')

    return W
