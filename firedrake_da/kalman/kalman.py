""" kernel based kalman update calculation for an ensemble and observations - NB: Independent observation error! """

from __future__ import division

from __future__ import absolute_import

from firedrake import *

import numpy as np

from firedrake_da.kalman.cov import *
from firedrake_da.observations import *
from firedrake_da.kalman.kalman_kernel import *
from firedrake_da.utils import *
from firedrake_da.localisation import *

import scipy.sparse as scp

from pyop2.profiling import timed_stage


def kalman_update(ensemble, observation_operator, observation_coords, observations,
                  sigma, r_loc=0):

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

        :arg r_loc: Radius of covariance localisation
        :type r_loc: int

    """

    if len(ensemble) < 1:
        raise ValueError('ensemble cannot be indexed')
    mesh = ensemble[0].function_space().mesh()

    # original space
    fs = ensemble[0].function_space()

    n = len(ensemble)
    nc = len(ensemble[0].dat.data)

    # make into vector functions and preallocate covariance
    deg = fs.ufl_element().degree()
    fam = fs.ufl_element().family()

    vfsn = VectorFunctionSpace(mesh, fam, deg, dim=n)
    tfs = TensorFunctionSpace(mesh, fam, deg, (nc, n))

    ensemble_f = Function(vfsn)
    if n == 1:
        ensemble_f.dat.data[:] = ensemble[0].dat.data[:]
    else:
        for i in range(n):
            ensemble_f.dat.data[:, i] = ensemble[i].dat.data[:]

    # covariance
    cov = covariance(ensemble_f)

    # generate localisation
    loc = CovarianceLocalisation(cov.function_space(), r_loc)

    # compute hadamard product
    with timed_stage("Computing Hadamard Product of localisation and covariance functions"):
        cov_loc = HadamardProduct(cov, loc)

    # generate inverse plus observation error
    inv_cov_plus_R = Function(cov_loc.function_space())
    R = Function(cov_loc.function_space())

    # check that covariance is of correct shape
    assert np.shape(R.dat.data)[0] == nc

    R.dat.data[:] = np.diag(np.ones(nc) * sigma)
    inv_cov_plus_R.dat.data[:] = np.linalg.inv(np.reshape(cov_loc.dat.data[:] + R.dat.data[:],
                                                          ((nc, nc))))

    # find kalman gain
    kalman_gain = Function(cov_loc.function_space())
    kalman_gain.dat.data[:] = np.dot(np.reshape(cov_loc.dat.data[:],
                                                ((nc, nc))),
                                     np.reshape(inv_cov_plus_R.dat.data[:],
                                                ((nc, nc))))

    # difference in the observation space
    p = 1
    with timed_stage("Initial observation instance"):
        observation_operator.update_observation_operator(observation_coords, observations)
    with timed_stage("Preallocating functions"):
        diff = Function(vfsn)
    with timed_stage("Calculating observation differences"):
        if n == 1:
            f = observation_operator.difference(ensemble[0], p)
            diff.dat.data[:] = f.dat.data[:]
        else:
            for i in range(n):
                f = observation_operator.difference(ensemble[i], p)
                diff.dat.data[:, i] = f.dat.data[:]

    # preallocate new ensemble function
    with timed_stage("Preallocating functions"):
        new_ensemble_f = Function(vfsn)
        new_ensemble = []
        for i in range(n):
            f = Function(fs)
            new_ensemble.append(f)

    # now carry out kernel on multiplying kalman gain to differences
    with timed_stage("Preallocating functions"):
        increments = Function(vfsn)
        tensor_increments = Function(tfs)

    # take the transpose of kalman gain matrix
    kalman_gain.dat.data[:] = np.matrix.transpose(np.reshape(kalman_gain.dat.data[:],
                                                             ((nc, nc))))

    # generate kernel and dictionary
    with timed_stage("Carrying out kalman update"):
        k_kernel = kalman_kernel_generation(n, nc)
        Dict = {}
        Dict.update({"product_tensor": (tensor_increments, WRITE)})
        Dict.update({"input_matrix": (kalman_gain, READ)})
        Dict.update({"input_vector": (diff, READ)})
        par_loop(k_kernel.kalman_kernel, dx, Dict)

        increments.dat.data[:] = np.sum(tensor_increments.dat.data, axis=0)

    # add functions to find new ensemble
    with timed_stage("Carrying out kalman update"):
        new_ensemble_f = assemble(ensemble_f + increments)

    # put back into individual functions
    if n == 1:
        new_ensemble[0].dat.data[:] = new_ensemble_f.dat.data[:]
    else:
        for i in range(n):
            new_ensemble[i].dat.data[:] = new_ensemble_f.dat.data[:, i]

    return new_ensemble
