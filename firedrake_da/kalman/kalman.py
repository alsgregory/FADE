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
import scipy.sparse.linalg as scplinalg

from pyop2.profiling import timed_stage


def kalman_update(ensemble, observation_operator, observation_coords, observations,
                  sigma, r_loc=None):

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

    # generate localisation if not None
    if r_loc is None:
        cov_loc = cov

    else:
        loc = CovarianceLocalisation(cov.function_space(), r_loc)

        # compute hadamard product
        with timed_stage("Computing Hadamard Product of localisation and covariance functions"):
            cov_loc = HadamardProduct(cov, loc)

    # check that covariance is of correct shape
    assert np.shape(cov_loc.dat.data)[0] == nc

    # define R (measurement error) and the inverse of the R + cov in sparse form
    R = np.diag(np.ones(nc) * sigma)
    cov_loc_matrix = np.reshape(cov_loc.dat.data[:], ((nc, nc)))
    cov_loc_plus_R = ConstructSparseMatrix(cov_loc_matrix + R)
    inv_cov_plus_R = scplinalg.inv(cov_loc_plus_R)

    # find kalman gain
    cov_loc_matrix_sparse = ConstructSparseMatrix(cov_loc_matrix)
    kalman_gain = cov_loc_matrix_sparse.dot(inv_cov_plus_R)

    # take the transpose of this kalman_gain matrix
    kalman_gain_t = kalman_gain.transpose()

    # put basis coefficients into a kalman gain function from this sparse matrix
    K = Function(cov.function_space())
    # find all non zero elements and loop over them
    non_zeros = scp.find(kalman_gain_t)
    for i in range(len(non_zeros[2])):
        if nc == 1:
            K.dat.data[non_zeros[0][i]] = non_zeros[2][i]
        else:
            K.dat.data[non_zeros[0][i], non_zeros[1][i]] = non_zeros[2][i]

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

    # generate kernel and dictionary
    with timed_stage("Carrying out kalman update"):
        k_kernel = kalman_kernel_generation(n, nc)
        Dict = {}
        Dict.update({"product_tensor": (tensor_increments, WRITE)})
        Dict.update({"input_matrix": (K, READ)})
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
