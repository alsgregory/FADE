from __future__ import division

from __future__ import absolute_import

from firedrake import *

from firedrake_da import *
from firedrake_da.kalman import *

import numpy as np


def test_kalman_perfect_obs():

    mesh = UnitIntervalMesh(1)

    v = FunctionSpace(mesh, 'DG', 0)

    ensemble = []
    n = 10
    for i in range(n):
        f = Function(v).assign(1.0)
        ensemble.append(f)

    # observations
    obs = np.array([1.0])
    coords = np.array([np.array([0.5])])

    # new ensemble
    observation_operator = Observations(v)
    sigma = 1.0
    new_ensemble = kalman_update(ensemble, observation_operator, coords, obs, sigma)

    for i in range(n):
        assert np.max(np.abs(new_ensemble[i].dat.data[:] - 1.0)) < 1e-5


def test_kalman():

    # true solution
    X = np.array([[1, 0], [1.5, 0.75]])
    Cov = np.cov(X)
    R = np.identity(2)
    D = np.array([[0.5], [1]])
    Diff = D - X
    K = np.dot(Cov, np.linalg.inv(Cov + R))
    XNew = X + np.dot(K, Diff)

    # programmed solution
    mesh = UnitIntervalMesh(2)

    fs = FunctionSpace(mesh, 'DG', 0)

    coords = np.array([np.array([0.25]), np.array([0.75])])
    cell_1 = mesh.locate_cell(coords[0])
    cell_2 = mesh.locate_cell(coords[1])

    ensemble = []
    f = Function(fs)
    f.dat.data[cell_1] = 1
    f.dat.data[cell_2] = 1.5
    ensemble.append(f)
    f = Function(fs)
    f.dat.data[cell_1] = 0
    f.dat.data[cell_2] = 0.75
    ensemble.append(f)

    obs = np.zeros(2)
    obs[cell_1] = 0.5
    obs[cell_2] = 1

    observation_operator = Observations(fs)
    sigma = 1.0
    new_ensemble = kalman_update(ensemble, observation_operator, coords, obs, sigma)

    # compare solutions
    XNew_trial = np.zeros((2, 2))
    XNew_trial[:, 0] = new_ensemble[0].dat.data[:]
    XNew_trial[:, 1] = new_ensemble[1].dat.data[:]

    assert np.max(np.abs(XNew_trial - XNew)) < 1e-5


if __name__ == "__main__":
    import os
    import pytest
    pytest.main(os.path.abspath(__file__))
