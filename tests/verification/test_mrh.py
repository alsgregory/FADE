from __future__ import division

from __future__ import absolute_import

from firedrake import *

from firedrake_da import *

import numpy as np


def test_mrh_n_equals_1():

    mesh = UnitIntervalMesh(2)
    fs = FunctionSpace(mesh, 'DG', 0)
    f = Function(fs).assign(1)

    obs = np.array([0.0, 0.0])
    coords = np.array([np.array([0.25]), np.array([0.75])])

    # set-up ensemble
    ensemble = [f]

    # set-up rank histogram class
    R = rank_histogram(fs, 1)
    for i in range(20):
        R.compute_rank(ensemble, coords, obs)

    assert np.all(R.ranks[0] == 0.5)


def test_mrh_2_obs():

    mesh = UnitIntervalMesh(2)
    fs = FunctionSpace(mesh, 'DG', 0)
    f = Function(fs).assign(1)

    obs = np.array([0.0, 2.0])
    coords = np.array([np.array([0.25]), np.array([0.75])])

    # set-up ensemble
    ensemble = [f]

    # set-up rank histogram class
    R = rank_histogram(fs, 1)
    for i in range(20):
        R.compute_rank(ensemble, coords, obs)

    assert np.all(R.ranks[0] >= 0.5) and np.all(R.ranks[0] <= 1.0)


def test_choose_uniform_rank():

    n = 500000
    mesh = UnitIntervalMesh(1)
    fs = FunctionSpace(mesh, 'DG', 0)
    ranks = np.zeros(n)
    R = rank_histogram(fs, 1)

    for i in range(n):
        ranks[i] = R._rank_histogram__choose_uniform_rank(0, 2)

    assert np.abs(np.mean(ranks) - 1.5) < 2e-3


if __name__ == "__main__":
    import os
    import pytest
    pytest.main(os.path.abspath(__file__))
