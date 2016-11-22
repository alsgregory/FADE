from __future__ import division

from __future__ import absolute_import

from firedrake import *

from firedrake_da import *

import numpy as np


def test_observation_one_element_node():

    mesh = UnitIntervalMesh(1)

    coord = tuple([np.array([0.5])])

    obs = tuple([1])

    # check that only cell is showing as containing obs
    obs_operator = Observations(coord, obs, mesh)

    assert obs_operator.cells == np.array([0])
    assert obs_operator.nodes == np.array([0])


def test_observation_one_element_difference():

    mesh = UnitIntervalMesh(1)

    coord = tuple([np.array([0.5])])

    obs = tuple([2])

    obs_operator = Observations(coord, obs, mesh)

    # iterate over degrees for in-function
    d = tuple([1, 2])

    for deg in d:
        # make in-function for difference
        v = FunctionSpace(mesh, 'DG', deg)
        f = Function(v)

        diff = obs_operator.difference(f)

        # check that all ((2 * d) + 1) basis coefficients is same difference
        assert len(diff.dat.data) == (1 + deg)
        for i in range(len(diff.dat.data)):
            assert np.abs(diff.dat.data[i] - 4.0) < 1e-8


def test_observation_difference():

    mesh = UnitSquareMesh(10, 10)

    coord = tuple([np.array([0.1, 0.1]), np.array([0.5, 0.5])])

    obs = tuple([2, 2])

    obs_operator = Observations(coord, obs, mesh)

    v = FunctionSpace(mesh, 'DG', 0)
    f = Function(v)

    diff = obs_operator.difference(f)

    assert np.sum(diff.dat.data) == 8

    assert len(np.unique(obs_operator.nodes)) == 2


if __name__ == "__main__":
    import os
    import pytest
    pytest.main(os.path.abspath(__file__))
