from __future__ import division

from __future__ import absolute_import

from firedrake import *

from fade import *

import numpy as np


def test_observation_one_element_node():

    mesh = UnitIntervalMesh(1)
    fs = FunctionSpace(mesh, 'DG', 0)

    coord = tuple([np.array([0.5])])

    obs = tuple([1])

    R = 0.05

    # check that only cell is showing as containing obs
    obs_operator = Observations(fs, R)
    obs_operator.update_observation_operator(coord, obs)

    assert obs_operator.cells == np.array([0])
    assert obs_operator.nodes == np.array([0])


def test_observation_one_element_difference():

    mesh = UnitIntervalMesh(1)

    coord = tuple([np.array([0.5])])

    obs = tuple([2])

    R = 0.05

    # iterate over degrees for in-function
    d = tuple([1, 2])
    for deg in d:
        fs = FunctionSpace(mesh, 'DG', deg)

        obs_operator = Observations(fs, R)
        obs_operator.update_observation_operator(coord, obs)

        # make in-function for difference
        f = Function(fs)

        diff = Function(fs)
        diff.assign(obs_operator.difference(f))

        # check that all ((2 * d) + 1) basis coefficients is same difference
        assert len(diff.dat.data) == (1 + deg)
        for i in range(len(diff.dat.data)):
            assert np.abs(diff.dat.data[i] - 4.0) < 1e-8


def test_observation_cells_and_nodes():

    mesh = UnitIntervalMesh(1)
    fs = FunctionSpace(mesh, 'DG', 0)

    coord = tuple([np.array([0.5])])

    obs = tuple([2.0])

    R = 0.05

    obs_operator = Observations(fs, R)

    obs_operator.update_observation_operator(coord, obs)

    assert len(obs_operator.cells) == 1
    assert obs_operator.cells[0] == 0
    assert np.unique(obs_operator.nodes == np.array([0]))


def test_observation_difference():

    mesh = UnitSquareMesh(10, 10)
    fs = FunctionSpace(mesh, 'DG', 0)

    coord = tuple([np.array([0.1, 0.1]), np.array([0.5, 0.5])])

    obs = tuple([2, 2])

    R = 0.05

    obs_operator = Observations(fs, R)

    f = Function(fs)

    obs_operator.update_observation_operator(coord, obs)
    diff = Function(fs)
    diff.assign(obs_operator.difference(f))

    assert np.sum(diff.dat.data) == 8

    assert len(np.unique(obs_operator.nodes)) == 2


def test_update_observation_difference():

    mesh = UnitIntervalMesh(1)
    fs = FunctionSpace(mesh, 'DG', 0)

    coord = tuple([np.array([0.5])])

    obs = tuple([2.0])

    R = 0.05

    obs_operator = Observations(fs, R)

    f = Function(fs)

    obs_operator.update_observation_operator(coord, obs)
    diff_1 = Function(fs)
    diff_1.assign(obs_operator.difference(f))
    assert np.max(np.abs(diff_1.dat.data - 4.0)) < 1e-5

    coord = tuple([np.array([0.5])])

    obs = tuple([3.0])

    f.assign(2.0)

    obs_operator.update_observation_operator(coord, obs)
    assert obs_operator.observations == tuple([3.0])
    diff_2 = Function(fs)
    diff_2.assign(obs_operator.difference(f))
    assert np.max(np.abs(diff_2.dat.data - 1.0)) < 1e-5


def test_no_observations():

    mesh = UnitIntervalMesh(2)
    fs = FunctionSpace(mesh, 'DG', 0)

    coord = tuple([])

    obs = tuple([])

    R = 0.05

    obs_operator = Observations(fs, R)

    obs_operator.update_observation_operator(coord, obs)

    func = Function(fs).assign(1.0)
    p = 2

    diff = obs_operator.difference(func, p)
    assert np.max(np.abs(diff.dat.data[:])) < 1e-5


def test_no_observations_2():

    mesh = UnitIntervalMesh(2)
    fs = FunctionSpace(mesh, 'DG', 0)

    coord = tuple([np.array([0.25])])
    # cell number of the other cell
    cell = mesh.locate_cell(np.array([0.75]))

    obs = tuple([1.0])

    R = 0.05

    obs_operator = Observations(fs, R)

    obs_operator.update_observation_operator(coord, obs)

    func = Function(fs).assign(1.0)
    p = 2

    diff = obs_operator.difference(func, p)
    assert diff.dat.data[cell] == 0.0


if __name__ == "__main__":
    import os
    import pytest
    pytest.main(os.path.abspath(__file__))
