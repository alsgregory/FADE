from __future__ import division

from __future__ import absolute_import

from firedrake import *

from fade import *
from fade.ensemble_transform import *

import numpy as np


def test_ensemble_transform_mean_preserving():

    mesh = UnitIntervalMesh(1)

    mesh_hierarchy = MeshHierarchy(mesh, 3)
    r_loc = 2
    r_loc_cost = 0

    coord = tuple([np.array([0.5])])

    obs = tuple([1])

    # build ensemble
    fs = FunctionSpace(mesh_hierarchy[-1], 'DG', 0)
    ensemble = [Function(fs), Function(fs)]
    weights = [Function(fs), Function(fs)]
    weights[0].assign(0.5)
    weights[1].assign(0.5)
    ensemble[0].assign(1.0)
    ensemble[1].assign(1.0)

    # compute weights - should be even
    sigma = 0.1
    observation_operator = Observations(fs)
    observation_operator.update_observation_operator(coord, obs)
    weights = weight_update(ensemble, weights, observation_operator, sigma, r_loc)

    # compute ensemble transform - should be 1.0's
    new_ensemble = ensemble_transform_update(ensemble, weights, r_loc_cost)

    assert np.max(new_ensemble[0].dat.data[:] - 1.0) < 1e-5
    assert np.max(new_ensemble[1].dat.data[:] - 1.0) < 1e-5


def test_transform():

    mesh = UnitIntervalMesh(1)

    V = FunctionSpace(mesh, 'DG', 0)

    # no coarsening localisation needed
    r_loc = 0

    weights = []
    ensemble = []
    f = Function(V).assign(0.5)
    ensemble.append(f)
    f = Function(V).assign(1.0)
    ensemble.append(f)
    g = Function(V).assign(0.25)
    weights.append(g)
    g = Function(V).assign(0.75)
    weights.append(g)

    new_ensemble = ensemble_transform_update(ensemble, weights, r_loc)

    assert new_ensemble[0].dat.data[0] == 0.75
    assert new_ensemble[1].dat.data[0] == 1.0

    # check if old ensemble was changed as well
    assert ensemble[0].dat.data[0] == 0.75
    assert ensemble[1].dat.data[0] == 1.0


def test_reset_weights():

    mesh = UnitIntervalMesh(1)

    V = FunctionSpace(mesh, 'DG', 0)

    # no coarsening localisation needed
    r_loc = 0

    weights = []
    ensemble = []
    f = Function(V).assign(0.5)
    ensemble.append(f)
    f = Function(V).assign(1.0)
    ensemble.append(f)
    g = Function(V).assign(0.25)
    weights.append(g)
    g = Function(V).assign(0.75)
    weights.append(g)

    ensemble_transform_update(ensemble, weights, r_loc)

    assert np.max(np.abs(weights[0].dat.data[0] - 0.5)) < 1e-5
    assert np.max(np.abs(weights[1].dat.data[0] - 0.5)) < 1e-5


if __name__ == "__main__":
    import os
    import pytest
    pytest.main(os.path.abspath(__file__))
