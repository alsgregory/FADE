from __future__ import division

from __future__ import absolute_import

from firedrake import *

from fade import *
from fade.ml import *

import numpy as np


def test_coupling_mean_preserving():

    mesh = UnitIntervalMesh(1)

    mesh_hierarchy = MeshHierarchy(mesh, 3)
    r_loc = 2
    r_loc_cost = 0

    coord = tuple([np.array([0.5])])

    obs = tuple([1])

    # build ensemble
    fsc = FunctionSpace(mesh_hierarchy[-2], 'DG', 0)
    fsf = FunctionSpace(mesh_hierarchy[-1], 'DG', 0)
    ensemble_c = [Function(fsc), Function(fsc)]
    ensemble_f = [Function(fsf), Function(fsf)]
    weights_c = [Function(fsc), Function(fsc)]
    weights_f = [Function(fsf), Function(fsf)]
    ensemble_c[0].assign(1.0)
    ensemble_c[1].assign(1.0)
    ensemble_f[0].assign(1.0)
    ensemble_f[1].assign(1.0)
    weights_c[0].assign(0.5)
    weights_c[1].assign(0.5)
    weights_f[0].assign(0.5)
    weights_f[1].assign(0.5)

    # compute weights - should be even
    sigma = 0.1
    observation_operator_c = Observations(fsc, sigma)
    observation_operator_f = Observations(fsf, sigma)
    observation_operator_c.update_observation_operator(coord, obs)
    observation_operator_f.update_observation_operator(coord, obs)
    weights_c = weight_update(ensemble_c, weights_c, observation_operator_c, r_loc)
    weights_f = weight_update(ensemble_f, weights_f, observation_operator_f, r_loc)

    # compute ensemble transform - should be 1.0's
    new_ensemble_c, new_ensemble_f = seamless_coupling_update(ensemble_c, ensemble_f,
                                                              weights_c, weights_f, r_loc_cost,
                                                              r_loc_cost)

    assert np.max(new_ensemble_c[0].dat.data[:] - 1.0) < 1e-5
    assert np.max(new_ensemble_c[1].dat.data[:] - 1.0) < 1e-5

    assert np.max(new_ensemble_f[0].dat.data[:] - 1.0) < 1e-5
    assert np.max(new_ensemble_f[1].dat.data[:] - 1.0) < 1e-5


def test_reset_weights():

    mesh = UnitIntervalMesh(1)

    mesh_hierarchy = MeshHierarchy(mesh, 3)
    r_loc = 2
    r_loc_cost = 0

    coord = tuple([np.array([0.5])])

    obs = tuple([0.75])

    # build ensemble
    fsc = FunctionSpace(mesh_hierarchy[-2], 'DG', 0)
    fsf = FunctionSpace(mesh_hierarchy[-1], 'DG', 0)
    ensemble_c = [Function(fsc), Function(fsc)]
    ensemble_f = [Function(fsf), Function(fsf)]
    weights_c = [Function(fsc), Function(fsc)]
    weights_f = [Function(fsf), Function(fsf)]
    ensemble_c[0].assign(1.0)
    ensemble_c[1].assign(1.5)
    ensemble_f[0].assign(1.0)
    ensemble_f[1].assign(1.5)
    weights_c[0].assign(0.5)
    weights_c[1].assign(0.5)
    weights_f[0].assign(0.5)
    weights_f[1].assign(0.5)

    # compute weights - should be even
    sigma = 0.1
    observation_operator_c = Observations(fsc, sigma)
    observation_operator_f = Observations(fsf, sigma)
    observation_operator_c.update_observation_operator(coord, obs)
    observation_operator_f.update_observation_operator(coord, obs)
    weights_c = weight_update(ensemble_c, weights_c, observation_operator_c, r_loc)
    weights_f = weight_update(ensemble_f, weights_f, observation_operator_f, r_loc)

    # compute ensemble transform - should be 1.0's
    seamless_coupling_update(ensemble_c, ensemble_f, weights_c, weights_f, r_loc_cost,
                             r_loc_cost)

    assert np.max(np.abs(weights_c[0].dat.data[:] - 0.5)) < 1e-5
    assert np.max(np.abs(weights_c[1].dat.data[:] - 0.5)) < 1e-5

    assert np.max(np.abs(weights_f[0].dat.data[:] - 0.5)) < 1e-5
    assert np.max(np.abs(weights_f[1].dat.data[:] - 0.5)) < 1e-5


if __name__ == "__main__":
    import os
    import pytest
    pytest.main(os.path.abspath(__file__))
