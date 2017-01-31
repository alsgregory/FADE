from __future__ import division

from __future__ import absolute_import

from fade import *

from firedrake import *

import numpy as np

import pytest


def test_invariant_transform():

    n = 10

    mesh = UnitIntervalMesh(1)

    V = FunctionSpace(mesh, 'DG', 0)

    # no coarsening localisation needed
    r_loc = 0

    weights = []
    ensemble = []
    keep_ensemble = []
    for i in range(n):
        f = Function(V)
        f.assign(i)
        g = Function(V)
        g.assign(1.0 / n)
        weights.append(g)
        ensemble.append(f)
        keep_ensemble.append(f)

    new_ensemble = ensemble_transform_update(ensemble, weights, r_loc)

    for i in range(n):
        assert np.max(np.abs(new_ensemble[i].dat.data - keep_ensemble[i].dat.data)) < 1e-5


def test_localisation():

    mesh = UnitIntervalMesh(1)

    mesh_hierarchy = MeshHierarchy(mesh, 2)

    V = VectorFunctionSpace(mesh_hierarchy[-1], 'DG', 0, dim=1)

    ensemble_f = Function(V).assign(1)

    r_locs = [0, 1]
    for r_loc in r_locs:
        cost_tensor = generate_localised_cost_tensor(ensemble_f, ensemble_f, r_loc)
        assert np.max(np.abs(cost_tensor.dat.data[:])) < 1e-5


def test_coarsening_cost_tensor():

    mesh = UnitIntervalMesh(1)

    mesh_hierarchy = MeshHierarchy(mesh, 1)

    V = VectorFunctionSpace(mesh_hierarchy[1], 'DG', 0, dim=3)

    ensemble_f = Function(V)
    ensemble_f.dat.data[0, 0] = 1
    ensemble_f.dat.data[0, 1] = 2
    ensemble_f.dat.data[0, 2] = 2

    # this will aggregate the difference in one finer subcell between the one coarse cell
    r_loc = 1
    cost_tensor = generate_localised_cost_tensor(ensemble_f, ensemble_f, r_loc)
    assert np.max(np.abs(cost_tensor.dat.data[:, 1, 0] - 1)) < 1e-5
    assert np.max(np.abs(cost_tensor.dat.data[:, 0, 1] - 1)) < 1e-5
    assert np.max(np.abs(cost_tensor.dat.data[:, 2, 0] - 1)) < 1e-5
    assert np.max(np.abs(cost_tensor.dat.data[:, 0, 2] - 1)) < 1e-5


def test_coarsening_cost_tensor_2():

    mesh = UnitIntervalMesh(1)

    mesh_hierarchy = MeshHierarchy(mesh, 1)

    V = VectorFunctionSpace(mesh_hierarchy[1], 'DG', 0, dim=3)

    ensemble_f = Function(V)
    ensemble_f.dat.data[0, 0] = 1
    ensemble_f.dat.data[0, 1] = 2
    ensemble_f.dat.data[0, 2] = 2
    ensemble2_f = Function(V)
    ensemble2_f.dat.data[0, 0] = 1
    ensemble2_f.dat.data[0, 1] = 3
    ensemble2_f.dat.data[0, 2] = 3

    # this will aggregate the difference in one finer subcell between the one coarse cell
    r_loc = 1
    cost_tensor = generate_localised_cost_tensor(ensemble_f, ensemble2_f, r_loc)
    assert np.max(np.abs(cost_tensor.dat.data[:, 0, 0] - 0)) < 1e-5
    assert np.max(np.abs(cost_tensor.dat.data[:, 1, 1] - 1)) < 1e-5
    assert np.max(np.abs(cost_tensor.dat.data[:, 2, 2] - 1)) < 1e-5
    assert np.max(np.abs(cost_tensor.dat.data[:, 0, 1] - 4)) < 1e-5
    assert np.max(np.abs(cost_tensor.dat.data[:, 1, 0] - 1)) < 1e-5
    assert np.max(np.abs(cost_tensor.dat.data[:, 0, 2] - 4)) < 1e-5
    assert np.max(np.abs(cost_tensor.dat.data[:, 2, 0] - 1)) < 1e-5
    assert np.max(np.abs(cost_tensor.dat.data[:, 1, 2] - 1)) < 1e-5
    assert np.max(np.abs(cost_tensor.dat.data[:, 2, 1] - 1)) < 1e-5


def test_coarsening_cost_tensor_assembly():

    mesh = UnitIntervalMesh(1)

    mesh_hierarchy = MeshHierarchy(mesh, 1)

    V = VectorFunctionSpace(mesh_hierarchy[1], 'DG', 0, dim=3)

    ensemble_f = Function(V)
    ensemble_f.dat.data[0, 0] = 1
    ensemble_f.dat.data[0, 1] = 2
    ensemble_f.dat.data[0, 2] = 2

    # this will aggregate the difference in one finer subcell between the one coarse cell
    r_loc = 1
    cost_tensor = generate_localised_cost_tensor(ensemble_f, ensemble_f, r_loc, "assembly")
    assert np.max(np.abs(cost_tensor.dat.data[:, 1, 0] - 1)) < 1e-5
    assert np.max(np.abs(cost_tensor.dat.data[:, 0, 1] - 1)) < 1e-5
    assert np.max(np.abs(cost_tensor.dat.data[:, 2, 0] - 1)) < 1e-5
    assert np.max(np.abs(cost_tensor.dat.data[:, 0, 2] - 1)) < 1e-5


def test_localisation_diff():

    mesh = UnitIntervalMesh(1)

    mesh_hierarchy = MeshHierarchy(mesh, 2)

    V = VectorFunctionSpace(mesh_hierarchy[0], 'DG', 0, dim=2)

    ensemble_f = Function(V)
    ensemble_f.dat.data[:, 0] = 1
    ensemble_f.dat.data[:, 1] = 2

    r_loc = 0
    cost_tensor = generate_localised_cost_tensor(ensemble_f, ensemble_f, r_loc)
    assert cost_tensor.dat.data[0, 0, 0] == 0
    assert cost_tensor.dat.data[0, 0, 1] == 1
    assert cost_tensor.dat.data[0, 1, 0] == 1
    assert cost_tensor.dat.data[0, 1, 1] == 0


def test_localisation_cg_proj():

    mesh = UnitIntervalMesh(1)

    mesh_hierarchy = MeshHierarchy(mesh, 2)

    V = VectorFunctionSpace(mesh_hierarchy[0], 'CG', 1, dim=2)

    ensemble_f = Function(V).assign(1)

    ensemble2_f = Function(V)

    r_loc = 0
    cost_tensor = generate_localised_cost_tensor(ensemble_f, ensemble2_f, r_loc)
    assert cost_tensor.dat.data[0, 0, 0] == 1
    assert cost_tensor.dat.data[0, 0, 1] == 1
    assert cost_tensor.dat.data[0, 1, 0] == 1
    assert cost_tensor.dat.data[0, 1, 1] == 1


if __name__ == "__main__":
    import os
    pytest.main(os.path.abspath(__file__))
