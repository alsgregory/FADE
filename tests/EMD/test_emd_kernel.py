from __future__ import division

from __future__ import absolute_import

from firedrake_da import *

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
    for i in range(n):
        f = Function(V)
        f.assign(i)
        g = Function(V)
        g.assign(1.0 / n)
        weights.append(g)
        ensemble.append(f)

    new_ensemble = ensemble_transform_update(ensemble, weights, r_loc)

    for i in range(n):
        assert np.max(np.abs(new_ensemble[i].dat.data - ensemble[i].dat.data)) < 1e-5


def test_localisation():

    mesh = UnitIntervalMesh(1)

    mesh_hierarchy = MeshHierarchy(mesh, 2)

    V = FunctionSpace(mesh_hierarchy[-1], 'DG', 0)

    ensemble = []
    f = Function(V).assign(1)
    ensemble.append(f)

    r_locs = [0, 1]
    for r_loc in r_locs:
        cost_tensor = generate_localised_cost_tensor(ensemble, ensemble, r_loc)
        assert np.max(np.abs(cost_tensor.dat.data[:])) < 1e-5


def test_coarsening_cost_tensor():

    mesh = UnitIntervalMesh(1)

    mesh_hierarchy = MeshHierarchy(mesh, 1)

    V = FunctionSpace(mesh_hierarchy[1], 'DG', 0)

    ensemble = []
    f = Function(V)
    f.dat.data[0] = 1
    ensemble.append(f)
    f = Function(V)
    f.dat.data[0] = 2
    ensemble.append(f)
    f = Function(V)
    f.dat.data[0] = 2
    ensemble.append(f)

    # this will aggregate the difference in one finer subcell between the one coarse cell
    r_loc = 1
    cost_tensor = generate_localised_cost_tensor(ensemble, ensemble, r_loc)
    assert np.max(np.abs(cost_tensor.dat.data[:, 1, 0] - 0.5)) < 1e-5
    assert np.max(np.abs(cost_tensor.dat.data[:, 0, 1] - 0.5)) < 1e-5
    assert np.max(np.abs(cost_tensor.dat.data[:, 2, 0] - 0.5)) < 1e-5
    assert np.max(np.abs(cost_tensor.dat.data[:, 0, 2] - 0.5)) < 1e-5


def test_localisation_diff():

    mesh = UnitIntervalMesh(1)

    mesh_hierarchy = MeshHierarchy(mesh, 2)

    V = FunctionSpace(mesh_hierarchy[0], 'DG', 0)

    ensemble = []
    f = Function(V).assign(1)
    ensemble.append(f)
    f = Function(V).assign(2)
    ensemble.append(f)

    r_loc = 0
    cost_tensor = generate_localised_cost_tensor(ensemble, ensemble, r_loc)
    assert cost_tensor.dat.data[0, 0, 0] == 0
    assert cost_tensor.dat.data[0, 0, 1] == 1
    assert cost_tensor.dat.data[0, 1, 0] == 1
    assert cost_tensor.dat.data[0, 1, 1] == 0


def test_localisation_cg_proj():

    mesh = UnitIntervalMesh(1)

    mesh_hierarchy = MeshHierarchy(mesh, 2)

    V = FunctionSpace(mesh_hierarchy[0], 'CG', 1)

    ensemble = []
    f = Function(V).assign(1)
    ensemble.append(f)
    f = Function(V).assign(1)
    ensemble.append(f)

    ensemble2 = []
    f = Function(V)
    ensemble2.append(f)
    f = Function(V)
    ensemble2.append(f)

    r_loc = 0
    cost_tensor = generate_localised_cost_tensor(ensemble, ensemble2, r_loc)
    assert cost_tensor.dat.data[0, 0, 0] == 1
    assert cost_tensor.dat.data[0, 0, 1] == 1
    assert cost_tensor.dat.data[0, 1, 0] == 1
    assert cost_tensor.dat.data[0, 1, 1] == 1


if __name__ == "__main__":
    import os
    pytest.main(os.path.abspath(__file__))
