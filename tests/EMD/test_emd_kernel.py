from __future__ import division

from __future__ import absolute_import

from firedrake import *

from firedrake_da import *

import numpy as np

import pytest


def test_invariant_transform():

    n = 10

    mesh = UnitIntervalMesh(1)

    V = FunctionSpace(mesh, 'DG', 0)

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

    f = Function(V)
    cost_funcs = ([[f]])
    r_locs = [0, 1]
    for r_loc in r_locs:
        cost_funcs = generate_localised_cost_funcs(ensemble, ensemble, cost_funcs, r_loc)
        assert np.max(np.abs(cost_funcs[0][0].dat.data[:])) < 1e-5


def test_coarsening_cost_function():

    mesh = UnitIntervalMesh(1)

    mesh_hierarchy = MeshHierarchy(mesh, 1)

    V = FunctionSpace(mesh_hierarchy[-1], 'DG', 0)
    Vc = FunctionSpace(mesh_hierarchy[0], 'DG', 0)

    ensemble = []
    f = Function(V)
    f.dat.data[0] = 1
    ensemble.append(f)
    f = Function(V)
    f.dat.data[0] = 2
    ensemble.append(f)

    cost_funcs = []
    for i in range(2):
        cost_funcs.append([])
        for j in range(2):
            f = Function(V)
            cost_funcs[i].append(f)

    # inject then prolong to work out actual cost func on finer mesh
    act = Function(V)
    act.dat.data[0] = 1.0
    inj = Function(Vc)
    inject(act, inj)
    inj_pro = Function(V)
    prolong(inj, inj_pro)

    # this will aggregate the difference in one finer subcell between the one coarse cell
    r_loc = 1
    cost_funcs = generate_localised_cost_funcs(ensemble, ensemble, cost_funcs, r_loc)
    assert np.max(np.abs(cost_funcs[1][0].dat.data[:] - inj_pro.dat.data[:])) < 1e-5
    assert np.max(np.abs(cost_funcs[0][1].dat.data[:] - inj_pro.dat.data[:])) < 1e-5


def test_localisation_diff():

    mesh = UnitIntervalMesh(1)

    mesh_hierarchy = MeshHierarchy(mesh, 2)

    V = FunctionSpace(mesh_hierarchy[0], 'DG', 0)

    ensemble = []
    f = Function(V).assign(1)
    ensemble.append(f)
    f = Function(V).assign(2)
    ensemble.append(f)

    cost_funcs = []
    for i in range(2):
        cost_funcs.append([])
        for j in range(2):
            f = Function(V)
            cost_funcs[i].append(f)

    r_loc = 0
    cost_funcs = generate_localised_cost_funcs(ensemble, ensemble, cost_funcs, r_loc)
    assert cost_funcs[0][0].dat.data[0] == 0
    assert cost_funcs[0][1].dat.data[0] == 1
    assert cost_funcs[1][0].dat.data[0] == 1
    assert cost_funcs[1][1].dat.data[0] == 0
    assert len(cost_funcs) == 2
    assert len(cost_funcs[0]) == 2


if __name__ == "__main__":
    import os
    pytest.main(os.path.abspath(__file__))
