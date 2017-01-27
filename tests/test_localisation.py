from __future__ import division

from __future__ import absolute_import

from firedrake import *

from fade import *

import numpy as np

import pytest


def test_coarsening_localisation_single_cell_dg0():

    mesh = UnitIntervalMesh(1)

    mesh_hierarchy = MeshHierarchy(mesh, 3)

    r_loc = 2

    fs_hierarchy = tuple([FunctionSpace(m, 'DG', 0) for m in mesh_hierarchy])

    WLoc = Function(fs_hierarchy[-1]).assign(1.0)
    WLoc_ = Function(fs_hierarchy[-1]).assign(WLoc * 4)

    WLoc = CoarseningLocalisation(WLoc, r_loc)

    assert norm(assemble(WLoc - WLoc_)) == 0


def test_coarsening_localisation_single_cell_dg0_2():

    mesh = UnitIntervalMesh(1)

    mesh_hierarchy = MeshHierarchy(mesh, 1)

    r_loc = 1

    fs_hierarchy = tuple([FunctionSpace(m, 'DG', 0) for m in mesh_hierarchy])

    WLoc = Function(fs_hierarchy[1])
    WLoc.dat.data[0] = 1.0

    WLoc = CoarseningLocalisation(WLoc, r_loc)

    assert np.abs(1 - WLoc.dat.data[0]) < 1e-5
    assert np.abs(1 - WLoc.dat.data[1]) < 1e-5


def test_coarsening_localisation_no_localisation():

    mesh = UnitIntervalMesh(1)

    mesh_hierarchy = MeshHierarchy(mesh, 3)

    r_loc = 0

    fs_hierarchy = tuple([FunctionSpace(m, 'DG', 0) for m in mesh_hierarchy])

    WLoc = Function(fs_hierarchy[-1])

    WLoc.dat.data[0] += 1.0

    WLoc_ = Function(fs_hierarchy[-1]).assign(WLoc)

    WLoc = CoarseningLocalisation(WLoc, r_loc)

    assert norm(assemble(WLoc - WLoc_)) == 0


def test_coarsening_localisation_no_hierarchy():

    mesh = UnitIntervalMesh(1)

    fs = FunctionSpace(mesh, 'DG', 0)

    r_locs = [0, 1]

    for r in r_locs:

        WLoc = Function(fs)

        WLoc.dat.data[0] += 1.0

        WLoc_ = Function(fs).assign(WLoc)

        WLoc = CoarseningLocalisation(WLoc, r)
        assert norm(assemble(WLoc - WLoc_)) == 0


def test_coarsening_localisation_list():

    mesh = UnitIntervalMesh(1)

    mesh_hierarchy = MeshHierarchy(mesh, 1)

    r_loc = 1

    fs_hierarchy = tuple([FunctionSpace(m, 'DG', 0) for m in mesh_hierarchy])

    WLoc = []
    for i in range(2):
        WLoc.append(Function(fs_hierarchy[1]))
        WLoc[i].dat.data[0] = 1.0

    WLoc = CoarseningLocalisation(WLoc, r_loc)

    for i in range(2):
        assert np.abs(1 - WLoc[i].dat.data[0]) < 1e-5
        assert np.abs(1 - WLoc[i].dat.data[1]) < 1e-5


if __name__ == "__main__":
    import os
    pytest.main(os.path.abspath(__file__))
