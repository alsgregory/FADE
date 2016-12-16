from __future__ import division

from __future__ import absolute_import

from firedrake import *

from firedrake_da import *

import numpy as np

import pytest


def test_coarsening_localisation_single_cell_dg0():

    mesh = UnitIntervalMesh(1)

    mesh_hierarchy = MeshHierarchy(mesh, 3)

    r_loc = 2

    fs_hierarchy = tuple([FunctionSpace(m, 'DG', 0) for m in mesh_hierarchy])

    WLoc = Function(fs_hierarchy[-1]).assign(1.0)
    WLoc_ = Function(fs_hierarchy[-1]).assign(WLoc)

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

    assert np.abs(0.5 - WLoc.dat.data[0]) < 1e-5
    assert np.abs(0.5 - WLoc.dat.data[1]) < 1e-5


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

        if r == 1:
            with pytest.raises(Exception):
                CoarseningLocalisation(WLoc, r)

        if r == 0:
            WLoc = CoarseningLocalisation(WLoc, r)
            assert norm(assemble(WLoc - WLoc_)) == 0


def test_covariance_localisation_1():

    mesh = UnitIntervalMesh(10)

    vfs = VectorFunctionSpace(mesh, 'DG', 0, dim=10)

    r_loc = 2

    L = CovarianceLocalisation(vfs, r_loc)

    assert np.shape(L.dat.data) == (10, 10)

    assert np.max(np.abs(np.diagonal(L.dat.data) - np.ones(10))) < 1e-5
    assert np.max(np.abs(np.diagonal(L.dat.data, 1) - (0.75 * np.ones(9)))) < 1e-5
    assert np.max(np.abs(np.diagonal(L.dat.data, 2) - (0.25 * np.ones(8)))) < 1e-5
    assert np.max(np.abs(np.diagonal(L.dat.data, -1) - (0.75 * np.ones(9)))) < 1e-5
    assert np.max(np.abs(np.diagonal(L.dat.data, -2) - (0.25 * np.ones(8)))) < 1e-5


def test_covariance_localisation_2():

    mesh = UnitIntervalMesh(10)

    fs = FunctionSpace(mesh, 'CG', 1)
    dim = fs.dof_dset.size
    vfs = VectorFunctionSpace(mesh, 'CG', 1, dim=dim)

    r_loc = 0

    L = CovarianceLocalisation(vfs, r_loc)

    assert np.shape(L.dat.data) == (dim, dim)

    assert np.max(np.abs(np.diagonal(L.dat.data) - np.ones(dim))) < 1e-5


def test_covariance_localisation_3():

    mesh = UnitSquareMesh(10, 10)

    fs = FunctionSpace(mesh, 'DG', 1)
    dim = fs.dof_dset.size
    vfs = VectorFunctionSpace(mesh, 'DG', 1, dim=dim)

    r_loc = 0

    L = CovarianceLocalisation(vfs, r_loc)

    assert np.shape(L.dat.data) == (dim, dim)

    assert np.max(np.abs(np.diagonal(L.dat.data) - np.ones(dim))) < 1e-5


if __name__ == "__main__":
    import os
    pytest.main(os.path.abspath(__file__))
