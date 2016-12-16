from __future__ import division

from __future__ import absolute_import

from firedrake import *

from firedrake_da import *

import numpy as np

import pytest


def test_hadamard_product_fs():

    mesh = UnitIntervalMesh(2)

    V = FunctionSpace(mesh, 'DG', 0)

    f1 = Function(V)
    f2 = Function(V)

    f1.dat.data[0] = 2.0
    f1.dat.data[1] = 1.0
    f2.dat.data[0] = 2.0
    f2.dat.data[1] = 1.0

    H = HadamardProduct(f1, f2)

    assert np.max(np.abs(H.dat.data[:] - np.array([4.0, 1.0]))) < 1e-5


def test_hadamard_product_vfs():

    mesh = UnitIntervalMesh(2)

    V = VectorFunctionSpace(mesh, 'DG', 0, dim=2)

    f1 = Function(V)
    f2 = Function(V)

    f1.dat.data[0, 0] = 2.0
    f1.dat.data[1, 0] = 1.0
    f1.dat.data[0, 1] = 1.0
    f1.dat.data[1, 1] = 2.0
    f2.dat.data[0, 0] = 2.0
    f2.dat.data[1, 0] = 1.0
    f2.dat.data[0, 1] = 1.0
    f2.dat.data[1, 1] = 2.0

    H = HadamardProduct(f1, f2)

    assert np.max(np.abs(H.dat.data[:] - np.array([[4.0, 1.0], [1.0, 4.0]]))) < 1e-5


def test_hadamard_product_tfs():

    mesh = UnitIntervalMesh(2)

    V = TensorFunctionSpace(mesh, 'DG', 0, ((2, 2)))

    f1 = Function(V)
    f2 = Function(V)

    f1.dat.data[0, 0, :] = 2.0
    f1.dat.data[1, 0, :] = 1.0
    f1.dat.data[0, 1, :] = 1.0
    f1.dat.data[1, 1, :] = 2.0
    f2.dat.data[0, 0, :] = 2.0
    f2.dat.data[1, 0, :] = 1.0
    f2.dat.data[0, 1, :] = 1.0
    f2.dat.data[1, 1, :] = 2.0

    H = HadamardProduct(f1, f2)

    assert np.max(np.abs(H.dat.data[:, :, 1] - np.array([[4.0, 1.0], [1.0, 4.0]]))) < 1e-5
    assert np.max(np.abs(H.dat.data[:, :, 0] - np.array([[4.0, 1.0], [1.0, 4.0]]))) < 1e-5


if __name__ == "__main__":
    import os
    pytest.main(os.path.abspath(__file__))
