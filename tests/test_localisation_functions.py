from __future__ import absolute_import

from __future__ import division

from firedrake import *
from firedrake_da import *


def generate_localisation_functions(fs, r_loc_func):

    L = LocalisationFunctions(fs, r_loc_func)
    for i in range(len(L)):
        assert np.abs(1.0 - L[i].dat.data[i]) < 1e-5


def test_loc_func_int_dg0_r0():
    m = UnitIntervalMesh(2)
    fs = FunctionSpace(m, 'DG', 0)
    generate_localisation_functions(fs, 0)


def test_loc_func_int_dg1_r0():
    m = UnitIntervalMesh(2)
    fs = FunctionSpace(m, 'DG', 1)
    generate_localisation_functions(fs, 0)


def test_loc_func_int_dg0_r1():
    m = UnitIntervalMesh(2)
    fs = FunctionSpace(m, 'DG', 0)
    generate_localisation_functions(fs, 1)


def test_loc_func_int_dg1_r1():
    m = UnitIntervalMesh(2)
    fs = FunctionSpace(m, 'DG', 1)
    generate_localisation_functions(fs, 1)


def test_loc_func_sq_dg0_r0():
    m = UnitSquareMesh(2, 2)
    fs = FunctionSpace(m, 'DG', 0)
    generate_localisation_functions(fs, 0)


def test_loc_func_sq_dg1_r0():
    m = UnitSquareMesh(2, 2)
    fs = FunctionSpace(m, 'DG', 1)
    generate_localisation_functions(fs, 0)


def test_loc_func_sq_dg0_r1():
    m = UnitSquareMesh(2, 2)
    fs = FunctionSpace(m, 'DG', 0)
    generate_localisation_functions(fs, 1)


def test_loc_func_sq_dg1_r1():
    m = UnitSquareMesh(2, 2)
    fs = FunctionSpace(m, 'DG', 1)
    generate_localisation_functions(fs, 1)


def test_loc_func_int_cg1_r0():
    m = UnitIntervalMesh(2)
    fs = FunctionSpace(m, 'CG', 1)
    generate_localisation_functions(fs, 0)


def test_loc_func_int_cg1_r1():
    m = UnitIntervalMesh(2)
    fs = FunctionSpace(m, 'CG', 1)
    generate_localisation_functions(fs, 1)


def test_loc_func_sq_cg1_r0():
    m = UnitSquareMesh(2, 2)
    fs = FunctionSpace(m, 'CG', 1)
    generate_localisation_functions(fs, 0)


def test_loc_func_sq_cg1_r1():
    m = UnitSquareMesh(2, 2)
    fs = FunctionSpace(m, 'CG', 1)
    generate_localisation_functions(fs, 1)


def test_loc_func_length():
    m = UnitIntervalMesh(10)
    fs = FunctionSpace(m, 'DG', 0)
    L = LocalisationFunctions(fs, 0)
    assert len(L) == 10


def test_loc_func_iterate():
    m = UnitIntervalMesh(10)
    fs = FunctionSpace(m, 'DG', 0)
    L = LocalisationFunctions(fs, 1)
    i = 0
    for l in L:
        if i == 0:
            assert l.dat.data[i] == 1.0
            assert l.dat.data[i + 1] == 0.5
        elif i == len(L) - 1:
            assert l.dat.data[i] == 1.0
            assert l.dat.data[i - 1] == 0.5
        else:
            assert l.dat.data[i] == 1.0
            assert l.dat.data[i + 1] == 0.5
            assert l.dat.data[i - 1] == 0.5
        i += 1


if __name__ == "__main__":
    import os
    import pytest
    pytest.main(os.path.abspath(__file__))
