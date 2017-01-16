from __future__ import division

from __future__ import absolute_import

from firedrake import *

from firedrake.mg.utils import get_level

from fade import *

import pytest


def test_1_cell_1d():

    mesh = FadeMesh("IntervalMesh", 1, 1)

    h, l = get_level(mesh)

    assert len(h) == 1
    assert l == 0
    assert mesh.num_cells() == 1
    assert mesh == h[-1]


def test_1_cell_2d():

    mesh = FadeMesh("SquareMesh", 1, 1, 1)

    h, l = get_level(mesh)

    assert len(h) == 1
    assert l == 0
    assert mesh.num_cells() == 2
    assert mesh == h[-1]


def test_16_cell_1d():

    mesh = FadeMesh("IntervalMesh", 16, 1)

    h, l = get_level(mesh)

    assert len(h) == 5
    assert l == 4
    assert mesh.num_cells() == 16
    assert h[0].num_cells() == 1
    assert mesh == h[-1]


def test_16_cell_2d():

    mesh = FadeMesh("SquareMesh", 16, 16, 1)

    h, l = get_level(mesh)

    assert len(h) == 5
    assert l == 4
    assert mesh.num_cells() == 512
    assert h[0].num_cells() == 2
    assert mesh == h[-1]


if __name__ == "__main__":
    import os
    pytest.main(os.path.abspath(__file__))
