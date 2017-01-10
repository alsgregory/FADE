from __future__ import division

from __future__ import absolute_import

from firedrake import *

from firedrake_da import *

import numpy as np

import pytest


def test_point_to_cell():

    mesh = UnitIntervalMesh(1)

    coords = [np.array([0.5])]

    cells = PointToCell(coords, mesh)

    assert cells[0] == 0
    assert len(cells) == 1


def test_cell_to_node_dg0():

    mesh = UnitIntervalMesh(1)
    fs = FunctionSpace(mesh, 'DG', 0)

    cells = [np.array([0])]

    nodes = CellToNode(cells, fs)

    assert nodes[0][0] == 0
    assert len(nodes[0]) == 1


def test_cell_to_node_dg1():

    mesh = UnitIntervalMesh(1)
    fs = FunctionSpace(mesh, 'DG', 1)

    cells = [np.array([0])]

    nodes = CellToNode(cells, fs)

    assert nodes[0][0][0] == 0
    assert nodes[0][0][1] == 1
    assert len(nodes[0][0]) == 2


if __name__ == "__main__":
    import os
    pytest.main(os.path.abspath(__file__))
