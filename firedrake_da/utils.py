""" util functions / kernels for data assimilation in Firedrake """

from __future__ import division

from __future__ import absolute_import

from firedrake import *

import numpy as np


def PointToCell(pointsList, mesh):

    """ finds cells on a :class:`Mesh` that contain points in pointsList """

    # number of points to find cells of
    ny = len(pointsList)

    # Initialize cell array
    cells = np.zeros(ny)

    # iterate through points finding cells
    for i in range(ny):
        cells[i] = mesh.locate_cell(pointsList[i])

    return cells


def CellToNode(cells, fs):

    """ finds nodes in a :class:`FunctionSpace` """

    # number of cells
    ny = len(cells)

    # initialize node list
    nodes = []

    # iterate through cells finding nodes
    for i in range(ny):
        nodes.append(fs.cell_node_list[cells[i].astype(int)])

    return nodes
