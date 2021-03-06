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


def update_Dictionary(Dict, ensemble, generic_label, access):

    """ This updates a dictionary with an ensemble of functions and their labels

        :arg Dict: Dictionary to update
        :type Dict: dict

        :arg ensemble: ensemble of functions that need to be labeled
        :type ensemble: list / tuple

        :arg generic_label: The generic prefix of the function label, that then gets numbered
        :type generic_label: str

        :arg access: Access level of functions in ensemble
        :type access: str. Either READ, WRITE or RW

    """

    n = len(ensemble)

    if type(Dict) is not dict:
        raise ValueError("Dictionary to update is not of dict type")

    if type(generic_label) is not str:
        raise ValueError("label for ensemble functions must be of str type")

    access_opts = ["READ", "WRITE", "RW"]
    if (type(access) is not str) or (access not in access_opts):
        raise ValueError("Access option is not of str type or an available access level")

    for i in range(n):

        member_label = generic_label + str(i)
        if access == "READ":
            Dict.update({member_label: (ensemble[i], READ)})
        if access == "WRITE":
            Dict.update({member_label: (ensemble[i], WRITE)})
        if access == "RW":
            Dict.update({member_label: (ensemble[i], RW)})

    return Dict
