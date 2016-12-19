""" util functions / kernels for data assimilation in Firedrake """

from __future__ import division

from __future__ import absolute_import

from firedrake import *

import scipy.sparse as scp

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


def HadamardProduct(f1, f2):

    """ Finds the Hadamard Product between two :class:`Function`s on the same space """

    # check that the functions exist on same space
    fs = f1.function_space()
    assert fs == f2.function_space()

    # define string for kernel for product
    vshape = fs.ufl_element().value_shape()

    itr_str = """ """
    if len(vshape) == 0:
        itr_str += """product[k][0]=f1[k][0]*f2[k][0];\n"""
    if len(vshape) == 1:
        for i in range(vshape[0]):
            itr_str += """product[k][""" + str(i) + """]=f1[k][""" + str(i) + """]*f2[k][""" + str(i) + """];\n"""
    if len(vshape) == 2:
        for i in range(vshape[0]):
            dim = vshape[0]
            for j in range(vshape[1]):
                itr_str += """product[k][""" + str((i * dim) + j) + """]=f1[k][""" + str((i * dim) + j) + """]*f2[k][""" + str((i * dim) + j) + """];\n"""

    if itr_str == """ """:
        raise ValueError('Dimension of shape of functions is not compatible')

    # generate the dictionary
    Dict = {}
    product = Function(fs)
    Dict.update({"product": (product, WRITE)})
    Dict.update({"f1": (f1, READ)})
    Dict.update({"f2": (f2, READ)})

    # generate the kernel
    hp_kernel = """
    for (int k=0;k<f1.dofs;k++){
        """ + itr_str + """
    }
    """

    # implement kernel
    par_loop(hp_kernel, dx, Dict)

    # evaluate kernel
    product.dat.data[:] += 0

    return product


def ConstructSparseMatrix(matrix):

    """ Constructs a sparse matrix out of a matrix by using diag searches """

    sparse_matrix = scp.lil_matrix(np.shape(matrix))

    # check that it's a square matrix. do we need this condition? revisit!
    assert np.shape(matrix)[0] == np.shape(matrix)[1]

    # iterate over diags and skip if all elements are 0
    for i in range(np.shape(matrix)[0]):
        # could change this to be not just block diags (and change just one element
        # if only one element is non-zero. revisit!
        if np.any(matrix.diagonal((i + 1) - np.shape(matrix)[0]) != 0):
            sparse_matrix.setdiag(matrix.diagonal((i + 1) - np.shape(matrix)[0]),
                                  (i + 1) - np.shape(matrix)[0])

    for i in range(np.shape(matrix)[1] - 1):
        if np.any(matrix.diagonal(i + 1) != 0):
            sparse_matrix.setdiag(matrix.diagonal(i + 1),
                                  i + 1)

    return sparse_matrix


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
