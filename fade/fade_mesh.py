""" meshes for FADE """

from __future__ import division

from __future__ import absolute_import

from firedrake import *


def FadeMesh(mesh_type, *args):

    """ Creates a :class:`Mesh` of type `mesh_type` for the FADE package. This creates a hierarchy
        under the returned finest mesh, up until the coarsest available mesh.

        Example:
            FadeMesh(UnitSquareMesh, 16, 16) returns
            mesh = UnitSquareMesh(16, 16) which is part of a hierarchy where the coarsest mesh
            corresponds to a `UnitSquareMesh` with 2 cells in each geometric dimension.

        :arg mesh_type: The string of the type of :class:`Mesh` Function one requires
        :type mesh_type: str

        :arg *args: Arguments for evaluating `mesh_type`
        :type *args: tuple

    """

    if isinstance(mesh_type, str) is False:
        raise ValueError("`mesh_type` must be a string")

    try:
        eval(mesh_type)
    except NameError:
        raise ValueError("`mesh_type` must be a type of Mesh function")

    try:
        mesh = eval(mesh_type)(*args)
    except TypeError:
        raise TypeError("Mesh of `mesh_type` does not have arguments *args")

    # recover the dimension of mesh
    d = mesh.geometric_dimension()

    # recover the number of cells of mesh
    if d == 1:
        nx = args[0]
    elif d == 2:
        nx = args[0]
        ny = args[1]
    else:
        raise TypeError("Mesh of `mesh_type` cannot be any more than 2-dimensional")

    # coarsen number of cells until not whole numbers
    l = 0
    converge = 0
    while converge == 0:

        if d == 1:

            nnx = nx / 2.0

            if (nnx % 1 == 0) and (nnx != 0):
                l += 1
                nx = nnx
            else:
                converge = 1

        if d == 2:

            nnx = nx / 2.0
            nny = ny / 2.0

            if (nnx % 1 == 0) and (nny % 1 == 0) and (nnx != 0) and (nny != 0):
                l += 1
                nx = nnx
                ny = nny
            else:
                converge = 1

    # define coarsest mesh
    new_args = []
    for i in range(len(args)):
        new_args.append(args[i])
    if d == 1:
        new_args[0] = nx
    if d == 2:
        new_args[0] = nx
        new_args[1] = ny
    new_args = tuple(new_args)
    coarsest_mesh = eval(mesh_type)(*new_args)

    # create mesh hierarchy
    mesh_hierarchy = MeshHierarchy(coarsest_mesh, l)

    # return finest mesh
    return mesh_hierarchy[-1]
