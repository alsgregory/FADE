""" localisation of basis coeffs of functions in Firedrake via coarsening """

from __future__ import division

from __future__ import absolute_import

from firedrake import *

from firedrake.mg.utils import get_level

from pyop2.profiling import timed_stage

import numpy as np


def CoarseningLocalisation(f, r_loc):

    """ Creates a :class:`Function` that localises another :class:`Function` using injection.
        Needs fs to be part of a hierarchy in mg firedrake (even in standard etpf).

        :arg f: The :class:`Function` for localising
        :type f: :class:`Function:

        :arg r_loc: The radius of localisation
        :type r_loc: int

    """

    # if no localisation return same
    if r_loc == 0:
        return f

    # function space and hierarchy check
    fs = f.function_space()

    hierarchy, lvl = get_level(f.function_space().mesh())
    assert lvl is not None

    # check r_loc is within hierarchy
    if r_loc < 0 or (lvl - r_loc) < 0:
        raise ValueError('radius of localisation needs to be from 0 to max level of hierarchy')

    # inject down
    fc = Function(FunctionSpace(hierarchy[lvl - r_loc], fs.ufl_element()))
    inject(f, fc)

    # reset f and prolong back again
    f.assign(0)
    prolong(fc, f)

    return f


def CovarianceLocalisation(vfs, r_loc):

    """ Creates a localisation vector :class:`Function` of exponentially decaying scaling
    coefficients for the Hadamard product of a vector :class:`Function` representing
    the covariance of an ensemble of :class:`Function`s.

        :arg fs: The :class:`VectorFunctionSpace` for the basis coeffs
        :type fs: :class:`VectorFunctionSpace:

        :arg r_loc: The radius of localisation
        :type r_loc: int

    """

    # Check that it's a vector function space
    vshape = vfs.ufl_element().value_shape()
    assert len(vshape) == 1

    # Define vfs's for localisation
    dim = vshape[0]
    dg_vfs = VectorFunctionSpace(vfs.mesh(), 'DG', 0, dim=dim)
    cg_vfs = VectorFunctionSpace(vfs.mesh(), 'CG', 1, dim=dim)

    # Generate a Function in vfs, dg_vfs and cg_vfs
    fvfs = Function(vfs)
    fdg_vfs = Function(dg_vfs)
    fcg_vfs = Function(cg_vfs)

    # Define cell max to vertex kernel
    with timed_stage("Defining kernel for localisation"):

        cellvertexmax_str = """ """
        vfunc = """vertq[i]["""
        for j in range(dim):
            cellvertexmax_str += vfunc + str(j) + """]=fmax(""" + vfunc + str(j) + """],cell[0][""" + str(j) + """]);\n"""

        cellmax2vertex_kernel = """
        for(int i=0;i<vertq.dofs;i++){
            """ + cellvertexmax_str + """
        }
        """

    # Check square matrix
    if len(np.shape(fvfs.dat.data)) == 2:
        assert np.shape(fvfs.dat.data)[0] == np.shape(fvfs.dat.data)[1]

    # Place ones in diagonal in the index of basis coeffs
    fvfs.dat.data[:] = np.diag(np.ones(dim))

    # If no localisation return diagonal matrix
    if r_loc == 0:
        return fvfs

    with timed_stage("Projections for covariance localisation"):
        # Projector from cg to dg
        cg_to_dgProjector = Projector(fcg_vfs, fdg_vfs)

        # Project fvfs to fdg_vfs to start
        fdg_vfs.project(fvfs)

    # Renormalize
    fdg_vfs.dat.data[:] = fdg_vfs.dat.data[:] / np.max(fdg_vfs.dat.data[:])

    # Carry out iterative spreading of this diagonal
    with timed_stage("Iterations of vertex averaging for covariance localisation"):
        for i in range(r_loc):

            # Carry out cell max kernel projection to vertices
            par_loop(cellmax2vertex_kernel, dx, {"vertq": (fcg_vfs, RW),
                                                 "cell": (fdg_vfs, READ)})

            # Use projector to dg
            cg_to_dgProjector.project()

    # Project back to original vector function space
    with timed_stage("Projections for covariance localisation"):
        fvfs.project(fdg_vfs)

    # Normalize
    fvfs.dat.data[:] = fvfs.dat.data[:] / np.max(fvfs.dat.data[:])

    # Truncate at zero
    fvfs.dat.data[:] = np.multiply(fvfs.dat.data[:], (fvfs.dat.data[:] >= 0))

    # Check that no value exceeds 1 or goes below 0
    assert np.max(fvfs.dat.data[:]) <= 1.0
    assert np.min(fvfs.dat.data[:]) >= 0.0

    return fvfs
