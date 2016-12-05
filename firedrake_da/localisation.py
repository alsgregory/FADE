""" localisation of basis coeffs of functions in Firedrake """

from __future__ import division

from __future__ import absolute_import

from firedrake import *

import numpy as np

from firedrake.mg.utils import get_level

from pyop2.profiling import timed_stage


def Localisation(fs, r_loc, index, rate=1.0):

    """ Creates a localisation :class:`Function` of scaling coefficients
    for the basis coeffs for a :class:`Function` in a certain
    :class:`FunctionSpace`. NB: Degenrate function!

        :arg fs: The :class:`FunctionSpace` for the basis coeffs
        :type fs: :class:`FunctionSpace:

        :arg r_loc: The radius of localisation (how many iterative 'jumps' of basis coeffs in loc)
        :type r_loc: int

        :arg index: index of basis coefficient
        :type index: int

        :arg rate: the scale parameter which adjusts the shape of the localisation
        :type rate: float

    """

    # Define coarsening localisation kernels

    with timed_stage("Defining kernels for localisation"):
        cell2vertex_kernel = """
        for(int i=0;i<vertq.dofs;i++){
            vertq[i][0]=fmax(vertq[i][0],cell[0][0]);
        }
        """

        vertex2cellaverage_kernel = """ float scale=0; const rate=%(RATE)s;
        for(int i=0;i<vertq.dofs;i++){
            new_cell[0][0]+=pow(vertq[i][0],rate);
            scale=scale+1.0;
        }
        new_cell[0][0]=new_cell[0][0]/(scale);
        """
        vertex2cellaverage_kernel = vertex2cellaverage_kernel % {"RATE": rate}

    # project fs to DG
    new_fs = FunctionSpace(fs.mesh(), 'DG', 0)

    # Generate a CG function space
    cg_fs = FunctionSpace(fs.mesh(), 'CG', 1)
    cg_f = Function(cg_fs)

    # Generate a Function in fs
    ffs = Function(fs)

    # Place a one in the index of basis coeffs
    ffs.dat.data[index] += 1.0

    # if no localisation return same
    if r_loc == 0:
        return ffs

    # Generate a Function in new_fs with 1.0 in the cell centre
    with timed_stage("Projections for localisation"):
        f = Function(new_fs).project(ffs)

    # Renormalize
    f.dat.data[:] = f.dat.data[:] / np.max(f.dat.data[:])

    # Carry out iterative spreading of this index of this for r_loc times
    fn = Function(new_fs)
    fn.assign(f)
    dg_f = Function(new_fs)

    # if r_loc = 0.0, f stays as it is
    with timed_stage("Iterations of vertex averaging for localisation functions"):
        for i in range(r_loc):

            cg_f.assign(0)
            dg_f.assign(0)

            par_loop(cell2vertex_kernel, dx, {"vertq": (cg_f, RW),
                                              "cell": (fn, READ)})
            par_loop(vertex2cellaverage_kernel, dx, {"new_cell": (dg_f, RW),
                                                     "vertq": (cg_f, READ)})

            fn.assign(dg_f)

    # Project back to original function space
    with timed_stage("Projections for localisation"):
        ffs.project(fn)

    # Normalize
    ffs.dat.data[:] = ffs.dat.data[:] / np.max(ffs.dat.data[:])

    # Truncate at zero
    ffs.dat.data[:] = np.multiply(ffs.dat.data[:], (ffs.dat.data[:] >= 0))

    # check that no value exceeds 1 or goes below 0
    assert np.max(ffs.dat.data[:]) <= 1.0
    assert np.min(ffs.dat.data[:]) >= 0.0

    return ffs


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
    d = fs.ufl_element().degree()
    family = fs.ufl_element().family()
    fc = Function(FunctionSpace(hierarchy[lvl - r_loc], family, d))

    inject(f, fc)

    # prolong back again
    f_new = Function(fs)
    prolong(fc, f_new)

    return f_new
