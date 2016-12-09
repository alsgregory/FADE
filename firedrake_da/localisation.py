""" localisation of basis coeffs of functions in Firedrake via coarsening """

from __future__ import division

from __future__ import absolute_import

from firedrake import *

from firedrake.mg.utils import get_level


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

    try:
        inject(f, fc)
    except ValueError:
        raise Exception("Unfortunately this localisation is only compatible with Tensor or " +
                        "standard FunctionSpaces")

    # prolong back again
    f_new = Function(fs)
    prolong(fc, f_new)

    return f_new
