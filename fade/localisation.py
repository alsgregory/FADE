""" localisation of basis coeffs of functions in Firedrake via coarsening """

from __future__ import division

from __future__ import absolute_import

from firedrake import *

from firedrake.mg.utils import get_level


def CoarseningLocalisation(f, r_loc):

    """ Creates a :class:`Function` that localises another :class:`Function` using injection.
        Needs fs to be part of a hierarchy in mg firedrake (even in standard etpf).

        :arg f: The :class:`Function` or list of :class:`Function`s for localising
        :type f: :class:`Function: / list of :class:`Function`s

        :arg r_loc: The radius of localisation
        :type r_loc: int

    """

    # check if f is an ensemble of functions
    if (isinstance(f, list) is True) or (isinstance(f, tuple) is True):
        # function space
        fs = f[0].function_space()

    else:
        # function space
        fs = f.function_space()

    # check for hierarchy and if not in one, cap r_loc at 0
    hierarchy, lvl = get_level(fs.mesh())
    if lvl is None:
        r_loc = 0

    # if no localisation return same
    if r_loc == 0:
        return f

    # check r_loc is within hierarchy
    if r_loc < 0 or (lvl - r_loc) < 0:
        raise ValueError('Radius of localisation needs to be from 0 to max level of hierarchy.')

    # make scale factor for number of finer subcells to coarse cell
    scale = 2 ** (fs.mesh().geometric_dimension() * r_loc)

    # create function to inject to
    fc = Function(FunctionSpace(hierarchy[lvl - r_loc], fs.ufl_element()))

    # inject down
    if (isinstance(f, list) is True) or (isinstance(f, tuple) is True):
        for i in range(len(f)):
            inject(f[i], fc)

            # reset f and prolong back again
            f[i].assign(0)
            prolong(fc, f[i])
            f[i].assign(f[i] * scale)
    else:
        inject(f, fc)

        # reset f and prolong back again
        f.assign(0)
        prolong(fc, f)
        f.assign(f * scale)

    return f
