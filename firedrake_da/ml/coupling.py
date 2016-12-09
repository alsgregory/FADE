""" a kernel implementation of a (localised) seamless update from a coupling between
two weighted ensembles (coarse and fine) to two coupled evenly weighted ensembles """

from __future__ import absolute_import

from __future__ import division

from firedrake import *
from firedrake.mg.utils import get_level
from firedrake_da import *
from firedrake_da.ml import *
from firedrake_da.EMD.emd_kernel import *

import numpy as np

from pyop2.profiling import timed_stage


def seamless_coupling_update(ensemble_1, ensemble_2, weights_1, weights_2, r_loc_c, r_loc_f):

    """ performs a seamless coupling (localised) ensemble transform update from a coupling
        between two weighted ensembles (coarse and fine) into two evenly weighted ensembles.
        NB: The two ensembles have to belong to the same hierarchy

        :arg ensemble_1: list of :class:`Function`s in the coarse ensemble
        :type ensemble_1: tuple / list

        :arg ensemble_2: list of :class:`Function`s in the fine ensemble
        :type ensemble_2: tuple / list

        :arg weights_1: list of :class:`Function`s representing the importance weights for first
                        ensemble
        :type weights_1: tuple / list

        :arg weights_2: list of :class:`Function`s representing the importance weights for second
                        ensemble
        :type weights_2: tuple / list

        :arg r_loc_c: Radius of coarsening localisation for the coarse cost functions
        :type r_loc_c: int

        :arg r_loc_f: Radius of coarsening localisation for the fine cost functions
        :type r_loc_f: int

    """

    if len(ensemble_1) < 1 or len(ensemble_2) < 1:
        raise ValueError('ensembles cannot be indexed')

    # check that ensemble_1 and ensemble_2 have inputs in the same hierarchy
    mesh_1 = ensemble_1[0].function_space().mesh()
    mesh_2 = ensemble_2[0].function_space().mesh()
    hierarchy_1, lvl_1 = get_level(mesh_1)
    hierarchy_2, lvl_2 = get_level(mesh_2)
    if lvl_1 is None or lvl_2 is None:
        raise ValueError('Both ensembles to be coupled need to be on meshes part of same hierarchy')
    if hierarchy_1 is not hierarchy_2:
        raise ValueError('Both ensembles to be coupled need to be on meshes part of same hierarchy')

    # check if 1 is coarse and 2 is fine
    if lvl_1 < lvl_2:
        ensemble_c = ensemble_1
        weights_c = weights_1
        ensemble_f = ensemble_2
        weights_f = weights_2
    else:
        raise ValueError('Coarse ensemble needs to be the first ensemble, followed by a finer one')

    # function spaces of both ensembles
    fsc = ensemble_c[0].function_space()
    fsf = ensemble_f[0].function_space()

    n = len(ensemble_c)
    if n is not len(ensemble_f):
        raise ValueError('Both ensembles need to be of the same length')

    # check that weights have same length
    assert len(weights_c) == n
    assert len(weights_f) == n

    # check that weights add up to one
    with timed_stage("Checking weights are normalized"):
        ncc = len(ensemble_c[0].dat.data)
        ncf = len(ensemble_f[0].dat.data)
        cc = np.zeros(ncc)
        cf = np.zeros(ncf)
        for k in range(n):
            cc += weights_c[k].dat.data[:]
            cf += weights_f[k].dat.data[:]

        if np.max(np.abs(cc - 1)) > 1e-3 or np.max(np.abs(cf - 1)) > 1e-3:
            raise ValueError('Coarse weights dont add up to 1')

    # preallocate new / intermediate ensembles
    with timed_stage("Preallocating functions"):
        new_ensemble_c = []
        new_ensemble_f = []
        int_ensemble_c = []
        for i in range(n):
            f = Function(fsc)
            new_ensemble_c.append(f)
            g = Function(fsf)
            new_ensemble_f.append(g)
            h = Function(fsc)
            int_ensemble_c.append(h)

    # define even weights
    even_weights_c = []
    even_weights_f = []
    for k in range(n):
        fc = Function(fsc).assign(1.0 / n)
        even_weights_c.append(fc)
        ff = Function(fsf).assign(1.0 / n)
        even_weights_f.append(ff)

    # inject fine weights and ensembles down to coarse mesh
    with timed_stage("Injecting finer ensemble / weights down to coarse mesh"):
        inj_ensemble_f = []
        inj_weights_f = []
        totals = np.zeros(ncc)
        for i in range(n):
            f = Function(fsc)
            g = Function(fsc)
            inj_ensemble_f.append(f)
            inj_weights_f.append(g)
            inject(ensemble_f[i], inj_ensemble_f[i])
            inject(weights_f[i], inj_weights_f[i])
            totals += inj_weights_f[i].dat.data[:]

    # re-normalize injected fine weights
    for i in range(n):
        inj_weights_f[i].dat.data[:] = np.divide(inj_weights_f[i].dat.data[:], totals)

    with timed_stage("Coupling between weighted coarse and fine ensembles"):
        kernel_transform(ensemble_c, inj_ensemble_f, weights_c,
                         inj_weights_f, int_ensemble_c, r_loc_c)

    with timed_stage("Finer ensemble transform"):
        kernel_transform(ensemble_f, ensemble_f, weights_f,
                         even_weights_f, new_ensemble_f, r_loc_f)

    with timed_stage("Coupling weighted intermediate ensemble and transformed finer ensemble"):
        # inject transformed finer ensemble
        inj_new_ensemble_f = []
        for i in range(n):
            f = Function(fsc)
            inj_new_ensemble_f.append(f)
            inject(new_ensemble_f[i], inj_new_ensemble_f[i])

        kernel_transform(int_ensemble_c, inj_new_ensemble_f,
                         inj_weights_f, even_weights_c, new_ensemble_c, r_loc_c)

    # check that components have the same mean
    with timed_stage("Checking posterior mean consistency"):
        mnc = np.zeros(ncc)
        mc = np.zeros(ncc)
        mnf = np.zeros(ncf)
        mf = np.zeros(ncf)
        for k in range(n):
            mnc += new_ensemble_c[k].dat.data[:] * (1.0 / n)
            mc += np.multiply(ensemble_c[k].dat.data[:], weights_c[k].dat.data[:])
            mnf += new_ensemble_f[k].dat.data[:] * (1.0 / n)
            mf += np.multiply(ensemble_f[k].dat.data[:], weights_f[k].dat.data[:])

        assert np.max(np.abs(mnc - mc)) < 1e-5
        assert np.max(np.abs(mnf - mf)) < 1e-5

    return new_ensemble_c, new_ensemble_f
