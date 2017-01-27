""" a kernel implementation of a (localised) seamless update from a coupling between
two weighted ensembles (coarse and fine) to two coupled evenly weighted ensembles """

from __future__ import absolute_import

from __future__ import division

from firedrake import *
from firedrake.mg.utils import get_level
from fade import *
from fade.ml import *
from fade.emd.emd_kernel import *

import numpy as np

from pyop2.profiling import timed_stage


def seamless_coupling_update(ensemble_1, ensemble_2, weights_1, weights_2, r_loc_c=0, r_loc_f=0):

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

        Optional Arguments:

        :arg r_loc_c: Radius of coarsening localisation for the coarse cost functions. Default: 0
        :type r_loc_c: int

        :arg r_loc_f: Radius of coarsening localisation for the fine cost functions. Default: 0
        :type r_loc_f: int

    """

    if len(ensemble_1) < 1 or len(ensemble_2) < 1:
        raise ValueError('ensembles cannot be indexed')
    if len(weights_1) < 1 or len(weights_2) < 1:
        raise ValueError('weights cannot be indexed')

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

    n = len(ensemble_c)
    if n is not len(ensemble_f):
        raise ValueError('Both ensembles need to be of the same length')

    # function spaces of both ensembles and create vector function space
    fsc = ensemble_c[0].function_space()
    fsf = ensemble_f[0].function_space()
    fam = fsc.ufl_element().family()
    deg = fsc.ufl_element().degree()

    assert fam == fsf.ufl_element().family()
    assert deg == fsf.ufl_element().degree()

    vfsc = VectorFunctionSpace(mesh_1, fam, deg, dim=n)
    vfsf = VectorFunctionSpace(mesh_2, fam, deg, dim=n)

    # check that weights have same length
    assert len(weights_c) == n
    assert len(weights_f) == n

    # check that weights add up to one
    with timed_stage("Checking weights are normalized"):
        cc = Function(fsc)
        cf = Function(fsf)
        for k in range(n):
            cc.dat.data[:] += weights_c[k].dat.data[:]
            cf.dat.data[:] += weights_f[k].dat.data[:]

        if np.max(np.abs(cc.dat.data[:] - 1)) > 1e-3 or np.max(np.abs(cf.dat.data[:] - 1)) > 1e-3:
            raise ValueError('Coarse weights dont add up to 1')

    # preallocate new / intermediate ensembles and assign basis coeffs to new vector function
    with timed_stage("Preallocating functions"):
        ensemble_c_f = Function(vfsc)
        ensemble_f_f = Function(vfsf)
        new_ensemble_c_f = Function(vfsc)
        new_ensemble_f_f = Function(vfsf)
        int_ensemble_c_f = Function(vfsc)
        if n == 1:
            ensemble_c_f.dat.data[:] = ensemble_c[0].dat.data[:]
            ensemble_f_f.dat.data[:] = ensemble_f[0].dat.data[:]
        else:
            for i in range(n):
                ensemble_c_f.dat.data[:, i] = ensemble_c[i].dat.data[:]
                ensemble_f_f.dat.data[:, i] = ensemble_f[i].dat.data[:]

    # define even weights
    with timed_stage("Preallocating functions"):
        even_weights_c = []
        even_weights_f = []
        fc = Function(fsc).assign(1.0 / n)
        ff = Function(fsf).assign(1.0 / n)
        for k in range(n):
            even_weights_c.append(fc)
            even_weights_f.append(ff)

    # inject fine weights and ensembles down to coarse mesh
    with timed_stage("Injecting finer ensemble / weights down to coarse mesh"):
        inj_ensemble_f_f = Function(vfsc)
        inj_weights_f = []
        totals = Function(fsc)
        for i in range(n):
            g = Function(fsc)
            inject(weights_f[i], g)
            inj_weights_f.append(g)
            totals.dat.data[:] += inj_weights_f[i].dat.data[:]
        inject(ensemble_f_f, inj_ensemble_f_f)

    # re-normalize injected fine weights
    for i in range(n):
        inj_weights_f[i].dat.data[:] = np.divide(inj_weights_f[i].dat.data[:], totals.dat.data[:])

    with timed_stage("Coupling between weighted coarse and fine ensembles"):
        kernel_transform(ensemble_c_f, inj_ensemble_f_f, weights_c,
                         inj_weights_f, int_ensemble_c_f, r_loc_c)

    with timed_stage("Finer ensemble transform"):
        kernel_transform(ensemble_f_f, ensemble_f_f, weights_f,
                         even_weights_f, new_ensemble_f_f, r_loc_f)

    with timed_stage("Coupling weighted intermediate ensemble and transformed finer ensemble"):
        # inject transformed finer ensemble
        inj_new_ensemble_f_f = Function(vfsc)
        inject(new_ensemble_f_f, inj_new_ensemble_f_f)

        kernel_transform(int_ensemble_c_f, inj_new_ensemble_f_f,
                         inj_weights_f, even_weights_c, new_ensemble_c_f, r_loc_c)

    # check that components have the same mean
    with timed_stage("Checking posterior mean consistency"):
        mc = Function(fsc)
        mf = Function(fsf)
        for k in range(n):
            mc.dat.data[:] += np.multiply(ensemble_c[k].dat.data[:], weights_c[k].dat.data[:])
            mf.dat.data[:] += np.multiply(ensemble_f[k].dat.data[:], weights_f[k].dat.data[:])

    # override ensembles
    if n == 1:
        ensemble_c[0].dat.data[:] = new_ensemble_c_f.dat.data[:]
        ensemble_f[0].dat.data[:] = new_ensemble_f_f.dat.data[:]
    else:
        for i in range(n):
            ensemble_c[i].dat.data[:] = new_ensemble_c_f.dat.data[:, i]
            ensemble_f[i].dat.data[:] = new_ensemble_f_f.dat.data[:, i]

    # reset weights
    for i in range(n):
        weights_c[i].assign(1.0 / n)
        weights_f[i].assign(1.0 / n)

    # check that components have the same mean
    with timed_stage("Checking posterior mean consistency"):
        mnc = Function(fsc)
        mnf = Function(fsf)
        for k in range(n):
            mnc.dat.data[:] += np.multiply(ensemble_c[k].dat.data[:], weights_c[k].dat.data[:])
            mnf.dat.data[:] += np.multiply(ensemble_f[k].dat.data[:], weights_f[k].dat.data[:])

        assert np.max(np.abs(mnc.dat.data[:] - mc.dat.data[:])) < 1e-5
        assert np.max(np.abs(mnf.dat.data[:] - mf.dat.data[:])) < 1e-5

    return ensemble_c, ensemble_f
