""" seamless update (with localisation) from a coupling between two weighted ensembles
(coarse and fine) to two coupled evenly weighted ensembles. NB: Two ensembles have
to belong to the same hierarchy """

from __future__ import absolute_import

from __future__ import division

from firedrake import *
from firedrake.mg.utils import get_level
from firedrake_da import *
from firedrake_da.localisation import *
from firedrake_da.emd import *
from firedrake_da.ml import *

import numpy as np

"""" The 1D couplings are still done using emd here and not the cheap algorithm. To switch to this
needs the ability to sort both finer subcells that are in a coarse subcell. Thus we need to pick out
the indicies of the finer subcells and sort them according to the sorted coarse cell at the start
of each cheap coupling """


def seamless_coupling_update(ensemble_1, ensemble_2, weights_1, weights_2, r_loc_func):

    """ performs a seamless coupling ensemble transform update (with localisation) from a coupling
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

        :arg r_loc_func: radius of localisation function
        :type r_loc_func: int

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
    ncc = len(ensemble_c[0].dat.data)
    ncf = len(ensemble_f[0].dat.data)
    for i in range(ncc):

        c = 0
        for k in range(n):
            c += weights_c[k].dat.data[i]

        if np.abs(c - 1) > 1e-3:
            raise ValueError('Coarse weights dont add up to 1')
    for i in range(ncf):

        c = 0
        for k in range(n):
            c += weights_f[k].dat.data[i]

        if np.abs(c - 1) > 1e-3:
            raise ValueError('Fine weights dont add up to 1')

    # design localisation functions for coarse and fine meshes (NB: make this in script at
    # the start / class for it, rather than doing it at each assimilation step!)
    Cc = []
    Cf = []
    for i in range(ncc):
        Cc.append(Localisation(fsc, r_loc_func, i))
    for i in range(ncf):
        Cf.append(Localisation(fsf, r_loc_func, i))

    # preallocate new / intermediate ensembles
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

    # inject fine weights and ensembles down to coarse mesh
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

    # find particle and weights matrcies
    particles_c = np.zeros((ncc, n))
    w_c = np.zeros((ncc, n))
    particles_f = np.zeros((ncf, n))
    w_f = np.zeros((ncf, n))
    inj_particles_f = np.zeros((ncc, n))
    inj_w_f = np.zeros((ncc, n))
    for j in range(ncc):
        for k in range(n):
            particles_c[j, k] = ensemble_c[k].dat.data[j]
            w_c[j, k] = weights_c[k].dat.data[j]
            inj_particles_f[j, k] = inj_ensemble_f[k].dat.data[j]
            inj_w_f[j, k] = inj_weights_f[k].dat.data[j]
    for j in range(ncf):
        for k in range(n):
            particles_f[j, k] = ensemble_f[k].dat.data[j]
            w_f[j, k] = weights_f[k].dat.data[j]

    # re-normalize injected fine weights
    for i in range(n):
        inj_weights_f[i].dat.data[:] = np.divide(inj_weights_f[i].dat.data[:], totals)

    """ initial weighted coupling between coarse and fine """

    # for each coarse component carry out emd
    int_particles_c = np.zeros((ncc, n))
    for j in range(ncc):

        # design cost matrix, using localisation functions
        Cost = np.zeros((n, n))
        for i in range(ncc):
            pc = np.reshape(particles_c[i, :], ((1, n)))
            pf = np.reshape(inj_particles_f[i, :], ((1, n)))
            Cost += Cc[j].dat.data[i] * CostMatrix(pc, pf)

        # transform
        Pc = np.reshape(particles_c[j, :], ((1, n)))
        Pf = np.reshape(inj_particles_f[j, :], ((1, n)))

        ens = transform(Pc, Pf, w_c[j, :], inj_w_f[j, :], Cost)

        # into intermediate ensemble
        for k in range(n):
            int_ensemble_c[k].dat.data[j] = ens[0, k]
            int_particles_c[j, k] = ens[0, k]

    """ transform for fine ensemble """

    # for each fine component carry out emd
    for j in range(ncf):

        # design cost matrix, using localisation functions
        Cost = np.zeros((n, n))
        for i in range(ncf):
            pf = np.reshape(particles_f[i, :], ((1, n)))
            Cost += Cf[j].dat.data[i] * CostMatrix(pf, pf)

        # transform
        Pf = np.reshape(particles_f[j, :], ((1, n)))
        ens = transform(Pf, Pf, w_f[j, :], np.ones(n) * (1.0 / n), Cost)

        # into new ensemble
        for k in range(n):
            new_ensemble_f[k].dat.data[j] = ens[0, k]

    """ coupling between weighted intermediate ensemble and transformed finer ensemble """

    # inject transformed finer ensemble
    inj_new_ensemble_f = []
    for i in range(n):
        f = Function(fsc)
        inj_new_ensemble_f.append(f)
        inject(new_ensemble_f[i], inj_new_ensemble_f[i])

    # find particle matrices
    inj_new_particles_f = np.zeros((ncc, n))
    for j in range(ncc):
        for k in range(n):
            inj_new_particles_f[j, k] = inj_new_ensemble_f[k].dat.data[j]

    # for each coarse component carry out emd
    for j in range(ncc):

        # design cost matrix, using localisation functions
        Cost = np.zeros((n, n))
        for i in range(ncc):
            pc = np.reshape(int_particles_c[i, :], ((1, n)))
            pf = np.reshape(inj_new_particles_f[i, :], ((1, n)))
            Cost += Cc[j].dat.data[i] * CostMatrix(pc, pf)

        # transform
        Pc = np.reshape(int_particles_c[j, :], ((1, n)))
        Pf = np.reshape(inj_new_particles_f[j, :], ((1, n)))
        ens = transform(Pc, Pf, inj_w_f[j, :], np.ones(n) * (1.0 / n), Cost)

        # into new ensemble
        for k in range(n):
            new_ensemble_c[k].dat.data[j] = ens[0, k]

    # check that components have the same mean
    for j in range(ncc):
        mn = 0
        m = 0
        for k in range(n):
            mn += new_ensemble_c[k].dat.data[j] * (1.0 / n)
            m += ensemble_c[k].dat.data[j] * weights_c[k].dat.data[j]
        assert np.abs(mn - m) < 1e-5
    for j in range(ncf):
        mn = 0
        m = 0
        for k in range(n):
            mn += new_ensemble_f[k].dat.data[j] * (1.0 / n)
            m += ensemble_f[k].dat.data[j] * weights_f[k].dat.data[j]
        assert np.abs(mn - m) < 1e-5

    return new_ensemble_c, new_ensemble_f
