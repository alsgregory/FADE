""" Functions generating an Earth Movers Distance kernel for the ensemble_transform_update and
seamless_coupling_update """

from __future__ import absolute_import

from firedrake import *

from firedrake_da.localisation import *

import numpy as np

import os

from pyop2.profiling import timed_stage

from ufl.classes import IndexSum, MultiIndex, Product


# NOTE: Only need to generate these kernels (well just EMD_KERNEL AS THE REST WILL FOLLOW) once at the start of the assimilation, and just use this for whole process. Aslong as the labels for dictionary stay the same!


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
            Dict.update({member_label:(ensemble[i], READ)})
        if access == "WRITE":
            Dict.update({member_label:(ensemble[i], WRITE)})
        if access == "RW":
            Dict.update({member_label:(ensemble[i], RW)})

    return Dict


def get_emd_kernel(n):

    """ Generates an emd kernel for functions in Firedrake

        :arg n: size of ensembles
        :type n: int

    """

    cost_str = get_cost_str(n)

    # get feature string
    feature_str = get_feature_str(n)

    # get output string
    output_str = get_output_str(n)

    emd_kernel = """ int n=%(size_n)s;
    for (int k=0;k<input_f_0.dofs;k++){
        float matrix_identity[%(size_n)s][%(size_n)s];
        float _COST[%(size_n)s][%(size_n)s];
        float _M[%(size_n)s][%(size_n)s];
        for (int i=0;i<n;i++){
            for (int j=0;j<n;j++){
                _M[i][j]=0.0;
            }
        }
        float dist(feature_t *F1, feature_t *F2) { return _COST[*F1][*F2]; }
        """ + cost_str + """
        int         flowSize=%(size_n)s+%(size_n)s-1;
        """ + feature_str + """
        signature_t s1 = { %(size_n)s, f1, w1},
                    s2 = { %(size_n)s, f2, w2};
        flow_t      flow[flowSize];
        float e;
        float F[flowSize];
        int ip[flowSize];
        int jp[flowSize];
        e = emd(&s1, &s2, dist, flow, &flowSize, F, ip, jp);
        for (int i=0;i<flowSize;i++){
            int xpi=ip[i];
            int xpj=jp[i];
            _M[xpi][xpj]=0.0;
            _M[xpi][xpj]+=F[i];
        }
        for (int i=0;i<n;i++){
            for (int j=0;j<n;j++){
                matrix_identity[i][j]=0.0;
                matrix_identity[i][j]+=_M[i][j]*(1.0/w2[j]);
            }
        }
    """ + output_str + """
    }
    """
    emd_kernel = emd_kernel % {"size_n": n}

    return emd_kernel


def get_cost_str(n):

    cost_str = " "
    for i in range(n):
        for j in range(n):
            cost_str += "_COST[" + str(i) + "][" + str(j) + "]=cost_tensor[k][" + str((i * n) + j) + "];\n"

    return cost_str


def get_output_str(n):

    output_str = ""
    for i in range(n):
        output_str += "output_f_" + str(i) + "[k][0] = "
        for j in range(n):
            output_str += "(matrix_identity[" + str(j) + "][" + str(i) + "]*input_f_" + str(j) + "[k][0])"
            if j < n - 1:
                output_str += "+"
            else:
                output_str += ";\n"

    return output_str


def get_feature_str(n):

    feature_str = " "
    feature_str += "float w1[%(size_n)s] = {"
    for i in range(n):
        if i < n - 1:
            feature_str += "input_weight_" + str(i) + "[k][0],"
        else:
            feature_str += "input_weight_" + str(i) + "[k][0]"
    feature_str += "};\n float w2[%(size_n)s] = {"
    for i in range(n):
        if i < n - 1:
            feature_str += "input_weight2_" + str(i) + "[k][0],"
        else:
            feature_str += "input_weight2_" + str(i) + "[k][0]"
    feature_str += "};\n feature_t f1[%(size_n)s] = {"
    for i in range(n):
        if i < n - 1:
            feature_str += str(i) + ","
        else:
            feature_str += str(i)
    feature_str += "};\n feature_t f2[%(size_n)s] = {"
    for i in range(n):
        if i < n - 1:
            feature_str += str(i) + ","
        else:
            feature_str += str(i)
    feature_str += "};"

    return feature_str


def generate_localised_cost_tensor(ensemble, ensemble2, r_loc):

    """ Computes a (localised) cost tensor function for the squared different between two ensembles

        :arg ensemble: The first ensemble of functions to couple
        :type ensemble: list / tuple

        :arg ensemble2: The second ensemble of functions to couple
        :type ensemble2: list / tuple

        :arg r_loc: Radius of coarsening localisation for the cost tensor
        :type r_loc: int 

    """

    n = len(ensemble)
    assert len(ensemble2) == n

    mesh = ensemble[0].function_space().mesh()

    # get deg and fam of function space
    deg = ensemble[0].function_space().ufl_element().degree()
    fam = ensemble[0].function_space().ufl_element().family()

    # assert that its the same in ensemble2
    assert deg == ensemble2[0].function_space().ufl_element().degree()
    assert fam == ensemble2[0].function_space().ufl_element().family()

    # if in CG project to DG
    if fam == 'CG' or fam == 'Lagrange':
        with timed_stage("Projecting from CG to DG"):
            fs = FunctionSpace(mesh, 'DG', deg)
            for i in range(n):
                f = Function(fs)
                project(ensemble[i], f)
                ensemble[i] = f
                f = Function(fs)
                project(ensemble2[i], f)
                ensemble2[i] = f

    # make tensor function space and vector function space
    tfs = TensorFunctionSpace(mesh, 'DG', deg, (n, n))
    tfsp = TensorFunctionSpace(mesh, fam, deg, (n, n))
    vfs = VectorFunctionSpace(mesh, 'DG', deg, dim=n)

    # make test function and ensemble functions
    phi = TestFunction(tfs)
    ensemble_f = Function(vfs)
    ensemble2_f = Function(vfs)
    if n == 1:
        ensemble_f.dat.data[:] = ensemble[0].dat.data[:]
        ensemble2_f.dat.data[:] = ensemble2[0].dat.data[:]

    else:
        for i in range(n):
            ensemble_f.dat.data[:, i] = ensemble[i].dat.data
            ensemble2_f.dat.data[:, i] = ensemble2[i].dat.data

    # compute unlocalised DG cost function tensor
    nc = ensemble[0].function_space().dof_dset.size
    i, j = indices(2)
    with timed_stage("Creating the cost tensor"):
        f = ((IndexSum(IndexSum(Product(nc * phi[i, j], Product(ensemble_f[i], ensemble_f[i])),
                                MultiIndex((i,))), MultiIndex((j,))) * dx) +
             (IndexSum(IndexSum(Product(nc * phi[i, j], Product(ensemble2_f[j], ensemble2_f[j])),
                                MultiIndex((i,))), MultiIndex((j,))) * dx) -
             (IndexSum(IndexSum(2 * nc * Product(phi[i, j], Product(ensemble_f[i], ensemble2_f[j])),
                                MultiIndex((i,))), MultiIndex((j,))) * dx))

        cost_tensor = assemble(f)

    # assign to functions from tensor
    fs = ensemble[0].function_space()
    cost_funcs = []
    if n == 1:
        cost_funcs.append([])
        f = Function(fs)
        f.dat.data[:] = cost_tensor.dat.data[:]
        cost_funcs[0].append(f)

    else:
        for i in range(n):
            cost_funcs.append([])
            for j in range(n):
                f = Function(fs)
                f.dat.data[:] = cost_tensor.dat.data[:, i, j]
                cost_funcs[i].append(f)

    # carry out coarsening localisation
    with timed_stage("Coarsening localisation"):
        for i in range(n):
            for j in range(n):
                cost_funcs[i][j] = CoarseningLocalisation(cost_funcs[i][j], r_loc)

    # put basis coefficients back to tensor and project back to old function space
    if n == 1:
        cost_tensor.dat.data[:] = cost_funcs[0][0].dat.data[:]
        if fam == 'CG' or fam == 'Lagrange':
            with timed_stage("Projecting from DG to CG"):
                cost_tensor_p = Function(tfsp)
                cost_tensor_p.project(cost_tensor)

    else:
        for i in range(n):
            for j in range(n):
                cost_tensor.dat.data[:, i, j] = cost_funcs[i][j].dat.data[:]
        if fam == 'CG' or fam == 'Lagrange':
            with timed_stage("Projecting from DG to CG"):
                cost_tensor_p = Function(tfsp)
                cost_tensor_p.project(cost_tensor)

    return cost_tensor


def kernel_transform(ensemble, ensemble2, weights, weights2, out_func, r_loc):

    """ Carries out a coupling transform using kernels

        :arg ensemble: The first ensemble of functions to couple
        :type ensemble: list / tuple

        :arg ensemble2: The second ensemble of functions to couple
        :type ensemble2: list / tuple

        :arg weights: The importance weights of first ensemble of functions to couple
        :type weights: list / tuple

        :arg weights2: The importance weights of second ensemble of functions to couple
        :type weights2: list / tuple

        :arg out_func: A list of functions to output the transform ensemble to
        :type out_func: list / tuple

        :arg r_loc: Radius of coarsening localisation for the cost function
        :type r_loc: int

    """

    # generate emd kernel
    emd_k = get_emd_kernel(len(ensemble))

    # generate cost funcs
    cost_tensor = generate_localised_cost_tensor(ensemble, ensemble2, r_loc)

    # make dictionary
    Dict = {}

    # update with functions
    Dict = update_Dictionary(Dict, ensemble, "input_f_", "READ")

    # update with functions
    Dict = update_Dictionary(Dict, ensemble2, "input_f2_", "READ")

    # update with weights
    Dict = update_Dictionary(Dict, weights, "input_weight_", "READ")

    # update with weights
    Dict = update_Dictionary(Dict, weights2, "input_weight2_", "READ")

    # update with output functions
    Dict = update_Dictionary(Dict, out_func, "output_f_", "WRITE")

    # update with cost tensor
    Dict.update({"cost_tensor":(cost_tensor, READ)})

    # current working directory
    p = os.getcwd()

    # key options for par_loop
    ldargs=["-L" + p + "/firedrake_da/EMD", "-Wl,-rpath," + p + "/firedrake_da/EMD", "-lemd"]
    headers=["#include <emd.h>"]
    include_dirs=[p + "/firedrake_da/EMD"]

    # carry out par_loop -> out_func gets overwritten
    with timed_stage("Ensemble transform"):
        par_loop(emd_k, dx, Dict, ldargs=ldargs, headers=headers,
                 include_dirs=include_dirs)
