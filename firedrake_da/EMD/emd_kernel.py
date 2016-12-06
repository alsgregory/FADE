""" Functions generating an Earth Movers Distance kernel for the ensemble_transform_update and
seamless_coupling_update """

from __future__ import absolute_import

from firedrake import *

from firedrake_da.localisation import *

import numpy as np

import os

from pyop2.profiling import timed_stage


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


def get_emd_kernel(n, r_loc):

    """ Generates an emd kernel for functions in Firedrake

        :arg n: size of ensembles
        :type n: int

        :arg r_loc: Radius of coarsening localisation for the cost functions
        :type r_loc: int

    """

    cost_str = get_cost_str(n, r_loc)

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


def get_cost_str(n, r_loc):

    # let r_loc act like a switch parameter to check whether there is localisation
    if r_loc > 0:
        cost_str = " "
        for i in range(n):
            for j in range(n):
                cost_str += "_COST[" + str(i) + "][" + str(j) + "]=cost_f_" + str(i) + "_" + str(j) + "[k][0];\n"
    # WARNING: This doesn't look like it's improving funtime for r_loc = 0 cases, is there any benefit?
    if r_loc == 0:
        cost_str = " "
        for i in range(n):
            for j in range(n):
                cost_str += "_COST[" + str(i) + "][" + str(j) + "]=(input_f_" + str(i) + "[k][0]-input_f2_" + str(j) + "[k][0])*(input_f_" + str(i) + "[k][0]-input_f2_" + str(j) + "[k][0]);\n"

    return cost_str


def get_cost_func_kernel(n):

    cost_func_str = " "
    for i in range(n):
        for j in range(n):
            cost_func_str += "cost_f_" + str(i) + "_" + str(j) + "[k][0]=(input_f_" + str(i) + "[k][0]-input_f2_" + str(j) + "[k][0])*(input_f_" + str(i) + "[k][0]-input_f2_" + str(j) + "[k][0]);\n"

    cost_func_kernel = """
    for (int k=0;k<input_f_0.dofs;k++){
    """ + cost_func_str + """
    }
    """

    return cost_func_kernel


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


def generate_localised_cost_funcs(ensemble, ensemble2, cost_funcs, r_loc):

    # generate cost function kernel
    cost_func_kernel = get_cost_func_kernel(len(ensemble))

    # make cost func dictionary
    CostDict = {}

    # update cost func dictionary with functions
    CostDict = update_Dictionary(CostDict, ensemble, "input_f_", "READ")

    # update with functions
    CostDict = update_Dictionary(CostDict, ensemble2, "input_f2_", "READ")

    # update with cost functions
    for i in range(len(ensemble)):
        CostDict = update_Dictionary(CostDict, cost_funcs[i][:], "cost_f_" + str(i) + "_", "WRITE")

    # implement cost function generation
    with timed_stage("Generating cost functions to minimise"):
        par_loop(cost_func_kernel, dx, CostDict)

    # carry out coarsening localisation
    with timed_stage("Coarsening localisation"):
        for i in range(len(ensemble)):
            for j in range(len(ensemble)):
                cost_funcs[i][j] = CoarseningLocalisation(cost_funcs[i][j], r_loc)

    return cost_funcs


def kernel_transform(ensemble, ensemble2, weights, weights2, out_func, cost_funcs, r_loc):

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

        :arg cost_funcs: A list within a list of N x N preallocated functions of cost from i to j member
        :type cost_funcs: list / tuple

        :arg r_loc: Radius of coarsening localisation for the cost function
        :type r_loc: int

    """

    # generate emd kernel
    emd_k = get_emd_kernel(len(ensemble), r_loc)

    # generate cost funcs
    cost_funcs = generate_localised_cost_funcs(ensemble, ensemble2, cost_funcs, r_loc)

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

    # update with cost functions
    for i in range(len(ensemble)):
        Dict = update_Dictionary(Dict, cost_funcs[i][:], "cost_f_" + str(i) + "_", "READ")

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
