""" Functions generating an Earth Movers Distance kernel for the ensemble_transform_update and
seamless_coupling_update.

The C code used in the emd_kernels is courtesy of:
Y. Rubner, C. Tomasi, and L. J. Guibas. The earth movers distance as a metric for image retrieval. IJCV, 2000. """

from __future__ import division

from __future__ import absolute_import

from firedrake import *

from firedrake_da.localisation import *
from firedrake_da.utils import *

import numpy as np

import os

from pyop2.profiling import timed_stage

from ufl.classes import IndexSum, MultiIndex, Product


class emd_kernel_generation(object):

    def __init__(self, n):

        """ Generates the kernel needed for the emd transform using strings
            for different components

            :arg n: Size of ensemble
            :type n: int

        """

        self.n = n

        # get cost string
        cost_str = self.__get_cost_str()

        # get feature string
        feature_str = self.__get_feature_str()

        # get output string
        output_str = self.__get_output_str()

        # generate emd kernel
        self.emd_kernel = """ int n=%(size_n)s;
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

        # replace n in kernel with constant
        self.emd_kernel = self.emd_kernel % {"size_n": self.n}

        super(emd_kernel_generation, self).__init__()

    def __get_cost_str(self):

        cost_str = " "
        for i in range(self.n):
            for j in range(self.n):
                cost_str += "_COST[" + str(i) + "][" + str(j) + "]=cost_tensor[k][" + str((i * self.n) + j) + "];\n"

        return cost_str

    def __get_output_str(self):

        output_str = ""
        for i in range(self.n):
            output_str += "output_f_" + str(i) + "[k][0] = "
            for j in range(self.n):
                output_str += "(matrix_identity[" + str(j) + "][" + str(i) + "]*input_f_" + str(j) + "[k][0])"
                if j < self.n - 1:
                    output_str += "+"
                else:
                    output_str += ";\n"

        return output_str

    def __get_feature_str(self):

        feature_str = " "
        feature_str += "float w1[%(size_n)s] = {"
        for i in range(self.n):
            if i < self.n - 1:
                feature_str += "input_weight_" + str(i) + "[k][0],"
            else:
                feature_str += "input_weight_" + str(i) + "[k][0]"
        feature_str += "};\n float w2[%(size_n)s] = {"
        for i in range(self.n):
            if i < self.n - 1:
                feature_str += "input_weight2_" + str(i) + "[k][0],"
            else:
                feature_str += "input_weight2_" + str(i) + "[k][0]"
        feature_str += "};\n feature_t f1[%(size_n)s] = {"
        for i in range(self.n):
            if i < self.n - 1:
                feature_str += str(i) + ","
            else:
                feature_str += str(i)
        feature_str += "};\n feature_t f2[%(size_n)s] = {"
        for i in range(self.n):
            if i < self.n - 1:
                feature_str += str(i) + ","
            else:
                feature_str += str(i)
        feature_str += "};"

        return feature_str


def get_cost_func_kernel(n):

    cost_func_str = " "
    for i in range(n):
        for j in range(n):
            cost_func_str += "cost_tensor[k][" + str((i * n) + j) + "]=(input_f_" + str(i) + "[k][0]-input_f2_" + str(j) + "[k][0])*(input_f_" + str(i) + "[k][0]-input_f2_" + str(j) + "[k][0]);\n"

    cost_func_kernel = """
    for (int k=0;k<input_f_0.dofs;k++){
    """ + cost_func_str + """
    }
    """

    return cost_func_kernel


def generate_localised_cost_tensor(ensemble, ensemble2, r_loc, option="kernel"):

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

    # get degree and family of function space
    deg = ensemble[0].function_space().ufl_element().degree()
    fam = ensemble[0].function_space().ufl_element().family()

    # assert that its the same in ensemble2
    assert deg == ensemble2[0].function_space().ufl_element().degree()
    assert fam == ensemble2[0].function_space().ufl_element().family()

    # make tensor function space and vector function space
    tfs = TensorFunctionSpace(mesh, fam, deg, (n, n))
    vfs = VectorFunctionSpace(mesh, fam, deg, dim=n)

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

    # compute unlocalised cost function tensor
    if option == "assembly":

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

    if option == "kernel":

        with timed_stage("Creating the cost tensor"):
            cost_tensor_kernel = get_cost_func_kernel(n)
            cost_tensor = Function(tfs)
            Dict = {}
            Dict = update_Dictionary(Dict, ensemble, "input_f_", "READ")
            Dict = update_Dictionary(Dict, ensemble2, "input_f2_", "READ")
            Dict.update({"cost_tensor": (cost_tensor, WRITE)})
            par_loop(cost_tensor_kernel, dx, Dict)

    # assign basis coefficients from cost tensor to functions and localise them
    fs = ensemble[0].function_space()
    cost_funcs = []
    if n == 1:
        f = Function(fs)
        f.dat.data[:] = cost_tensor.dat.data[:]
        cost_funcs.append([f])
    else:
        for i in range(n):
            cost_funcs.append([])
            for j in range(n):
                f = Function(fs)
                f.dat.data[:] = cost_tensor.dat.data[:, i, j]
                cost_funcs[i].append(f)

    # carry out coarsening localisation and put back into tensor
    with timed_stage("Coarsening localisation"):
        if n == 1:
            x = CoarseningLocalisation(cost_funcs[0][0], r_loc)
            cost_tensor.dat.data[:] = x.dat.data[:]
        else:
            for i in range(n):
                for j in range(n):
                    x = CoarseningLocalisation(cost_funcs[i][j], r_loc)
                    cost_tensor.dat.data[:, i, j] = x.dat.data[:]

    return cost_tensor


def kernel_transform(ensemble, ensemble2, weights, weights2, out_func, r_loc, option="kernel"):

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

        :arg r_loc: Radius of coarsening localisation for the cost tensor
        :type r_loc: int

    """

    # generate emd kernel
    emd_k = emd_kernel_generation(len(ensemble))

    # generate cost funcs
    cost_tensor = generate_localised_cost_tensor(ensemble, ensemble2, r_loc, option)

    # make dictionary
    Dict = {}

    # update with first functions
    Dict = update_Dictionary(Dict, ensemble, "input_f_", "READ")

    # update with second functions
    Dict = update_Dictionary(Dict, ensemble2, "input_f2_", "READ")

    # update with first weights
    Dict = update_Dictionary(Dict, weights, "input_weight_", "READ")

    # update with second weights
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
        par_loop(emd_k.emd_kernel, dx, Dict, ldargs=ldargs, headers=headers,
                 include_dirs=include_dirs)
