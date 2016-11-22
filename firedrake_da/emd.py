""" Earth Mover's Distance / Ensemble Transform functions (Algorithm courtesy of O. Pele and M. Werman (2009)) """

from __future__ import division

from __future__ import absolute_import

import numpy as np

from pulp import *

import re


def onedtransform(wa, wb, pa):

    w1 = np.copy(wa)
    w2 = np.copy(wb)

    N = len(w1)

    ensembletransform = np.zeros(N)

    i = N
    j = N

    while i * j >= 1:

        if w1[i - 1] < w2[j - 1]:
            ensembletransform[j - 1] += (w1[i - 1] * (1.0 / wb[j - 1])) * pa[i - 1]
            w2[j - 1] = w2[j - 1] - w1[i - 1]
            i = i - 1

        else:
            ensembletransform[j - 1] += (w2[j - 1] * (1.0 / wb[j - 1])) * pa[i - 1]
            w1[i - 1] = w1[i - 1] - w2[j - 1]
            j = j - 1

    return ensembletransform


def transform(feature1, feature2, w1, w2, Cost):

    H = feature1.shape[1]
    I = feature2.shape[1]
    D = feature1.shape[0]

    wa = np.copy(w1)
    wb = np.copy(w2)

    distances = Cost

    # Set variables for EMD calculations
    variablesList = []
    for i in range(H):
        tempList = []
        for j in range(I):
            tempList.append(LpVariable("x" + str(i) + " " + str(j), lowBound=0))
        variablesList.append(tempList)
    problem = LpProblem("EMD", LpMinimize)

    # objective function
    constraint = []
    objectiveFunction = []
    for i in range(H):
        for j in range(I):
            objectiveFunction.append(variablesList[i][j] * distances[i][j])
            constraint.append(variablesList[i][j])
    problem += lpSum(objectiveFunction)
    tempMin = min(sum(wa), sum(wb))
    problem += lpSum(constraint) == tempMin

    # constraints
    for i in range(H):
        constraint1 = [variablesList[i][j] for j in range(I)]
        problem += lpSum(constraint1) == wa[i]
    for j in range(I):
        constraint2 = [variablesList[i][j] for i in range(H)]
        problem += lpSum(constraint2) == wb[j]

    # solve
    problem.writeLP("EMD.lp")
    problem.solve(GLPK_CMD(msg=False))
    OptimalMatrix = np.zeros((H, I))
    new_feature1 = np.zeros((D, H))
    for variable in problem.variables():

        if str(variable) is not '__dummy':
            [i, j] = re.findall('\d+', str(variable))
            ii = int(i)
            jj = int(j)
            OptimalMatrix[ii, jj] = variable.varValue
            new_feature1[:, jj] += feature1[:, ii] * (1.0 / wb[jj]) * OptimalMatrix[ii, jj]

    return new_feature1


def CostMatrix(a, b):

    dim, N = np.shape(a)

    for i in range(dim):

        if i == 0:
            Cost = (np.repeat(np.square(a[i, :])[:, np.newaxis], N, axis=1) +
                    np.repeat(np.square(b[i, :])[np.newaxis, :], N, axis=0) -
                    (2 * np.dot(np.reshape(a[i, :], ((N, 1))), np.reshape(b[i, :], ((1, N))))))

        else:
            Cost += (np.repeat(np.square(a[i, :])[:, np.newaxis], N, axis=1) +
                     np.repeat(np.square(b[i, :])[np.newaxis, :], N, axis=0) -
                     (2 * np.dot(np.reshape(a[i, :], ((N, 1))), np.reshape(b[i, :], ((1, N))))))

    return Cost
