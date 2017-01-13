# Tutorial of an ensemble transform particle filter step in FADE
# ==============================================================
#
# This is a tutorial on how to compute an ensemble transform
# update, that takes the place of a random resampling step
# in the ensemble transform particle filter (Reich [2011]),
# with an ensemble of functions :math:`f \in V`, where :math:`V`
# is a function space.
#
# The ensemble is updated according to each function's importance
# weight, which is also a function :math:`w \in V`. For this
# tutorial, we will assume that the posterior importance weight
# functions have already been computed (and thus give them
# explicitly). For a tutorial on how to actually compute them
# given observations go to `<tutorial_weight_update.py>`__.


from __future__ import absolute_import

from __future__ import division

from fade import *

import numpy as np


# First of all we require a mesh, that has to be part of a hierarchy
# of meshes to allow for the localisation that is used in FADE.

mesh = UnitSquareMesh(2, 2)
L = 1
mesh_hierarchy = MeshHierarchy(mesh, L)

# This creates a mesh hierarchy of length `L`.
#
# Now let's build a function space on the finest level of the mesh
# hierarchy.

V = FunctionSpace(mesh_hierarchy[-1], 'DG', 0)

# We are ready to build an ensemble of functions on that space, of size
# `n`. Ensembles of functions can be written as lists or tuples for FADE.
#
# Here, in this tutorial, one function has an assigned value of 0.0, and
# the other 1.0.

n = 2

ensemble = []
for i in range(n):
    f = Function(V).assign(i)
    ensemble.append(f)

ensemble = tuple(ensemble)

# Just like with the ensembles of functions, we will now define an
# ensemble of pre-defined posterior weights. As in
# `<tutorial_weight_update.py>`__, importance weights are given as
# functions that are a part of a list or tuple.
#
# Here, in this tutorial, the first function has normalized weight
# of 0.25 for all basis coefficients, and the other 0.75 for all
# basis coefficients.

weights = []
weights.append(Function(V).assign(0.25))
weights.append(Function(V).assign(0.75))

weights = tuple(weights)

# So that we can check that the ensemble mean is preserved after the
# transformation, let's compute it.

ensemble_mean = Function(V)
for i in range(n):
    ensemble_mean.dat.data[:] += np.multiply(ensemble[i].dat.data[:],
                                             weights[i].dat.data[:])

# We can now transform the ensemble, to an evenly weighted one, that
# preserves the ensemble mean function.
#
# For localisation, we will use total localisation, given by `r_loc=0`
# which means that all basis coefficients get transformed independently
# of one another.

r_loc = 0
ensemble_transform_update(ensemble, weights, r_loc)

# Now our ensemble is updated, and we can clarify that the ensemble mean
# has been preserved. By calling this method, the weights are also reset
# to even weights.

new_ensemble_mean = Function(V)
for i in range(n):
    new_ensemble_mean.dat.data[:] += np.multiply(ensemble[i].dat.data[:],
                                                 weights[i].dat.data[:])

print 'error between ensemble and transformed ensemble means: '
print norm(assemble(new_ensemble_mean - ensemble_mean))
