# Tutorial of a weight update of an ensemble of functions in FADE
# ===============================================================
#
# This is a tutorial on how to update the importance weights,
# given as functions :math:`w \in V`, where :math:`V` is a function
# space, of an ensemble of functions :math:`f \in V`.

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

n = 50

ensemble = []
for i in range(n):
    f = Function(V).assign(np.random.normal(0, 1))
    ensemble.append(f)

ensemble = tuple(ensemble)

# To find the posterior importance weights of each function in the ensemble
# we are required to specify the prior weights, which can simply be even.
#
# These weights are also functions :math:`w \in V` and should also be part
# of a list or tuple.

weights = []
for i in range(n):
    w = Function(V).assign(1.0 / n)
    weights.append(w)

weights = tuple(weights)

# Let's make some data to assimilate!
#
# In FADE, observations are taken from reference functions via a list or
# tuple of coordinates in the domain. These observations are also stored
# as lists or tuples. First let's generate a reference function.

ref = Function(V).assign(np.random.normal(0, 1))

# For the benefit of this tutorial, each observation will be taken at a random
# coordinate in the domain and can be perturbed by some normal measurement
# with variance `R`. There will be `ny` observations.

R = 0.005
ny = 50
coords = []
obs = []
for i in range(ny):
    coords.append(np.array([np.random.uniform(0, 1),
                            np.random.uniform(0, 1)]))
    obs.append(ref.at(coords[i]) +
               np.random.normal(0, np.sqrt(R)))

# Observations need to be put into an object that represents an observation
# operator. This allows observations to be projected on to ensemble space.
#
# One can initialize the object and then is required to update it with
# new observations and coordinates every time a new set becomes available.

observation_operator = Observations(V)
observation_operator.update_observation_operator(coords, obs)

# So that we can compare the error away from the reference function, let's
# take the prior mean of the ensemble.

prior_mean = Function(V)
for i in range(n):
    prior_mean.dat.data[:] += np.multiply(weights[i].dat.data[:],
                                          ensemble[i].dat.data[:])

# We are all ready to compute the posterior importance weight functions
# of the ensemble member functions. Calling this method overwrites the prior
# weight functions.

weight_update(ensemble, weights, observation_operator, R)

# Compute the posterior mean in the same way as we did with the prior weights.

posterior_mean = Function(V)
for i in range(n):
    posterior_mean.dat.data[:] += np.multiply(weights[i].dat.data[:],
                                              ensemble[i].dat.data[:])

# Finally we compare the errors of the prior mean and posterior mean away from
# the reference solution.
#
# The latter is smaller given that we have weighted the
# ensemble members around the observations taken from the reference solution.

print 'prior mean error from ref: ', norm(assemble(prior_mean - ref))
print 'posterior mean error from ref: ', norm(assemble(posterior_mean - ref))
