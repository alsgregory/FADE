Tutorials
=========

In this section, a number of tutorials will demonstrate how one can implement the features of FADE.
Python files containing these tutorials can be found by heading to the `/tutorials/` directory, listed in
:ref:`directory_overview`.

.. _tutorial_weight_update:

Tutorial: Importance Weight Updates
-----------------------------------

This is a tutorial on how to update the importance weights,
given as functions :math:`w \in V`, where :math:`V` is a function
space, of an ensemble of functions :math:`f \in V`.

.. code::
    
    from __future__ import absolute_import
    
    from __future__ import division
    
    from fade import *
    
    import numpy as np


First of all we require a mesh. For the type of localisation used in FADE
this mesh is required to be part of a hierarchy. Let's work on the finest
mesh in a hierarchy of length 2.

.. code::
    
    mesh_hierarchy = MeshHierarchy(UnitSquareMesh(1, 1), 1)
    mesh = mesh_hierarchy[-1]

Now let's build a function space on the mesh.

.. code::
    
    V = FunctionSpace(mesh, 'DG', 0)

We are ready to build an ensemble of functions on that space, of size
`n`. Ensembles of functions can be written as lists or tuples for FADE.

.. code::
    
    n = 50
    
    ensemble = []
    for i in range(n):
        f = Function(V).assign(np.random.normal(0, 1))
        ensemble.append(f)
    
    ensemble = tuple(ensemble)

To find the posterior importance weights of each function in the ensemble
we are required to specify the prior weights, which can simply be even.

These weights are also functions :math:`w \in V` and should also be part
of a list or tuple.

.. code::
    
    weights = []
    for i in range(n):
        w = Function(V).assign(1.0 / n)
        weights.append(w)
    
    weights = tuple(weights)

Let's make some data to assimilate!

In FADE, observations are taken from reference functions via a list or
tuple of coordinates in the domain. These observations are also stored
as lists or tuples. First let's generate a reference function.

.. code::
    
    ref = Function(V).assign(np.random.normal(0, 1))

For the benefit of this tutorial, each observation will be taken at a random
coordinate in the domain. The basis coefficients of the reference function will be
perturbed by some normally distributed measurement error
with variance `R` and observations will be taken from this new function.
There will be `ny` observations.

.. code::
    
    R = 0.005
    
    ref_per = Function(V)
    ref_per.dat.data[:] = ref.dat.data[:] + np.random.normal(0, np.sqrt(R),
                                                             len(ref.dat.data))
    
    ny = 50
    coords = []
    obs = []
    for i in range(ny):
        coords.append(np.array([np.random.uniform(0, 1),
                                np.random.uniform(0, 1)]))
        obs.append(ref_per.at(coords[i]))

Observations need to be put into an object that represents an observation
operator. This allows observations to be projected on to ensemble space.

One can initialize the object and then is required to update it with
new observations and coordinates every time a new set becomes available.

.. code::
    
    observation_operator = Observations(V, R)
    observation_operator.update_observation_operator(coords, obs)

So that we can compare the error away from the reference function, let's
take the prior mean of the ensemble.

.. code::
    
    prior_mean = Function(V)
    for i in range(n):
        prior_mean.dat.data[:] += np.multiply(weights[i].dat.data[:],
                                              ensemble[i].dat.data[:])

We are all ready to compute the posterior importance weight functions
of the ensemble member functions. Calling this method overwrites the prior
weight functions.

.. code::
    
    weight_update(ensemble, weights, observation_operator)

Compute the posterior mean in the same way as we did with the prior weights.

.. code::
    
    posterior_mean = Function(V)
    for i in range(n):
        posterior_mean.dat.data[:] += np.multiply(weights[i].dat.data[:],
                                                  ensemble[i].dat.data[:])

Finally we compare the errors of the prior mean and posterior mean away from
the reference solution.

The latter is smaller given that we have weighted the
ensemble members around the observations taken from the reference solution.
We shall now display these two errors to confirm this.

.. code::
    
    print 'prior mean error from ref: ', norm(assemble(prior_mean - ref))
    print 'posterior mean error from ref: ', norm(assemble(posterior_mean - ref))


Tutorial: An Ensemble Transform Particle Filter Step
----------------------------------------------------


This is a tutorial on how to compute an ensemble transform
update, that takes the place of a random resampling step
in the ensemble transform particle filter (Reich, 2011),
with an ensemble of functions :math:`f \in V`, where :math:`V`
is a function space.

The ensemble is updated according to each function's importance
weight, which is also a function :math:`w \in V`. For this
tutorial, we will assume that the posterior importance weight
functions have already been computed (and thus give them
explicitly). For a tutorial on how to actually compute them
given observations go to :ref:`tutorial_weight_update`.

.. code::
    
    from __future__ import absolute_import
    
    from __future__ import division
    
    from fade import *
    
    import numpy as np


First of all we require a mesh. For the type of localisation used in FADE
this mesh is required to be part of a hierarchy. Let's work on the finest
mesh in a hierarchy of length 2.

.. code::
    
    mesh_hierarchy = MeshHierarchy(UnitSquareMesh(1, 1), 1)
    mesh = mesh_hierarchy[-1]

Now let's build a function space on the mesh.

.. code::
    
    V = FunctionSpace(mesh, 'DG', 0)

We are ready to build an ensemble of functions on that space, of size
`n`. Ensembles of functions can be written as lists or tuples for FADE.

Here, in this tutorial, one function has an assigned value of 0, and
the other 1.

.. code::
    
    n = 2
    
    ensemble = []
    for i in range(n):
        f = Function(V).assign(i)
        ensemble.append(f)
    
    ensemble = tuple(ensemble)

Just like with the ensembles of functions, we will now define an
ensemble of pre-defined posterior weights. As in
:ref:`tutorial_weight_update`, importance weights are given as
functions that are a part of a list or tuple.

Here, in this tutorial, the first function has normalized weight
of 0.25 for all basis coefficients, and the other 0.75 for all
basis coefficients.

.. code::
    
    weights = []
    weights.append(Function(V).assign(0.25))
    weights.append(Function(V).assign(0.75))
    
    weights = tuple(weights)

So that we can check that the ensemble mean is preserved after the
transformation, let's compute it.

.. code::
    
    ensemble_mean = Function(V)
    for i in range(n):
        ensemble_mean.dat.data[:] += np.multiply(ensemble[i].dat.data[:],
                                                 weights[i].dat.data[:])

We can now transform the ensemble, to an evenly weighted one, that
preserves the ensemble mean function.

For localisation, we will use total localisation, given by `r_loc=0`
which means that all basis coefficients get transformed independently
of one another.

.. code::
    
    r_loc = 0
    ensemble_transform_update(ensemble, weights, r_loc)

Now our ensemble is updated, and we can clarify that the ensemble mean
has been preserved. By calling this method, the weights are also reset
to even weights. The error between the two ensemble means should be 0:

.. code::
    
    new_ensemble_mean = Function(V)
    for i in range(n):
        new_ensemble_mean.dat.data[:] += np.multiply(ensemble[i].dat.data[:],
                                                     weights[i].dat.data[:])
    
    print 'error between ensemble and transformed ensemble means: '
    print norm(assemble(new_ensemble_mean - ensemble_mean))


Tutorial: Multilevel Ensemble Transform Particle Filtering
----------------------------------------------------------

This is a demonstration of how to use FADE, alongside `Firedrake-mlmc <https://github.com/firedrakeproject/firedrake-mlmc>`_, to construct a multilevel filtering :ref:`estimator`
of a discretized field given partial observations of a reference function.
This tutorial follows one of the examples in the directory `examples/ml`.
