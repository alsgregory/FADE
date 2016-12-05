from __future__ import division

from __future__ import absolute_import

from firedrake import *

from firedrake_da import *

import numpy as np

import pytest


def test_invariant_transform():

    n = 10

    mesh = UnitIntervalMesh(1)

    V = FunctionSpace(mesh, 'DG', 0)

    r_loc = 0

    weights = []
    ensemble = []
    for i in range(n):
        f = Function(V)
        f.assign(i)
        g = Function(V)
        g.assign(1.0 / n)
        weights.append(g)
        ensemble.append(f)

    new_ensemble = ensemble_transform_update(ensemble, weights, r_loc)

    for i in range(n):
        assert np.max(np.abs(new_ensemble[i].dat.data - ensemble[i].dat.data)) < 1e-5


if __name__ == "__main__":
    import os
    pytest.main(os.path.abspath(__file__))
