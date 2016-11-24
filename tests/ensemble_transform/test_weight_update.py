from __future__ import division

from __future__ import absolute_import

from firedrake import *

from firedrake_da import *
from firedrake_da.ensemble_transform import *

import numpy as np


def test_weight_update_perfect_observation():

    mesh = UnitIntervalMesh(1)

    mesh_hierarchy = MeshHierarchy(mesh, 3)
    r_loc = 2

    coord = tuple([np.array([0.5])])

    obs = tuple([1])

    # build ensemble
    fs = FunctionSpace(mesh_hierarchy[-1], 'DG', 0)
    ensemble = [Function(fs), Function(fs)]
    ensemble[0].assign(1.0)
    ensemble[1].assign(1.0)

    # compute weights - should be even
    sigma = 0.1
    weights = weight_update(ensemble, coord, obs, sigma, r_loc)

    # check weights are even
    assert np.max(np.abs(weights[0].dat.data[:] - 0.5)) < 1e-5
    assert np.max(np.abs(weights[1].dat.data[:] - 0.5)) < 1e-5


if __name__ == "__main__":
    import os
    import pytest
    pytest.main(os.path.abspath(__file__))
