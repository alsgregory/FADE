from __future__ import division

from __future__ import absolute_import

from firedrake import *

from firedrake_da import *
from firedrake_da.kalman import *

import numpy as np


def test_kalman_perfect_obs():

    mesh = UnitIntervalMesh(1)

    v = FunctionSpace(mesh, 'DG', 0)

    ensemble = []
    n = 10
    for i in range(n):
        f = Function(v).assign(1.0)
        ensemble.append(f)

    # observations
    obs = np.array([1.0])
    coords = np.array([np.array([0.5])])

    # new ensemble
    observation_operator = Observations(v)
    sigma = 1.0
    new_ensemble = kalman_update(ensemble, observation_operator, coords, obs, sigma)

    for i in range(n):
        assert np.max(np.abs(new_ensemble[i].dat.data[:] - 1.0)) < 1e-5


if __name__ == "__main__":
    import os
    import pytest
    pytest.main(os.path.abspath(__file__))
