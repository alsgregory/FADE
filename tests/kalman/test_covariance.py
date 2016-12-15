from __future__ import division

from __future__ import absolute_import

from firedrake import *

from firedrake_da import *
from firedrake_da.kalman import *

import numpy as np


def test_covariance_zero_var():

    mesh = UnitIntervalMesh(1)

    n = 10

    v = VectorFunctionSpace(mesh, 'DG', 0, dim=n)

    ensemble_f = Function(v)
    for i in range(n):
        ensemble_f.dat.data[:, i] = 1.0

    # zero variance
    cov = covariance(ensemble_f)

    assert np.max(np.abs(cov.dat.data)) == 0


if __name__ == "__main__":
    import os
    import pytest
    pytest.main(os.path.abspath(__file__))
