from __future__ import division

from __future__ import absolute_import

from firedrake import *

from firedrake_da import *

import numpy as np


def test_covariance_zero_var():
    
    mesh = UnitIntervalMesh(1)
    
    v = FunctionSpace(mesh, 'DG', 0)
    
    ensemble = []
    n = 10
    for i in range(n):
        f = Function(v)
        ensemble.append(f)
    
    # zero variance
    cov, in_ensemble_dat = Covariance(ensemble)
    
    assert np.max(np.abs(cov)) == 0


def test_covariance_projection():
    
    mesh = UnitIntervalMesh(1)
    
    v = FunctionSpace(mesh, 'DG', 1)
    fs_to_project_to = FunctionSpace(mesh, 'DG', 2)
    
    ensemble = []
    n = 10
    for i in range(n):
        f = Function(v)
        ensemble.append(f)
    
    # different observation spaces
    options = [v, fs_to_project_to]
    degs = [1, 2]
    i = 0
    for fs in options:
        cov, in_ensemble_dat = Covariance(ensemble, fs)
        assert np.shape(cov) == tuple([(degs[i] + 1), (degs[i] + 1)])
        i += 1


if __name__ == "__main__":
    import os
    import pytest
    pytest.main(os.path.abspath(__file__))
