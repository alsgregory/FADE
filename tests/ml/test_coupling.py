from __future__ import division

from __future__ import absolute_import

from firedrake import *

from firedrake_da import *
from firedrake_da.ml import *

import numpy as np


def test_coupling_mean_preserving():

    mesh = UnitIntervalMesh(1)

    mesh_hierarchy = MeshHierarchy(mesh, 3)
    r_loc = 2
    r_loc_func = 2

    coord = tuple([np.array([0.5])])

    obs = tuple([1])

    # build ensemble
    fsc = FunctionSpace(mesh_hierarchy[-2], 'DG', 0)
    fsf = FunctionSpace(mesh_hierarchy[-1], 'DG', 0)
    ensemble_c = [Function(fsc), Function(fsc)]
    ensemble_f = [Function(fsf), Function(fsf)]
    ensemble_c[0].assign(1.0)
    ensemble_c[1].assign(1.0)
    ensemble_f[0].assign(1.0)
    ensemble_f[1].assign(1.0)

    # compute weights - should be even
    sigma = 0.1
    weights_c = weight_update(ensemble_c, coord, obs, sigma, r_loc)
    weights_f = weight_update(ensemble_f, coord, obs, sigma, r_loc)

    # compute ensemble transform - should be 1.0's
    new_ensemble_c, new_ensemble_f = seamless_coupling_update(ensemble_c, ensemble_f,
                                                              weights_c, weights_f,
                                                              r_loc_func)

    assert np.max(new_ensemble_c[0].dat.data[:] - 1.0) < 1e-5
    assert np.max(new_ensemble_c[1].dat.data[:] - 1.0) < 1e-5

    assert np.max(new_ensemble_f[0].dat.data[:] - 1.0) < 1e-5
    assert np.max(new_ensemble_f[1].dat.data[:] - 1.0) < 1e-5


if __name__ == "__main__":
    import os
    import pytest
    pytest.main(os.path.abspath(__file__))
