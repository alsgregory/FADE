from __future__ import division

from __future__ import absolute_import

from firedrake import *

from firedrake_da import *

import numpy as np

import pytest


def test_invariant_onedtransform():

    a = np.reshape(np.linspace(0, 9, 10), ((1, 10)))

    wa = np.ones(10) * 0.1
    wb = np.ones(10) * 0.1

    new_ensemble = np.copy(a)
    transformed_ensemble = np.reshape(onedtransform(wa, wb, a[0, :]),
                                      ((1, 10)))

    assert np.sum(np.abs(transformed_ensemble - new_ensemble)) == 0.0


def test_invariant_transform():

    a = np.reshape(np.linspace(0, 9, 10), ((1, 10)))
    b = np.reshape(np.linspace(0, 9, 10), ((1, 10)))

    wa = np.ones(10) * 0.1
    wb = np.ones(10) * 0.1

    new_ensemble = np.copy(a)
    Cost = CostMatrix(a, b)
    transformed_ensemble = transform(a, b, wa, wb, Cost)

    assert np.sum(np.abs(transformed_ensemble - new_ensemble)) == 0.0


def test_cost_matrix():

    a = np.ones((1, 20))
    b = np.ones((1, 20))

    Cost = CostMatrix(a, b)

    assert np.shape(Cost) == tuple([20, 20])
    assert np.sum(np.abs(Cost)) == 0.0


if __name__ == "__main__":
    import os
    pytest.main(os.path.abspath(__file__))
