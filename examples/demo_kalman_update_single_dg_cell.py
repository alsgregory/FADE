from firedrake_da import *

import numpy as np

mesh = UnitSquareMesh(1, 1)

V = FunctionSpace(mesh, 'DG', 1)
fs = FunctionSpace(mesh, 'DG', 0)

coords = tuple([np.array([0.5, 0.5])])
obs = tuple([0.5])

ensemble = []

n = 10
sigma = .1

for i in range(n):
    f = Function(V).assign(np.random.normal(0.4, 0.2, 1)[0])
    ensemble.append(f)

cov = Covariance(ensemble, fs)
X = Kalman_update(ensemble, coords, obs, .1, fs)
