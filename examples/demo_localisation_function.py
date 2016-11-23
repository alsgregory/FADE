""" demo of localisation function """

from __future__ import absolute_import

from __future__ import division

from firedrake import *
from firedrake_da import *


# create mesh, function space

mesh = UnitSquareMesh(10, 10)

V = FunctionSpace(mesh, 'DG', 0)

# local point

local_point = 20

# create function to write to files

ffs = Function(V)

# localise over different radius

r_locs = [0, 1, 2]

for i in range(3):

    r_loc = r_locs[i]

    loc = Localisation(V, r_loc, local_point)
    ffs.assign(loc)
    ffs.rename("radius " + str(r_loc) + " localisation")
    ffsFile = File("localisation_" + str(r_loc) + "function.pvd")
    ffsFile.write(ffs)
