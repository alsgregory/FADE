""" demo of coarsening localisation function """

from __future__ import absolute_import

from __future__ import division

from firedrake import *
from fade import *


# create mesh, function space

mesh = UnitSquareMesh(6, 6)
mesh_hierarchy = MeshHierarchy(mesh, 2)

V = FunctionSpace(mesh_hierarchy[-1], 'DG', 0)

# local point

local_point = 20

# create function to write to files

ffs = Function(V)
ffs.dat.data[local_point] = 1.0

# localise over different radius

r_locs = [0, 1, 2]

for i in range(3):

    r_loc = r_locs[i]

    loc = CoarseningLocalisation(ffs, r_loc)
    ffs.assign(loc)
    ffs.rename("radius " + str(r_loc) + " localisation")
    ffsFile = File("localisation_" + str(r_loc) + "function.pvd")
    ffsFile.write(ffs)
