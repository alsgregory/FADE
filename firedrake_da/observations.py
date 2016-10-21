""" class taking in observations (given by coordinates) and using them in firedrake_dataassimilation """

from __future__ import division

from __future__ import absolute_import

from firedrake import *

import numpy as np

from firedrake_da.utils import *


class Observations(object):

    def __init__(self, observation_coords, observations, mesh):

        """

            :arg observation_coords: tuple / list defining the coords of observations
            :type observation_coords: tuple / list

            :arg observations: tuple / list of observation state values
            :type observations: tuple / list

            :arg mesh: mesh that observations need to be interpolated on
            :type mesh: :class:`Mesh`

        """

        self.mesh = mesh
        self.observation_coords = observation_coords
        self.observations = observations

        # find cells of mesh that contain observation
        self.cells = PointToCell(self.observation_coords, self.mesh)

        # define the DG0 function space
        self.cell_fs = FunctionSpace(self.mesh, 'DG', 0)
        self.cell_func = Function(self.cell_fs)

        # find nodes that contain observations
        self.nodes = CellToNode(self.cells, self.cell_fs)

        # length of observations
        self.ny = len(self.observations)

        super(Observations, self).__init__()

    def difference(self, func, p=2):

        """ finds p-norm difference between a function and observations for each cell in a DG0 fs

            :arg func: the :class:`Function` to find the difference between observations and it
            :type func: :class:`Function`

            :arg p: degree of p-norm
            :type p: arg:

        """

        out_func = Function(func.function_space())

        # find the DG0 diffs aggregated over all observations in a cell
        cell_diff = Function(self.cell_fs)

        # project func to DG0
        in_func = Function(self.cell_fs)
        in_func.project(func)

        # next use kernel to iterate over adding up cell averages for observation
        for i in range(self.ny):
            cell_diff.dat.data[self.nodes[i].astype(int)] += (self.observations[i] - in_func.dat.data[self.nodes[i].astype(int)]) ** p

        # project back to (depends on what post processing one wants surely!)
        out_func.project(cell_diff)

        return out_func
