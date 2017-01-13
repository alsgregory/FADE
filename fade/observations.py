""" class taking in observations (given by coordinates) and using them in FADE """

from __future__ import division

from __future__ import absolute_import

from firedrake import *

from fade.utils import *


class Observations(object):

    def __init__(self, fs, R):

        """ Initializes projections for ensemble members to be assimilated via an observation
            operator. Once updated with new set of coordinates and observations, one can
            find the p-norm differences between ensemble members and the observations space.

            :arg fs: :class:`FunctionSpace` of the functions in the ensemble to be assimilated
            :type fs: :class:`FunctionSpace`

            :arg R: This is the variance of the independent Gaussian measurement error for
                    each basis coefficient
            :type R: float

        """

        self.mesh = fs.mesh()
        self.fs = fs

        # measurement error variance
        self.R = R

        # check that the variance is above 0 (otherwise perfect observations)
        if self.R <= 0:
            raise ValueError('Variance of measurement error needs to be greater than 0')

        # define the DG0 function space
        self.cell_fs = FunctionSpace(self.mesh, 'DG', 0)
        self.cell_func = Function(self.cell_fs)

        # preallocate cells and nodes for observations as well as them themselves
        self.cells = None
        self.nodes = None
        self.observation_coords = None
        self.observations = None

        # projection operators
        self.in_func = Function(self.cell_fs)
        self.func = Function(self.fs)
        self.in_Project = Projector(self.func, self.in_func)

        self.cell_differences = Function(self.cell_fs)
        self.out_func = Function(self.fs)
        self.out_Project = Projector(self.cell_differences, self.out_func)

        self.observation_function = Function(self.cell_fs)
        self.observation_function_diff = Function(self.cell_fs)

        super(Observations, self).__init__()

    def update_observation_operator(self, observation_coords, observations):

        """ Updates observation operator (places coordinate observations on to mesh via DG0 cells)
            with new set of observations.

            :arg observation_coords: tuple / list defining the coords of observations
            :type observation_coords: tuple / list

            :arg observations: tuple / list of observation state values
            :type observations: tuple / list

        """

        # update observations and coordinates
        self.observation_coords = observation_coords
        self.observations = observations

        # find cells of mesh that contain observation
        self.cells = PointToCell(self.observation_coords, self.mesh)

        # find nodes that contain observations
        self.nodes = CellToNode(self.cells, self.cell_fs)

        # for each node, aggregate observations over cells
        ny = len(self.observations)

        # reset observation function
        self.observation_function.assign(0)

        norm_const = np.zeros(len(self.observation_function.dat.data))
        self.obs_num = np.ones(len(self.observation_function.dat.data))

        for i in range(ny):
            ind = self.nodes[i].astype(int)
            self.observation_function.dat.data[ind] += self.observations[i]

            # normalization constant
            norm_const[ind] += 1.0

            # state that that cell had an observation
            self.obs_num[ind] = 0.0

        norm_const = np.maximum(norm_const, np.ones(len(self.observation_function.dat.data)))
        self.observation_function.dat.data[:] = np.divide(self.observation_function.dat.data[:],
                                                          norm_const)

    def difference(self, func, p=2):

        """ finds p-norm difference between a function and observations
        for each cell in a DG0 fs

            :arg func: the :class:`Function` to find the p-norm difference between itself and
            observations
            :type func: :class:`Function`

            :arg p: degree of p-norm
            :type p: arg:

        """

        # check that observations have been initialized
        obs = [self.observation_coords, self.observations, self.cells, self.nodes]
        for ob in obs:
            if ob is None:
                raise ValueError("Observations havent been initialized. " +
                                 "Use Observations.update_observation_operator")

        # check that func belongs to function space that observation operator is initialized around
        if func.function_space() is not self.fs:
            raise ValueError("Function space of func is not same as one for observation operator " +
                             "initialization")

        # project func to DG0 using project and previously initialized in-function
        self.func.assign(func)
        self.in_Project.project()

        self.observation_function_diff.assign(0)

        # either observation difference or just the actual function
        self.observation_function_diff.dat.data[:] = (np.multiply(self.in_func.dat.data[:],
                                                                  self.obs_num) +
                                                      self.observation_function.dat.data[:])

        # next, find the squared distance between the function and aggregated observations
        self.cell_differences.assign(assemble((self.observation_function_diff -
                                               self.in_func) ** p))

        # project back to function space of ensemble
        self.out_Project.project()

        return self.out_func
