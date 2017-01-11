""" multidimensional rank histogram for an ensemble of Firedrake functions given observation
operator """

from __future__ import division

from __future__ import absolute_import

from firedrake import *

from firedrake_da.utils import *

import numpy as np

import matplotlib.pyplot as plot


class rank_histogram(object):

    def __init__(self, function_space, N):

        """ Can compute the multidimensional rank histogram of an ensemble of Firedrake functions
            using observations given by coordinates. NB: All ensemble members must stay same
            inde in the ensemble with all observations. Cannot do this between resampling /
            transform assimilation steps.

            :arg function_space: The :class:`FunctionSpace` of the ensemble :class:`Function`s
            :type function_space: :class:`FunctionSpace`

            :arg N: Ensemble size
            :type N: int

        """

        self.N = N

        self.mesh = function_space.mesh()

        # define function space and dg0 function of ensemble
        self.function_space = function_space
        self.dg0_function_space = FunctionSpace(self.mesh,
                                                self.function_space.ufl_element().family(),
                                                self.function_space.ufl_element().degree())
        self.dg0_function = Function(self.dg0_function_space)

        self.normalizing_function = Function(self.dg0_function_space)

        # make ensembles and dg0 ensembles
        self.in_ensemble = []
        self.in_dg0_ensemble = []
        self.inProjectors = []
        for i in range(self.N):
            self.in_ensemble.append(Function(self.function_space))
            self.in_dg0_ensemble.append(Function(self.dg0_function_space))
            self.inProjectors.append(Projector(self.in_ensemble[i], self.in_dg0_ensemble[i]))

        # define rank list
        self.ranks = []

        super(rank_histogram, self).__init__()

    def __choose_uniform_rank(self, a, b):

        # check whole numbers
        assert a % 1 == 0
        assert b % 1 == 0

        rank = a + 1 + np.where(np.random.multinomial(1, (np.ones(b) /
                                                          b)) == 1)[0][0]

        return rank

    def compute_rank(self, ensemble, observation_coords, observations):

        """

            :arg ensemble: list of :class:`Function`s in the ensemble
            :type ensemble: tuple / list

            :arg observation_coords: tuple / list defining the coords of observations
            :type observation_coords: tuple / list

            :arg observations: tuple / list of observation state values
            :type observations: tuple / list

        """

        if len(ensemble) < 1:
            raise ValueError('ensemble cannot be indexed')

        assert len(ensemble) is self.N

        # place ensemble into in_ensemble
        if ensemble[0].function_space() is not self.function_space:
            raise ValueError("ensemble needs to be on same function space as rank " +
                             "histogram class was initialized with")

        for i in range(self.N):
            self.in_ensemble[i].assign(ensemble[i])

        # number of coordinate observations - proxy for dimensions
        ny = len(observations)

        # find cells and nodes that contain observations
        cells = PointToCell(observation_coords, self.mesh)
        nodes = CellToNode(cells, self.dg0_function_space)
        unique_nodes = np.unique(nodes)

        # project ensemble to dg0 function space
        for i in range(self.N):
            self.in_ensemble[i].assign(ensemble[i])
            self.inProjectors[i].project()

        # preallocate a ensemble of state values at coordinates
        d = len(unique_nodes)
        state_ensemble = np.zeros((d, self.N + 1))

        self.normalizing_function.assign(0)
        self.dg0_function.assign(0)
        for i in range(ny):

            # place aggregated observations onto dg0 function space
            self.dg0_function.dat.data[nodes[i].astype(int)] += observations[i]
            self.normalizing_function.dat.data[nodes[i].astype(int)] += 1.0

        # normalize and extract to array of cells with observations in
        observedCells = np.divide(self.dg0_function.dat.data[unique_nodes.astype(int)],
                                  self.normalizing_function.dat.data[unique_nodes.astype(int)])

        # place observations into state_ensemble
        state_ensemble[:, 0] = observedCells

        for j in range(self.N):
            # calculate an ensemble of scalar state values
            state_ensemble[:, j + 1] = self.in_dg0_ensemble[j].dat.data[unique_nodes.astype(int)]

        # compute pre-ranks
        rho = np.zeros(self.N + 1)
        for i in range(self.N + 1):
            rho[i] = np.sum(np.prod(np.reshape(state_ensemble[:, i], ((d, 1))) >
                                    state_ensemble, axis=0))

        # make start / end points of s to pick uniformly from
        s_start = np.sum(rho < rho[0])
        s_end = np.sum(rho == rho[0])

        # uniform pick of intermediate rank
        self.ranks.append(self.__choose_uniform_rank(s_start, s_end) / (self.N + 1))

    def plot_histogram(self):

        # define bins
        bins = np.linspace(0, 1, self.N + 2)

        # plot histogram
        n, bins, patches = plot.hist(self.ranks, bins=bins, normed=1,
                                     facecolor='green', alpha=0.75)
        plot.xlabel('rank of observation')
        plot.ylabel('normalised frequency')
        plot.axis([0, 1, 0, 1e-1 + np.max(n)])
        plot.show()

    """ Iterative and Indexing functions """

    def __len__(self):
        """ Return the length of the rank array """
        return len(ranks)

    def __iter__(self):
        """ Iterate over the ranks """
        for en in self.ranks:
            yield en

    def __getitem__(self, idx):
        """ Return a rank

            :arg idx: The index of the rank to return

        """
        return self.ranks[idx]
