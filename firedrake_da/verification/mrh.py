""" multidimensional rank histogram for an ensemble of Firedrake functions given observation
operator """

from __future__ import division

from __future__ import absolute_import

from firedrake import *

import numpy as np

import matplotlib.pyplot as plot


class rank_histogram(object):

    def __init__(self, N):

        """ Can compute the multidimensional rank histogram of an ensemble of Firedrake functions
            using observations given by coordinates. NB: All ensemble members must stay same
            inde in the ensemble with all observations. Cannot do this between resampling /
            transform assimilation steps.

            :arg N: Ensemble size
            :type N: int

        """

        self.N = N

        # define rank list
        self.ranks = []

        super(rank_histogram, self).__init__()

    def __choose_uniform_rank(self, a, b):

        # check whole numbers
        assert a % 1 == 0
        assert b % 1 == 0

        flt = np.random.uniform(a, b + 1)
        int_rank = np.floor(flt)

        return int_rank

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

        # number of coordinate observations - proxy for dimensions
        ny = len(observations)

        # preallocate a ensemble of state values at coordinates
        state_ensemble = np.zeros((ny, self.N + 1))

        # place observations in
        state_ensemble[:, 0] = observations

        for i in range(ny):

            for j in range(self.N):
                # calculate an ensemble of scalar state values
                state_ensemble[i, j + 1] = ensemble[j].at(observation_coords[i])

        # compute pre-ranks
        rho = np.zeros(self.N + 1)
        for i in range(self.N + 1):
            rho[i] = np.sum(np.prod(np.reshape(state_ensemble[:, i], ((ny, 1))) >
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
