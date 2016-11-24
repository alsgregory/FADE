""" Stores localisation functions for a function space """

from __future__ import absolute_import

from __future__ import division

from firedrake import *
from firedrake_da import *
from firedrake_da.localisation import *


class LocalisationFunctions(object):

    """ Stores localisation functions for all components / basis coeffs in a given function space """

    def __init__(self, function_space, r_loc_func, rate=1.0):

        """

            :arg function_space: :class:`FunctionSpace` to find the localisation functions of
            :type function_space: :class:`FunctionSpace`

            :arg r_loc_func: The radius of localisation for each function
            :type r_loc_func: int

        """

        # number of basis coeffs
        f = Function(function_space)
        self.function_space = function_space
        self.nc = len(f.dat.data)
        self.r_loc_func = r_loc_func

        # iterate over dimensions, saving localisation functions for each one
        self._list = []
        for i in range(self.nc):
            self._list.append(Localisation(function_space, r_loc_func, i, rate))

        super(LocalisationFunctions, self).__init__()

    def __iter__(self):
        """Iterate over the list of localisation functions """
        for m in self._list:
            yield m

    def __len__(self):
        """Return the size of list of localisation functions (number of basis coeffs) """
        return len(self._list)

    def __getitem__(self, idx):
        """Return a localisation function in list

        :arg idx: The :class:`Function` to return """
        return self._list[idx]
