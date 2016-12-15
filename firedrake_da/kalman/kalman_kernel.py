""" Functions generating a kernel for a ensemble kalman filter update. """

from __future__ import division

from __future__ import absolute_import


class kalman_kernel_generation(object):

    def __init__(self, n, nc):

        """ Generates the kernel needed for a kalman update

            :arg n: Size of ensemble
            :type n: int

            :arg nc: Number of degrees of freedom in ensemble functions
            :type nc: int

        """

        self.n = n
        self.nc = nc

        # get matrix multiplication string
        mm_str = self.__get_matrix_multiplication_str(self.n, self.nc)

        # generate emd kernel
        self.kalman_kernel = """
        for (int k=0;k<input_vector.dofs;k++){
            """ + mm_str + """
        }
        """

        super(kalman_kernel_generation, self).__init__()

    def __get_matrix_multiplication_str(self, n, nc):

        mm_str = """ """
        for i in range(nc):
            for j in range(n):
                mm_str += """product_tensor[k][""" + str((i * nc) + j) + """]=input_matrix[k][""" + str(i) + """]*input_vector[k][""" + str(j) + """];\n"""

        return mm_str
