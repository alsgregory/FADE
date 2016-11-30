# firedrake-da
A toolbox for data-assimilation algorithms for use with Firedrake

## Linux terminal installation
As a prerequisite, it is required to have Firedrake installed alongside it's dependencies. Instructions on how to this can be found at: http://firedrakeproject.org/download.html. Then, it is required to have the Firedrake virtualenv activated; instructions for this can be found by again following the aforementioned link.

PuLP (https://pypi.python.org/pypi/PuLP) is also required for the Earth Mover's Distance algorithm included in this toolbox. To install, type the following command into the terminal:

1. `pip install pulp`

Finally, for some demos to work it is required to have Firedrake-MLMC installed (a side-package of http://firedrakeproject.org). Instructions on how to do this can be found at: https://github.com/firedrakeproject/firedrake-mlmc.

Then, to install firedrake_da, type the following commands into the terminal:

1. `git clone https://github.com/firedrake_da `
2. `pip install -e ./firedrake_da`

## Contact
For any enquiries, please contact: a.gregory14@imperial.ac.uk

## Author Details
Mr. Alastair Gregory, Research Postgraduate, Imperial College London
