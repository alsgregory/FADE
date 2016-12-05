from __future__ import absolute_import

from firedrake import *

# turn off lazy evaluation of par_loops
parameters["pyop2_options"]["lazy_evaluation"] = False

from firedrake_da.utils import *  # noqa
from firedrake_da.observations import *  # noqa
from firedrake_da.localisation import *  # noqa
from firedrake_da.localisation_functions import *  # noqa

from firedrake_da.kalman.cov import *  # noqa
from firedrake_da.kalman.kalman import *  # noqa

from firedrake_da.ensemble_transform.weight_update import *  # noqa
from firedrake_da.ensemble_transform.ensemble_transform import *  # noqa

from firedrake_da.EMD.emd_kernel import *  # noqa

from firedrake_da.ml.coupling import *  # noqa
