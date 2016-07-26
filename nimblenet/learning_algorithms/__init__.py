#from generalized_hebbian import *
from scaled_conjugate_gradient import scaled_conjugate_gradient
from resilient_backpropagation import resilient_backpropagation
from scipyoptimize import scipyoptimize


from backpropagation.variations import (
        vanilla as backpropagation,
        classical_momentum as backpropagation_classical_momentum,
        nesterov_momentum as backpropagation_nesterov_momentum,
        Adam as Adam,
        RMSprop as RMSprop,
        adagrad as adagrad
    )