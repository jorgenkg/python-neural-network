from scipy.stats import bernoulli
import numpy as np


class Instance:
    # This is a simple encapsulation of a `input signal : output signal`
    # pair in our training set.
    def __init__(self, features, target):
        self.features = np.array(features)
        self.targets  = np.array(target)
#endclass Instance


def dropout( X, p = 0. ):
    if p > 0:
        retain_p = 1 - p
        X = np.multiply( bernoulli.rvs( retain_p, size = X.shape ), X )
        X /= retain_p
    return X
#end  


def add_bias(A):
    return np.hstack(( np.ones((A.shape[0],1)), A )) # Add 1 as bias.
#end addBias