import numpy as np


def sum_squared_error( outputs, targets, derivative=False ):
    if derivative:
        return outputs - targets 
    else:
        return np.sum( np.power(outputs - targets,2) )
#end cost function


def cross_entropy_cost( outputs, targets, derivative=False ):
    if derivative:
        return (outputs - targets) / (outputs * (1 - outputs)) 
    else:
        return sum(
            -np.inner(target, output) - np.inner((1 - target), np.log(1 - output ))
            for output, target in zip(outputs, targets)
        ) / float(outputs.shape[0])
#end cost function