import numpy as np
import math

def sum_squared_error( outputs, targets, derivative=False ):
    if derivative:
        return outputs - targets 
    else:
        return 0.5 * np.sum( np.power(outputs - targets,2) )
#end cost function

def hellinger_distance( outputs, targets, derivative=False ):
    """
    The output signals should be in the range [0, 1]
    """
    root_difference = np.sqrt( outputs ) - np.sqrt( targets )
    
    if derivative:
        return root_difference/( np.sqrt(2) * np.sqrt( outputs ))
    else:
        return np.sum( np.power(root_difference, 2 )) / math.sqrt( 2 )
#end cost function


def cross_entropy_cost( outputs, targets, derivative=False ):
    """
    The output signals should be in the range [0, 1]
    """
    if derivative:
        return (outputs - targets) / (outputs * (1 - outputs)) 
    else:
        return np.mean(-np.sum(targets * np.log( outputs ) + (1 - targets) * np.log(1 - outputs), axis=1))
#end cost function


def softmax_cross_entropy_cost( outputs, targets, derivative=False ):
    """
    The output signals should be in the range [0, 1]
    """
    if derivative:
        return outputs - targets
    else:
        return np.mean(-np.sum(targets * np.log( outputs ), axis=1))
#end cost function


def exponential_cost( outputs, targets, derivative=False, tau = 2.0 ):
    diff = outputs - targets
    cost = tau * np.exp( np.sum( diff**2 ) / tau )
    
    if derivative:
        return 2/tau * diff * cost
    
    return np.sum(cost) - tau
#end cost function