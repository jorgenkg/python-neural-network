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
        return np.sum( np.power(root_difference, 2 ), axis = 1) / math.sqrt( 2 )
#end cost function


def cross_entropy_cost( outputs, targets, derivative=False, epsilon=1e-11 ):
    """
    The output signals should be in the range [0, 1]
    """
    # Prevent overflow
    outputs = np.clip(outputs, epsilon, 1 - epsilon)
    divisor = np.maximum(outputs * (1 - outputs), epsilon)
    
    if derivative:
        return (outputs - targets) / divisor
    else:
        return -np.sum(targets * np.log( outputs ) + (1 - targets) * np.log(1 - outputs))
#end cost function


def softmax_neg_loss( outputs, targets, derivative=False, epsilon=1e-11 ):
    """
    The output signals should be in the range [0, 1]
    """
    outputs = np.clip(outputs, epsilon, 1 - epsilon)
    
    if derivative:
        return outputs - targets
    else:
        return -np.sum(targets * np.log( outputs ))
#end cost function