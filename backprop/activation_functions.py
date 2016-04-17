import numpy as np

try:
    # PYPY hasn't got scipy
    from scipy.special import expit
except:
    expit = lambda x: 1.0 / (1 + np.exp(-x))


def softmax_function( signal, derivative=False ):
    # Calculate activation signal
    e_x = np.exp( signal - np.max(signal, axis=1, keepdims = True) )
    signal = e_x / np.sum( e_x, axis = 1, keepdims = True )
    
    if derivative:
        return np.ones( signal.shape )
    else:
        # Return the activation signal
        return signal
#end activation function


def sigmoid_function( signal, derivative=False ):
    # Prevent overflow.
    signal = np.clip( signal, -500, 500 )
    
    # Calculate activation signal
    signal = expit( signal )
    
    if derivative:
        # Return the partial derivation of the activation function
        return np.multiply(signal, 1 - signal)
    else:
        # Return the activation signal
        return signal
#end activation function


def elliot_function( signal, derivative=False ):
    """ A fast approximation of sigmoid """
    s = 1 # steepness
    
    abs_signal = (1 + np.abs(signal * s))
    if derivative:
        return 0.5 * s / abs_signal**2
    else:
        # Return the activation signal
        return 0.5*(signal * s) / abs_signal + 0.5
#end activation function


def symmetric_elliot_function( signal, derivative=False ):
    """ A fast approximation of tanh """
    s = 1.0 # steepness
    
    abs_signal = (1 + np.abs(signal * s))
    if derivative:
        return s / abs_signal**2
    else:
        # Return the activation signal
        return (signal * s) / abs_signal
#end activation function


def ReLU_function( signal, derivative=False ):
    if derivative:
        return (signal > 0).astype(float)
    else:
        # Return the activation signal
        return np.maximum( 0, signal )
#end activation function


def LReLU_function( signal, derivative=False, leakage = 0.01 ):
    """
    Leaky Rectified Linear Unit
    """
    if derivative:
        # Return the partial derivation of the activation function
        return np.clip(signal > 0, leakage, 1.0)
    else:
        # Return the activation signal
        output = np.copy( signal )
        output[ output < 0 ] *= leakage
        return output
#end activation function


def tanh_function( signal, derivative=False ):
    # Calculate activation signal
    signal = np.tanh( signal )
    
    if derivative:
        # Return the partial derivation of the activation function
        return 1-np.power(signal,2)
    else:
        # Return the activation signal
        return signal
#end activation function


def linear_function( signal, derivative=False ):
    if derivative:
        # Return the partial derivation of the activation function
        return np.ones( signal.shape )
    else:
        # Return the activation signal
        return signal
#end activation function


def softplus_function( signal, derivative=False ):
    if derivative:
        # Return the partial derivation of the activation function
        return np.exp(signal) / (1 + np.exp(signal))
    else:
        # Return the activation signal
        return np.log(1 + np.exp(signal))
#end activation function


def softsign_function( signal, derivative=False ):
    if derivative:
        # Return the partial derivation of the activation function
        return 1. / (1 + np.abs(signal))**2
    else:
        # Return the activation signal
        return signal / (1 + np.abs(signal))
#end activation function


