import numpy as np

try:
    # PYPY hasn't got scipy
    from scipy.special import expit
except:
    expit = lambda x: 1.0 / (1 + np.exp(-x))


def sigmoid_function( signal, derivative=False ):
    # Prevent overflow.
    signal = np.clip( signal, -500, 500 )
    
    # Calculate activation signal
    signal = expit( signal )
    
    if derivative:
        # Return the partial derivation of the activation function
        return np.multiply(signal, 1-signal)
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
        derivate = np.maximum( 0, signal )
        derivate[ derivate != 0 ] = 1.
        return derivate
    else:
        # Return the activation signal
        return np.maximum( 0, signal )
#end activation function


def LReLU_function( signal, derivative=False ):
    """
    Leaky Rectified Linear Unit
    """
    if derivative:
        derivate = np.copy( signal )
        derivate[ derivate < 0 ] = 0.01
        derivate[ derivate > 0 ] = 1.0
        # Return the partial derivation of the activation function
        return derivate
    else:
        # Return the activation signal
        output = np.copy( signal )
        output[ output < 0 ] *= 0.01
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
        return 1
    else:
        # Return the activation signal
        return signal
#end activation function




