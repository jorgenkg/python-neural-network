from scipy.special import expit
import numpy as np


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

def ReLU_function( signal, derivative=False ):
    # Prevent overflow.
    signal = np.clip( signal, -500, 500 )
    
    if derivative:
        # Return the partial derivation of the activation function
        return expit( signal )
    else:
        # Return the activation signal
        return np.log(1 + np.exp(signal))
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




