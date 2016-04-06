from activation_functions import softmax_function
from cost_functions import softmax_cross_entropy_cost
from tools import add_bias
import numpy as np

default_settings = {
    # Optional settings
    "weights_low"           : -0.1,     # Lower bound on initial weight range
    "weights_high"          : 0.1,      # Upper bound on initial weight range
    "save_trained_network"  : False,    # Whether to write the trained weights to disk
}

class NeuralNet:
    def __init__(self, settings ):
        self.__dict__.update( default_settings )
        self.__dict__.update( settings )
        
        assert not softmax_function in map(lambda (n_nodes, actfunc): actfunc, self.layers[:-1]),\
            "The softmax function can only be applied to the final layer in the network."
        
        # Count the required number of weights. This will speed up the random number generation when initializing weights
        self.n_weights = (self.n_inputs + 1) * self.layers[0][0] +\
                         sum( (self.layers[i][0] + 1) * layer[0] for i, layer in enumerate( self.layers[1:] ) )
        
        # Initialize the network with new randomized weights
        self.set_weights( self.generate_weights( self.weights_low, self.weights_high ) )
    #end
    
    
    def generate_weights(self, low = -0.1, high = 0.1):
        # Generate new random weights for all the connections in the network
        return np.random.uniform(low, high, size=(self.n_weights,))
    #end
    
    
    def set_weights(self, weight_list ):
        # This is a helper method for setting the network weights to a previously defined list
        # as it's useful for loading a previously optimized neural network weight set.
        # The method creates a list of weight matrices. Each list entry correspond to the 
        # connection between two layers.
        
        start, stop     = 0, 0
        self.weights    = [ ]
        previous_shape  = self.n_inputs + 1
        
        for n_neurons, activation_function in self.layers:
            stop += previous_shape * n_neurons
            self.weights.append( weight_list[ start:stop ].reshape( previous_shape, n_neurons ))
            
            previous_shape = n_neurons + 1
            start          = stop
    #end
    
    
    def get_weights(self, ):
        # This will stack all the weights in the network on a list, which may be saved to the disk.
        return [w for l in self.weights for w in l.flat]
    #end
    
    
    def error(self, weight_vector, training_data, training_targets, cost_function ):
        # assign the weight_vector as the network topology
        self.set_weights( np.array(weight_vector) )
        # perform a forward operation to calculate the output signal
        out = self.update( training_data )
        # evaluate the output signal with the cost function
        return cost_function(out, training_targets )
    #end
    
    
    def gradient(self, weight_vector, training_data, training_targets, cost_function ):
        # assign the weight_vector as the network topology
        self.set_weights( np.array(weight_vector) )
        
        layer_indexes              = range( len(self.layers) )[::-1]    # reversed
        input_signals, derivatives = self.update( training_data, trace=True )
        
        out                        = input_signals[-1]
        cost_derivative            = cost_function(out, training_targets, derivative=True).T
        delta                      = cost_derivative * derivatives[-1]
        error                      = cost_function(out, training_targets )
        
        layers = []
        for i in layer_indexes:
            # Loop over the weight layers in reversed order to calculate the deltas
            layers.append(np.dot( delta, add_bias(input_signals[i]) ).T.flat)
            
            if i!= 0:
                """Do not calculate the delta unnecessarily."""
                # Skip the bias weight
                weight_delta = np.dot( self.weights[ i ][1:,:], delta )
    
                # Calculate the delta for the subsequent layer
                delta = weight_delta * derivatives[i-1]
        #end weight adjustment loop
        
        return np.hstack( reversed(layers) )
    # end gradient
    
    
    def update(self, input_values, trace=False ):
        # This is a forward operation in the network. This is how we 
        # calculate the network output from a set of input signals.
        output          = input_values
        
        if trace: 
            derivatives = [ ]        # collection of the derivatives of the act functions
            outputs     = [ output ] # passed through act. func.
        
        for i, weight_layer in enumerate(self.weights):
            # Loop over the network layers and calculate the output
            signal = np.dot( output, weight_layer[1:,:] ) + weight_layer[0:1,:] # implicit bias
            output = self.layers[i][1]( signal )
            
            if trace: 
                outputs.append( output )
                # Calculate the derivative, used during weight update
                derivatives.append( self.layers[i][1]( signal, derivative = True ).T )
        
        if trace: 
            return outputs, derivatives
        
        return output
    #end
#end class