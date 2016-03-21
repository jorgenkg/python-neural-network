from activation_functions import softmax_function
from cost_functions import softmax_cross_entropy_cost
from tools import dropout, add_bias
import numpy as np

default_settings = {
    # Optional settings
    "weights_low"           : -0.1,     # Lower bound on initial weight range
    "weights_high"          : 0.1,      # Upper bound on initial weight range
    "save_trained_network"  : False,    # Whether to write the trained weights to disk
    
    "input_layer_dropout"   : 0.0,      # dropout fraction of the input layer
    "hidden_layer_dropout"  : 0.0,      # dropout fraction in all hidden layers
}

class NeuralNet:
    def __init__(self, settings ):
        self.__dict__.update( default_settings )
        self.__dict__.update( settings )
        
        if settings["layers"][0][0]:
            assert not softmax_function in map(lambda (n_nodes, actfunc): actfunc, self.layers[:-1]),\
                "The softmax function can only be applied to the final layer in the network."
        
            assert not self.cost_function == softmax_cross_entropy_cost or self.layers[-1][1] == softmax_function,\
                "The `softmax_cross_entropy_cost` cost function can only be used in combination with the softmax activation function."
        
            assert not self.layers[-1][1] == softmax_function or self.cost_function == softmax_cross_entropy_cost,\
                 "The current implementation of the softmax activation function require the cost function to be `softmax_cross_entropy_cost`."
        
        # Count the required number of weights. This will speed up the random number generation phase
        self.n_weights = (self.n_inputs + 1) * self.layers[0][0] +\
                         sum( (self.layers[i][0] + 1) * layer[0] for i, layer in enumerate( self.layers[1:] ) )
        
        # Initialize the network with new randomized weights
        self.set_weights( self.generate_weights( self.weights_low, self.weights_high ) )
        
        # Initialize the bias weights (bias values) to 1
        for i in xrange(len(self.weights)):
            self.weights[ i ][0,:] = 1.0
    #end
    
    
    def generate_weights(self, low = -0.1, high = 0.1):
        # Generate new random weights for all the connections in the network
        return np.random.uniform(low, high, size=(self.n_weights,))
    #end
    
    
    def unpack(self, weight_list ):
        # This method will create a list of weight matrices. Each list element
        # corresponds to the connection between two layers.
        
        start, stop     = 0, 0
        weight_layers   = [ ]
        previous_shape  = self.n_inputs + 1
        
        for n_neurons, activation_function in self.layers:
            stop += previous_shape * n_neurons
            weight_layers.append( weight_list[ start:stop ].reshape( previous_shape, n_neurons ))
            
            previous_shape = n_neurons + 1
            start = stop
        
        return weight_layers
    #end
    
    
    def set_weights(self, weight_list ):
        # This is a helper method for setting the network weights to a previously defined list.
        # This is useful for utilizing a previously optimized neural network weight set.
        self.weights = self.unpack( weight_list )
    #end
    
    
    def get_weights(self, ):
        # This will stack all the weights in the network on a list, which may be saved to the disk.
        return [w for l in self.weights for w in l.flat]
    #end
    
    
    def error(self, weight_vector, training_data, training_targets ):
        self.weights = self.unpack( np.array(weight_vector) )
        out          = self.update( training_data )
        
        return self.cost_function(out, training_targets )
    #end
    
    
    def gradient(self, weight_vector, training_data, training_targets ):
        layer_indexes              = range( len(self.layers) )[::-1]    # reversed
        self.weights               = self.unpack( np.array(weight_vector) )
        input_signals, derivatives = self.update( training_data, trace=True )
        
        out                        = input_signals[-1]
        cost_derivative            = self.cost_function(out, training_targets, derivative=True).T
        delta                      = cost_derivative * derivatives[-1]
        error                      = self.cost_function(out, training_targets )
        
        layers = []
        for i in layer_indexes:
            # Loop over the weight layers in reversed order to calculate the deltas
            
            # calculate the weight change
            dropped = dropout( 
                        input_signals[i], 
                        # dropout probability
                        self.hidden_layer_dropout if i > 0 else self.input_layer_dropout
                    )
                    
            layers.append(np.dot( delta, add_bias(dropped) ).T.flat)
            
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
    
    
    def print_test(self, testset ):
        test_data    = np.array( [instance.features for instance in testset ] )
        test_targets = np.array( [instance.targets  for instance in testset ] )
        
        input_signals, derivatives = self.update( test_data, trace=True )
        out                        = input_signals[-1]
        error                      = self.cost_function(out, test_targets )
        
        print "[testing] Network error: %.4g" % error
        print "[testing] Network results:"
        print "[testing]   input\tresult\ttarget"
        for entry, result, target in zip(test_data, out, test_targets):
            print "[testing]   %s\t%s\t%s" % tuple(map(str, [entry, result, target]))
    #end
    
    
    def save_to_file(self, filename = "network0.pkl" ):
        import cPickle, os, re
        """
        This save method pickles the parameters of the current network into a 
        binary file for persistant storage.
        """
        
        if filename == "network0.pkl":
            while os.path.exists( os.path.join(os.getcwd(), filename )):
                filename = re.sub('\d(?!\d)', lambda x: str(int(x.group(0)) + 1), filename)
        
        with open( filename , 'wb') as file:
            store_dict = {
                "cost_function"        : self.cost_function,
                "n_inputs"             : self.n_inputs,
                "layers"               : self.layers,
                "n_weights"            : self.n_weights,
                "weights"              : self.weights,
            }
            cPickle.dump( store_dict, file, 2 )
    #end
    
    
    @staticmethod
    def load_from_file( filename = "network.pkl" ):
        """
        Load the complete configuration of a previously stored network.
        """
        network = NeuralNet( {"n_inputs":1, "layers":[[0,None]]} )
        
        with open( filename , 'rb') as file:
            import cPickle
            store_dict                   = cPickle.load(file)
            
            network.n_inputs             = store_dict["n_inputs"]            
            network.n_weights            = store_dict["n_weights"]           
            network.layers               = store_dict["layers"]
            network.weights              = store_dict["weights"]             
            network.cost_function        = store_dict["cost_function"]
        
        return network
    #end
#end class