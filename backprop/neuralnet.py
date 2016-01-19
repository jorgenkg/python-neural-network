from tools import dropout, add_bias, confirm
import numpy as np
import collections

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
        
        # Count the required number of weights. This will speed up the random number generation phase
        self.n_weights = (self.n_inputs + 1) * self.layers[0][0] +\
                         sum( (self.layers[i][0] + 1) * layer[0] for i, layer in enumerate( self.layers[1:] ) )
        
        # Initialize the network with new randomized weights
        self.set_weights( self.generate_weights( self.weights_low, self.weights_high ) )
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
    
    def backpropagation(self, trainingset, ERROR_LIMIT = 1e-3, learning_rate = 0.03, momentum_factor = 0.9, max_iterations = ()  ):
        
        assert trainingset[0].features.shape[0] == self.n_inputs, \
                "ERROR: input size varies from the defined input setting"
        
        assert trainingset[0].targets.shape[0]  == self.layers[-1][0], \
                "ERROR: output size varies from the defined output setting"
        
        
        training_data              = np.array( [instance.features for instance in trainingset ] )
        training_targets           = np.array( [instance.targets  for instance in trainingset ] )
                                
        layer_indexes              = range( len(self.layers) )[::-1]    # reversed
        momentum                   = collections.defaultdict( int )
        MSE                        = ( ) # inf
        epoch                      = 0
        
        input_signals, derivatives = self.update( training_data, trace=True )
        
        out                        = input_signals[-1]
        error                      = (out - training_targets).T
        delta                      = error * derivatives[-1]
        MSE                        = np.mean( np.power(error,2) )
        
        while MSE > ERROR_LIMIT and epoch < max_iterations:
            epoch += 1
            
            for i in layer_indexes:
                # Loop over the weight layers in reversed order to calculate the deltas
                
                # perform dropout
                dropped = dropout( 
                            add_bias(input_signals[i]), 
                            # dropout probability
                            self.hidden_layer_dropout if i else self.input_layer_dropout
                        )
                
                # calculate the weight change
                dW = -learning_rate * np.dot( delta, dropped ).T + momentum_factor * momentum[i]
                
                if i!= 0:
                    """Do not calculate the delta unnecessarily."""
                    # Skip the bias weight
                    weight_delta = np.dot( self.weights[ i ][1:,:], delta )
        
                    # Calculate the delta for the subsequent layer
                    delta = weight_delta * derivatives[i-1]
                
                # Store the momentum
                momentum[i] = dW
                                    
                # Update the weights
                self.weights[ i ] += dW
            #end weight adjustment loop
            
            input_signals, derivatives = self.update( training_data, trace=True )
            out                        = input_signals[-1]
            error                      = (out - training_targets).T
            delta                      = error * derivatives[-1]
            MSE                        = np.mean( np.power(error,2) )
            
            
            if epoch%1000==0:
                # Show the current training status
                print "* current network error (MSE):", MSE
        
        print "* Converged to error bound (%.4g) with MSE = %.4g." % ( ERROR_LIMIT, MSE )
        print "* Trained for %d epochs." % epoch
        
        if self.save_trained_network and confirm( promt = "Do you wish to store the trained network?" ):
            self.save_to_file()
    # end backprop
    
    
    def update(self, input_values, trace=False ):
        # This is a forward operation in the network. This is how we 
        # calculate the network output from a set of input signals.
        output          = input_values
        
        if trace: 
            derivatives = [ ]        # collection of the derivatives of the act functions
            outputs     = [ output ] # passed through act. func.
        
        for i, weight_layer in enumerate(self.weights):
            # Loop over the network layers and calculate the output
            signal = np.dot( add_bias(output), weight_layer )
            output = self.layers[i][1]( signal )
            
            if trace: 
                outputs.append( output )
                # Calculate the derivative, used during weight update
                derivatives.append( self.layers[i][1]( signal, derivative = True ).T )
        
        if trace: 
            return outputs, derivatives
        
        return output
    #end
    
    def test(self, testset, DEBUG = False ):
        if DEBUG:
            for instance in testset:
                out = self.update( np.array([instance.features]) )
                error = out - instance.targets
                MSE = np.mean( np.power(error,2) )
            
                print "Output: {output} \tTarget: {target}\tError: {error}".format( 
                            output   = str(out), 
                            target   = str(instance.targets),
                            error    = str(MSE)
                        )
        #end debug
        
        test_data    = np.array( [instance.features for instance in testset ] )
        test_targets = np.array( [instance.targets  for instance in testset ] )
        
        out               = self.update( test_data )
        error             = (out - test_targets).T
        MSE               = np.mean( np.power(error,2) )
        
        return MSE
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
                "n_inputs"             : self.n_inputs,
                "n_outputs"            : self.n_outputs,
                "n_hiddens"            : self.n_hiddens,
                "n_hidden_layers"      : self.n_hidden_layers,
                "activation_functions" : self.activation_functions,
                "n_weights"            : self.n_weights,
                "weights"              : self.weights

            }
            cPickle.dump( store_dict, file, 2 )
    #end
    
    @staticmethod
    def load_from_file( filename = "network.pkl" ):
        """
        Load the complete configuration of a previously stored network.
        """
        network = NeuralNet( 0, 0, 0, 0, [0] )
        
        with open( filename , 'rb') as file:
            import cPickle
            store_dict = cPickle.load(file)
            
            network.n_inputs             = store_dict["n_inputs"]            
            network.n_outputs            = store_dict["n_outputs"]           
            network.n_hiddens            = store_dict["n_hiddens"]           
            network.n_hidden_layers      = store_dict["n_hidden_layers"]     
            network.n_weights            = store_dict["n_weights"]           
            network.weights              = store_dict["weights"]             
            network.activation_functions = store_dict["activation_functions"]
        
        return network
    #end
#end class