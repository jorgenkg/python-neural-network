from activation_functions import softmax_function
from cost_functions import softmax_neg_loss

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
        
        assert not softmax_function in map(lambda x: x[1], self.layers) or softmax_function == self.layers[-1][1],\
            "The `softmax` activation function may only be used in the final layer."
        
        # Count the required number of weights. This will speed up the random number generation when initializing weights
        self.n_weights = (self.n_inputs + 1) * self.layers[0][0] +\
                         sum( (self.layers[i][0] + 1) * layer[0] for i, layer in enumerate( self.layers[1:] ) )
        
        # Initialize the network with new randomized weights
        self.set_weights( self.generate_weights( self.weights_low, self.weights_high ) )
        
        # Initalize the bias to 0.01
        for index in xrange(len(self.layers)):
            self.weights[index][:1,:] = 0.01
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
        start, stop         = 0, 0
        self.weights        = [ ]
        previous_shape      = self.n_inputs + 1 # +1 because of the bias
        
        for n_neurons, activation_function in self.layers:
            stop           += previous_shape * n_neurons
            self.weights.append( weight_list[ start:stop ].reshape( previous_shape, n_neurons ))
            
            previous_shape  = n_neurons + 1     # +1 because of the bias
            start           = stop
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
    
    
    def measure_quality(self, training_data, training_targets, cost_function ):
        # perform a forward operation to calculate the output signal
        out = self.update( training_data )
        # calculate the mean error on the data classification
        mean_error = cost_function( out, training_targets ) / float(training_data.shape[0])
        # calculate the numeric range between the minimum and maximum output value
        range_of_predicted_values = np.max(out) - np.min(out)
        # return the measured quality 
        return 1 - (mean_error / range_of_predicted_values)
    #end
    
    
    def gradient(self, weight_vector, training_data, training_targets, cost_function ):
        assert softmax_function != self.layers[-1][1] or cost_function == softmax_neg_loss,\
            "When using the `softmax` activation function, the cost function MUST be `softmax_neg_loss`."
        assert cost_function != softmax_neg_loss or softmax_function == self.layers[-1][1],\
            "When using the `softmax_neg_loss` cost function, the activation function in the final layer MUST be `softmax`."
        
        # assign the weight_vector as the network topology
        self.set_weights( np.array(weight_vector) )
        
        input_signals, derivatives  = self.update( training_data, trace=True )                  
        out                         = input_signals[-1]
        cost_derivative             = cost_function(out, training_targets, derivative=True).T
        delta                       = cost_derivative * derivatives[-1]
        
        layer_indexes               = range( len(self.layers) )[::-1]    # reversed
        n_samples                   = float(training_data.shape[0])
        deltas_by_layer             = []
        
        for i in layer_indexes:
            # Loop over the weight layers in reversed order to calculate the deltas
            deltas_by_layer.append(list((np.dot( delta, add_bias(input_signals[i]) )/n_samples).T.flat))
            
            if i!= 0:
                # i!= 0 because we don't want calculate the delta unnecessarily.
                weight_delta        = np.dot( self.weights[ i ][1:,:], delta ) # Skip the bias weight
    
                # Calculate the delta for the subsequent layer
                delta               = weight_delta * derivatives[i-1]
        #end weight adjustment loop
        
        return np.hstack( reversed(deltas_by_layer) )
    # end gradient
    
    
    def check_gradient(self, trainingset, cost_function, epsilon = 1e-4 ):
        assert trainingset[0].features.shape[0] == self.n_inputs, \
            "ERROR: input size varies from the configuration. Configured as %d, instance had %d" % (self.n_inputs, trainingset[0].features.shape[0])
        assert trainingset[0].targets.shape[0]  == self.layers[-1][0], \
            "ERROR: output size varies from the configuration. Configured as %d, instance had %d" % (self.layers[-1][0], trainingset[0].targets.shape[0])
        
        training_data           = np.array( [instance.features for instance in trainingset ][:100] ) # perform the test with at most 100 instances
        training_targets        = np.array( [instance.targets  for instance in trainingset ][:100] )
        
        # assign the weight_vector as the network topology
        initial_weights         = np.array(self.get_weights())
        numeric_gradient        = np.zeros( initial_weights.shape )
        perturbed               = np.zeros( initial_weights.shape )
        n_samples               = float(training_data.shape[0])
        
        print "[gradient check] Running gradient check..."
        
        for i in xrange( self.n_weights ):
            perturbed[i]        = epsilon
            right_side          = self.error( initial_weights + perturbed, training_data, training_targets, cost_function )
            left_side           = self.error( initial_weights - perturbed, training_data, training_targets, cost_function )
            numeric_gradient[i] = (right_side - left_side) / (2 * epsilon)
            perturbed[i]        = 0
        #end loop
        
        # Reset the weights
        self.set_weights( initial_weights )
        
        # Calculate the analytic gradient
        analytic_gradient       = self.gradient( self.get_weights(), training_data, training_targets, cost_function )
        
        # Compare the numeric and the analytic gradient
        ratio                   = np.linalg.norm(analytic_gradient - numeric_gradient) / np.linalg.norm(analytic_gradient + numeric_gradient)
        
        if not ratio < 1e-6:
            raise Exception( "The numeric gradient check failed! %g" % ratio )
        else:
            print "[gradient check] Passed!"
        
        return ratio
    #end
    
    
    def update(self, input_values, trace=False ):
        # This is a forward operation in the network. This is how we 
        # calculate the network output from a set of input signals.
        output          = input_values
        
        if trace: 
            derivatives = [ ]        # collection of the derivatives of the act functions
            outputs     = [ output ] # passed through act. func.
        
        for i, weight_layer in enumerate(self.weights):
            # Loop over the network layers and calculate the output
            signal      = np.dot( output, weight_layer[1:,:] ) + weight_layer[0:1,:] # implicit bias
            output      = self.layers[i][1]( signal )
            
            if trace: 
                outputs.append( output )
                derivatives.append( self.layers[i][1]( signal, derivative = True ).T ) # the derivative used for weight update
        
        if trace: 
            return outputs, derivatives
        
        return output
    #end
    
    
    def predict(self, predict_set ):
        """
        This method accepts a list of Instances
        
        Eg: list_of_inputs = [ Instance([0.12, 0.54, 0.84]), Instance([0.15, 0.29, 0.49]) ]
        """
        predict_data           = np.array( [instance.features for instance in predict_set ] )
        
        return self.update( predict_data )
    #end
    
    def save_network_to_file(self, filename = "network0.pkl" ):
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
                "layers"               : self.layers,
                "n_weights"            : self.n_weights,
                "weights"              : self.weights,
            }
            cPickle.dump( store_dict, file, 2 )
    #end

    @staticmethod
    def load_network_from_file( filename ):
        import cPickle
        """
        Load the complete configuration of a previously stored network.
        """
        network = NeuralNet( {"n_inputs":1, "layers":[[0,None]]} )
    
        with open( filename , 'rb') as file:
            store_dict                   = cPickle.load(file)
        
            network.n_inputs             = store_dict["n_inputs"]            
            network.n_weights            = store_dict["n_weights"]           
            network.layers               = store_dict["layers"]
            network.weights              = store_dict["weights"]             
    
        return network
    #end
#end class