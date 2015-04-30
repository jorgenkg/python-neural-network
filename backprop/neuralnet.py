import numpy as np
import math
import random
import itertools
import collections

class NeuralNet:
    def __init__(self,n_inputs, n_outputs, n_hiddens, n_hidden_layers, activation_functions):
        self.n_inputs = n_inputs                # Number of network input signals
        self.n_outputs = n_outputs              # Number of desired outputs from the network
        self.n_hiddens = n_hiddens              # Number of nodes in each hidden layer
        self.n_hidden_layers = n_hidden_layers  # Number of hidden layers in the network
        self.activation_functions = activation_functions
        
        assert len(activation_functions)==(n_hidden_layers+1), "Requires "+(n_hidden_layers+1)+" activation functions, got: "+len(activation_functions)+"."
        
        if n_hidden_layers == 0:
            # Count the necessary number of weights for the input->output connection.
            # input -> [] -> output
            self.n_weights = ((n_inputs+1)*n_outputs)
        else:
            # Count the necessary number of weights summed over all the layers.
            # input -> [n_hiddens -> n_hiddens] -> output
            self.n_weights = (n_inputs+1)*n_hiddens+\
                             (n_hiddens**2+n_hiddens)*(n_hidden_layers-1)+\
                             n_hiddens*n_outputs+n_outputs
        
        # Initialize the network with new randomized weights
        self.set_weights( self.generate_weights() )
    #end
    
    
    def generate_weights(self, low=-0.1, high=0.1):
        # Generate new random weights for all the connections in the network
        if not False:
            # Support NumPy
            return [random.uniform(low,high) for _ in xrange(self.n_weights)]
        else:
            return np.random.uniform(low, high, size=(1,self.n_weights)).tolist()[0]
    #end
    
    
    def unpack(self, weight_list ):
        # This method will create a list of weight matrices. Each list element
        # corresponds to the connection between two layers.
        if self.n_hidden_layers == 0:
            return [ np.array(weight_list).reshape(self.n_inputs+1,self.n_outputs) ]
        else:
            weight_layers = [ np.array(weight_list[:(self.n_inputs+1)*self.n_hiddens]).reshape(self.n_inputs+1,self.n_hiddens) ]
            weight_layers += [ np.array(weight_list[(self.n_inputs+1)*self.n_hiddens+(i*(self.n_hiddens**2+self.n_hiddens)):(self.n_inputs+1)*self.n_hiddens+((i+1)*(self.n_hiddens**2+self.n_hiddens))]).reshape(self.n_hiddens+1,self.n_hiddens) for i in xrange(self.n_hidden_layers-1) ]
            weight_layers += [ np.array(weight_list[(self.n_inputs+1)*self.n_hiddens+((self.n_hidden_layers-1)*(self.n_hiddens**2+self.n_hiddens)):]).reshape(self.n_hiddens+1,self.n_outputs) ]
            
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
    
    def backpropagation(self, trainingset, ERROR_LIMIT=1e-3, learning_rate=0.3, momentum_factor=0.9  ):
        def addBias(A):
            # Add 1 as bias.
            return np.hstack((np.ones((A.shape[0],1)),A))
        #end addBias
        
        assert trainingset[0].features.shape[0] == self.n_inputs, "ERROR: input size varies from the defined input setting"
        assert trainingset[0].targets.shape[0] == self.n_outputs, "ERROR: output size varies from the defined output setting"
        
        training_data = np.array( [instance.features for instance in trainingset ] )
        training_targets = np.array( [instance.targets for instance in trainingset ] )
        
        output_activation_function = self.activation_functions[-1]
        
        MSE = ( ) # inf
        neterror = None
        
        momentum = collections.defaultdict( int )
        
        epoch = 0
        while MSE>ERROR_LIMIT:
            epoch += 1
            
            input_layers = self.update( training_data, trace=True )
            out = input_layers[-1]
            
            error = training_targets - out
            
            neterror = error
            
            #deltas = [error, ]
            deltas = [ np.multiply( output_activation_function( out, derivative = True ), error ) ]
            
            # Loop over the weight layers in reversed order to calculate the deltas
            loop = itertools.izip(
                            xrange(len(self.weights)-1, -1, -1),
                            reversed(self.weights),
                            reversed(input_layers[:-1]),
                        )
            
            for i, weight_layer, input_signals in loop:
                # The delta calculated from the previous layer.
                prev_delta = deltas[-1]
                
                if i!= 0:
                    """Do not calculate the delta unnecessarily."""
                    # Skipping the bias weight during calculation.
                    weight_delta = np.dot( prev_delta, weight_layer[1:,:].T )
            
                    # Calculate the delta for the subsequent layer
                    delta = np.multiply(  weight_delta, self.activation_functions[i-1]( input_signals, derivative=True) )
                    deltas.append( delta )
                  
                # Calculate weight change 
                dW = learning_rate * np.dot( addBias(input_signals).T, prev_delta )  + momentum_factor * momentum[i]
                
                # Store momentum
                momentum[i] = dW
                
                # Update weights
                self.weights[ i ] += dW
            
            MSE = np.mean( np.power(neterror,2) )
            if epoch%1000==0: print "* current network error:", MSE
        print "* epochs trained:", epoch
    # end backprop
    
    
    def update(self, input_values, trace=False ):
        # This is a forward operation in the network. This is how we calculate the network output
        # from a set of input signals.
        
        def addBias(A):
            # Add 1 as bias.
            return np.hstack((np.ones((A.shape[0],1)),A))
        #end addBias
        
        output = input_values
        if trace: tracelist = [ output ]
        
        for i, weight_layer in enumerate(self.weights):
            # Loop over the network layers and calculate the output
            output = addBias(output).dot( weight_layer )
            output = self.activation_functions[i]( output )
            if trace: tracelist.append( output )
        
        if trace: return tracelist
        
        return output
    #end
#end class