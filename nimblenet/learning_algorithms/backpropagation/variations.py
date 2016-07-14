import numpy as np
import collections

from base import backpropagation_foundation


default_configuration = {
    'ERROR_LIMIT'           : 0.001, 
    'learning_rate'         : 0.03, 
    'batch_size'            : 1, 
    'print_rate'            : 1000, 
    'save_trained_network'  : False,
    'input_layer_dropout'   : 0.0,
    'hidden_layer_dropout'  : 0.0, 
    'evaluation_function'   : None,
    'max_iterations'        : ()
}

def adagrad(network, trainingset, testset, cost_function, epsilon = 1e-8, **kwargs ):
    configuration = dict(default_configuration)
    configuration.update( kwargs )
    
    learning_rate = configuration["learning_rate"]
    cache         = [ np.zeros( shape = weight_layer.shape ) for weight_layer in network.weights ]
    
    def calculate_dW( layer_index, dX ):
        cache[ layer_index ] += np.power( dX, 2 )
        return -learning_rate * dX / (np.sqrt(cache[ layer_index ]) + epsilon)
    #end
    
    return backpropagation_foundation( network, trainingset, testset, cost_function, calculate_dW, **configuration  )
#end



def RMSprop(network, trainingset, testset, cost_function, decay_rate = 0.99, epsilon = 1e-8, **kwargs  ):
    configuration = dict(default_configuration)
    configuration.update( kwargs )
    
    learning_rate = configuration["learning_rate"]
    cache         = [ np.zeros( shape = weight_layer.shape ) for weight_layer in network.weights ]
    
    def calculate_dW( layer_index, dX ):
        cache[ layer_index ] = decay_rate * cache[ layer_index ] + (1 - decay_rate) * dX**2
        return -learning_rate * dX / (np.sqrt(cache[ layer_index ]) + epsilon)
    #end
    
    return backpropagation_foundation( network, trainingset, testset, cost_function, calculate_dW, **configuration )
#end



def Adam(network, trainingset, testset, cost_function, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, **kwargs ):
    configuration = dict(default_configuration)
    configuration.update( kwargs )
    
    learning_rate = configuration["learning_rate"]
    m = [ np.zeros( shape = weight_layer.shape ) for weight_layer in network.weights ]
    v = [ np.zeros( shape = weight_layer.shape ) for weight_layer in network.weights ]
    
    def calculate_dW( layer_index, dX ):
        m[ layer_index ] = beta1 * m[ layer_index ] + ( 1 - beta1 ) * dX
        v[ layer_index ] = beta2 * v[ layer_index ] + ( 1 - beta2 ) * ( dX**2 )
        
        return -learning_rate * m[ layer_index ] / ( np.sqrt(v[ layer_index ]) + epsilon )
    #end
    
    return backpropagation_foundation( network, trainingset, testset, cost_function, calculate_dW, **configuration  )
#end



def nesterov_momentum(network, trainingset, testset, cost_function, momentum_factor = 0.9, **kwargs  ):
    configuration = dict(default_configuration)
    configuration.update( kwargs )
    
    learning_rate = configuration["learning_rate"]
    momentum = collections.defaultdict( int )
    
    def calculate_dW( layer_index, dX ):
        dW = -learning_rate * dX + momentum_factor * momentum[ layer_index ]
        weight_change = -momentum_factor * momentum[ layer_index ] + (1 + momentum_factor) * dW
        
        # Store the dW after calculating the weight change since we would like 
        # to use the "previous" momentum.
        momentum[ layer_index ] = dW 
        
        return dW
    #end
    
    return backpropagation_foundation( network, trainingset, testset, cost_function, calculate_dW, **configuration  )
#end



def classical_momentum(network, trainingset, testset, cost_function, momentum_factor = 0.9, **kwargs  ):
    configuration = dict(default_configuration)
    configuration.update( kwargs )
    
    learning_rate = configuration["learning_rate"]
    momentum = collections.defaultdict( int )
    
    def calculate_dW( layer_index, dX ):
        dW = -learning_rate * dX + momentum_factor * momentum[ layer_index ]
        momentum[ layer_index ] = dW 
        return dW
    #end
    
    return backpropagation_foundation( network, trainingset, testset, cost_function, calculate_dW, **configuration  )
#end



def vanilla(network, trainingset, testset, cost_function, **kwargs ):
    configuration = dict(default_configuration)
    configuration.update( kwargs )
    
    learning_rate = configuration["learning_rate"]
    def calculate_dW( layer_index, dX ):
        return -learning_rate * dX
    #end
    
    return backpropagation_foundation( network, trainingset, testset, cost_function, calculate_dW, **configuration  )
#end