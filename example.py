from nimblenet.activation_functions import sigmoid_function
from nimblenet.cost_functions import cross_entropy_cost
from nimblenet.learning_algorithms import *
from nimblenet.neuralnet import NeuralNet
from nimblenet.preprocessing import construct_preprocessor, standarize
from nimblenet.data_structures import Instance
from nimblenet.tools import print_test


# Training set
dataset             = [ Instance( [0,0], [0] ), Instance( [1,0], [1] ), Instance( [0,1], [1] ), Instance( [1,1], [1] ) ]

preprocess          = construct_preprocessor( dataset, [standarize] ) 
training_data       = preprocess( dataset )
test_data           = preprocess( dataset )


cost_function       = cross_entropy_cost
settings            = {
    # Required settings
    "n_inputs"              : 2,       # Number of network input signals
    "layers"                : [  (3, sigmoid_function), (1, sigmoid_function) ],
                                        # [ (number_of_neurons, activation_function) ]
                                        # The last pair in the list dictate the number of output signals
    
    # Optional settings
    "initial_bias_value"    : 0.0,
    "weights_low"           : -0.1,     # Lower bound on the initial weight value
    "weights_high"          : 0.1,      # Upper bound on the initial weight value
}


# initialize the neural network
network             = NeuralNet( settings )
network.check_gradient( training_data, cost_function )

## load a stored network configuration
# network           = NeuralNet.load_network_from_file( "network0.pkl" )


# Train the network using backpropagation
RMSprop(
        network,                            # the network to train
        training_data,                      # specify the training set
        test_data,                          # specify the test set
        cost_function,                      # specify the cost function to calculate error
        
        ERROR_LIMIT             = 1e-2,     # define an acceptable error limit 
        #max_iterations         = 100,      # continues until the error limit is reach if this argument is skipped
                                
        batch_size              = 0,        # 1 := no batch learning, 0 := entire trainingset as a batch, anything else := batch size
        print_rate              = 1000,     # print error status every `print_rate` epoch.
        learning_rate           = 0.3,      # learning rate
        momentum_factor         = 0.9,      # momentum
        input_layer_dropout     = 0.0,      # dropout fraction of the input layer
        hidden_layer_dropout    = 0.0,      # dropout fraction in all hidden layers
        save_trained_network    = False     # Whether to write the trained weights to disk
    )



## Train the network using SciPy
#scipyoptimize(
#        network,
#        training_data,                      # specify the training set
#        test_data,                          # specify the test set
#        cost_function,                      # specify the cost function to calculate error
#        method               = "L-BFGS-B",
#        save_trained_network = False        # Whether to write the trained weights to disk
#    )



## Train the network using Scaled Conjugate Gradient
#scaled_conjugate_gradient(
#        network,
#        training_data,                  # specify the training set
#        test_data,                      # specify the test set
#        cost_function,                  # specify the cost function to calculate error
#        
#        # optional parameters
#        print_rate           = 1000,    # print error status every `print_rate` epoch.
#        ERROR_LIMIT          = 1e-4,    # define an acceptable error limit 
#        save_trained_network = False    # Whether to write the trained weights to disk
#    )


## Train the network using resilient backpropagation
#resilient_backpropagation(
#        network,
#        training_data,                  # specify the training set
#        test_data,                      # specify the test set
#        cost_function,                  # specify the cost function to calculate error
#        ERROR_LIMIT          = 1e-3,    # define an acceptable error limit
#        #max_iterations      = (),      # continues until the error limit is reach if this argument is skipped
#        
#        # optional parameters
#        print_rate           = 1000,    # print error status every `print_rate` epoch.
#        weight_step_max      = 50., 
#        weight_step_min      = 0., 
#        start_step           = 0.5, 
#        learn_max            = 1.2, 
#        learn_min            = 0.5,
#        save_trained_network = False    # Whether to write the trained weights to disk
#    )


# Print a network test
print_test( network, training_data, cost_function )


"""
Prediction Example
"""
prediction_set = [ Instance([0,1]), Instance([1,0]) ]
prediction_set = preprocess( prediction_set )
print network.predict( prediction_set ) # produce the output signal