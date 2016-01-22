from activation_functions import sigmoid_function, tanh_function, linear_function,\
                                 LReLU_function, ReLU_function, elliot_function, symmetric_elliot_function
from neuralnet import NeuralNet
from tools import Instance
import numpy as np


# Training sets
training_one    = [ Instance( [0,0], [0] ), Instance( [0,1], [1] ), Instance( [1,0], [1] ), Instance( [1,1], [0] ) ]

settings = {
    # Required settings
    "n_inputs"              : 2,       # Number of network input signals
    "layers"                : [ (3, tanh_function), (1, sigmoid_function) ],
                                        # [ (number_of_neurons, activation_function) ]
                                        # The last pair in you list describes the number of output signals
    
    # Optional settings
    "weights_low"           : -0.1,     # Lower bound on initial weight range
    "weights_high"          : 0.1,      # Upper bound on initial weight range
    "save_trained_network"  : False,    # Whether to write the trained weights to disk
    
    "input_layer_dropout"   : 0.0,      # dropout fraction of the input layer
    "hidden_layer_dropout"  : 0.0,      # dropout fraction in all hidden layers
}


# initialize the neural network
network = NeuralNet( settings )

# load a stored network configuration
# network = NeuralNet.load_from_file( "trained_configuration.pkl" )

# Train the network using Scaled Conjugate Gradient
network.scg(
                training_one, 
                ERROR_LIMIT = 1e-4
            )

# Train the network using backpropagation
#network.backpropagation( 
#                training_one,          # specify the training set
#                ERROR_LIMIT     = 1e-3, # define an acceptable error limit 
#                #max_iterations  = 100, # continues until the error limit is reach if this argument is skipped
#
#                # optional parameters
#                learning_rate   = 0.03, # learning rate
#                momentum_factor = 0.45, # momentum
#            )

# Train the network using resilient backpropagation
#network.resilient_backpropagation( 
#                training_one,          # specify the training set
#                ERROR_LIMIT     = 1e-3, # define an acceptable error limit
#                #max_iterations = (),   # continues until the error limit is reach if this argument is skipped
#                
#                # optional parameters
#                weight_step_max = 50., 
#                weight_step_min = 0., 
#                start_step = 0.5, 
#                learn_max = 1.2, 
#                learn_min = 0.5
#            )

print "Final MSE:", network.test( training_one )