from activation_functions import sigmoid_function, tanh_function, linear_function,\
                                 LReLU_function, ReLU_function, elliot_function, symmetric_elliot_function
from neuralnet import NeuralNet
from tools import Instance, load_wine_data
import numpy as np


# Training sets
training_one    = [ Instance( [0,0], [0] ), Instance( [0,1], [1] ), Instance( [1,0], [1] ), Instance( [1,1], [0] ) ]

training_wine   = load_wine_data() # Loads the dataset from UCI Machine Learning Repository http://archive.ics.uci.edu/ml/datasets/Wine


settings = {
    # Required settings
    "n_inputs"              : 13,        # Number of network input signals
    "layers"                : [ (3, tanh_function), (3, sigmoid_function) ],
                                        # [ (number_of_neurons, activation_function) ]
                                        # The last pair in you list describes the number of output signals
    
    # Optional settings
    "weights_low"           : -0.1,     # Lower bound on initial weight range
    "weights_high"          : 0.1,      # Upper bound on initial weight range
    "save_trained_network"  : False,    # Whether to write the trained weights to disk
    
    "input_layer_dropout"   : 0.2,      # dropout fraction of the input layer
    "hidden_layer_dropout"  : 0.5,      # dropout fraction in all hidden layers
}


# initialize the neural network
network = NeuralNet( settings )

# load a stored network configuration
# network = NeuralNet.load_from_file( "trained_configuration.pkl" )


# start training on test set one
network.backpropagation( 
                training_wine,           # specify the training set
                ERROR_LIMIT     = 1e-3,  # define an acceptable error limit 
                learning_rate   = 0.03,  # learning rate
                momentum_factor = 0.45,   # momentum
                #max_iterations  = 100,  # continues until the error limit is reach if this argument is skipped
            )

print "Final MSE:", network.test( training_wine )