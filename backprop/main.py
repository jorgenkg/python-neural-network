from activation_functions import sigmoid_function, tanh_function, linear_function,\
                                 LReLU_function, ReLU_function, elliot_function, symmetric_elliot_function
from neuralnet import NeuralNet
from tools import Instance
import numpy as np


# two training sets
training_one    = [ Instance( [0,0], [0] ), Instance( [0,1], [1] ), Instance( [1,0], [1] ), Instance( [1,1], [0] ) ]
training_two    = [ Instance( [0,0], [0,0] ), Instance( [0,1], [1,1] ), Instance( [1,0], [1,1] ), Instance( [1,1], [0,0] ) ]


settings = {
    # Required settings
    "n_inputs"              : 2,        # Number of network input signals
    "n_outputs"             : 1,        # Number of desired outputs from the network
    "n_hidden_layers"       : 1,        # Number of nodes in each hidden layer
    "n_hiddens"             : 2,        # Number of hidden layers in the network
    "activation_functions"  : [ tanh_function, sigmoid_function ], # specify activation functions per layer eg: [ hidden_layer, output_layer ]
    
    # Optional settings
    "weights_low"           : -0.1,     # Lower bound on initial weight range
    "weights_high"          : 0.1,      # Upper bound on initial weight range
    "save_trained_network"  : False,    # Whether to write the trained weights to disk
    
    "input_layer_dropout"   : 0.0,      # dropout fraction of the input layer
    "hidden_layer_dropout"  : 0.1,      # dropout fraction in all hidden layers
    
    "batch_size"            : 0,        # 1 := online learning, 0 := entire trainingset as batch, else := batch learning size
}


# initialize the neural network
network = NeuralNet( settings )

# load a stored network configuration
# network = NeuralNet.load_from_file( "trained_configuration.pkl" )


# start training on test set one
network.backpropagation( 
                training_one,           # specify the training set
                ERROR_LIMIT     = 1e-6, # define an acceptable error limit 
                learning_rate   = 0.03, # learning rate
                momentum_factor = 0.95  # momentum
            )


# Test the network by looping through the specified dataset and print the results.
for instance in training_one:
    print "Input: {features} -> Output: {output} \t| target: {target}".format( 
                features = str(instance.features), 
                output   = str(network.update( np.array([instance.features]) )), 
                target   = str(instance.targets)
            )


if settings.get("save_trained_network", False):
    # save the trained network
    network.save_to_file( "trained_configuration.pkl" )