from activation_functions import sigmoid_function, tanh_function, linear_function,\
                                 LReLU_function, ReLU_function, elliot_function, symmetric_elliot_function, softmax_function, softplus_function, softsign_function
from cost_functions import sum_squared_error, cross_entropy_cost, hellinger_distance, softmax_neg_loss
from learning_algorithms import backpropagation, scaled_conjugate_gradient, scipyoptimize, resilient_backpropagation
from neuralnet import NeuralNet
from preprocessing import construct_preprocessor, standarize, replace_nan, whiten
from data_structures import Instance, Dataset
from tools import print_test


# Training sets
dataset             = [ Instance( [0,0], [0] ), Instance( [1,0], [1] ), Instance( [0,1], [1] ), Instance( [1,1], [0] ) ]
preprocessor        = construct_preprocessor( dataset, [replace_nan, standarize] ) # it is important to use the same preprocessor on ANY data that will be fed into the network. The order of the functions in the second parameter affect the produced processor.
training_data       = preprocessor( dataset )
test_data           = preprocessor( dataset )


cost_function       = cross_entropy_cost
settings            = {
    # Required settings
    "n_inputs"              : 2,       # Number of network input signals
    "layers"                : [  (5, tanh_function), (1, sigmoid_function) ],
                                        # [ (number_of_neurons, activation_function) ]
                                        # The last pair in the list dictate the number of output signals
    
    # Optional settings
    "weights_low"           : -0.1,     # Lower bound on the initial weight value
    "weights_high"          : 0.1,      # Upper bound on the initial weight value
}


# initialize the neural network
network             = NeuralNet( settings )
network.check_gradient( training_data, cost_function )



# load a stored network configuration
# network           = NeuralNet.load_network_from_file( "network0.pkl" )
#
#
# Train the network using backpropagation
#backpropagation(
#        network,                        # the network to train
#        training_data,                   # specify the training set
#        test_data,
#        cost_function,                  # specify the cost function to calculate error
#        ERROR_LIMIT          = 1e-3,    # define an acceptable error limit 
#        #max_iterations      = 100,     # continues until the error limit is reach if this argument is skipped
#                    
#        # optional parameters
#        learning_rate        = 0.3,    # learning rate
#        momentum_factor      = 0.9,     # momentum
#        input_layer_dropout  = 0.0,     # dropout fraction of the input layer
#        hidden_layer_dropout = 0.0,     # dropout fraction in all hidden layers
#        save_trained_network = False    # Whether to write the trained weights to disk
#    )
#
# Train the network using SciPy
#scipyoptimize(
#        network,
#        training_data,                   # specify the training set
#        test_data,
#        cost_function,
#        method               = "L-BFGS-B",
#        save_trained_network = False    # Whether to write the trained weights to disk
#    )
#
# Train the network using Scaled Conjugate Gradient
#scaled_conjugate_gradient(
#        network,
#        training_data,                   # specify the training set
#        test_data,
#        cost_function,
#        ERROR_LIMIT          = 1e-4,
#        save_trained_network = False    # Whether to write the trained weights to disk
#    )
#
# Train the network using resilient backpropagation
resilient_backpropagation(
        network,
        training_data,                   # specify the training set
        test_data,
        cost_function,                 
        ERROR_LIMIT          = 1e-3,    # define an acceptable error limit
        #max_iterations      = (),      # continues until the error limit is reach if this argument is skipped
        
        # optional parameters
        weight_step_max      = 50., 
        weight_step_min      = 0., 
        start_step           = 0.5, 
        learn_max            = 1.2, 
        learn_min            = 0.5,
        save_trained_network = False    # Whether to write the trained weights to disk
    )

print_test( network, training_data, cost_function )