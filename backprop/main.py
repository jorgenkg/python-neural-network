from activation_functions import sigmoid_function, tanh_function, linear_function,\
                                 LReLU_function, ReLU_function, elliot_function, symmetric_elliot_function
from neuralnet import NeuralNet
import numpy as np


class Instance:
    # This is a simple encapsulation of a `input signal : output signal`
    # pair in our training set.
    def __init__(self, features, target):
        self.features = np.array(features)
        self.targets  = np.array(target)
#endclass Instance


# two training sets
training_one    = [ Instance( [0,0], [0] ), Instance( [0,1], [1] ), Instance( [1,0], [1] ), Instance( [1,1], [0] ) ]
training_two    = [ Instance( [0,0], [0,0] ), Instance( [0,1], [1,1] ), Instance( [1,0], [1,1] ), Instance( [1,1], [0,0] ) ]


n_inputs        = 2
n_outputs       = 1
n_hidden_layers = 1 # number of hidden layers
n_hiddens       = 2 # number of nodes in each hidden layer



# specify activation functions per layer eg: [ hidden_layer_1, hidden_layer_2, output_layer ]
activation_functions = [ ReLU_function ] * n_hidden_layers + [ sigmoid_function ]


# initialize the neural network
network = NeuralNet( n_inputs, n_outputs, n_hiddens, n_hidden_layers, activation_functions )


# start training on test set one
network.backpropagation( 
                training_one,           # specify the training set
                ERROR_LIMIT = 1e-6,     # define an acceptable error limit 
                learning_rate = 0.3,    #
                momentum_factor = 0.9   #
            )


# save the trained network
network.save_to_file( "trained_configuration.pkl" )


# load a stored network configuration
# network = NeuralNet.load_from_file( "trained_configuration.pkl" )


# Test the network by looping through the specified dataset and print the results.
for instance in training_one:
    print "Input: {features} -> Output: {output} \t| target: {target}".format( 
                features = str(instance.features), 
                output   = str(network.update( np.array([instance.features]) )), 
                target   = str(instance.targets)
            )