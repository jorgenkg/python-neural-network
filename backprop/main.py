from activation_functions import sigmoid_function, tanh_function, linear_function
from neuralnet import NeuralNet
import numpy as np


class Instance:
    def __init__(self, features, target):
        self.features = np.matrix(features)
        self.target = target
#end Instance


# training set
training_one =  [ Instance( [0,0], [0] ), Instance( [0,1], [1] ), Instance( [1,0], [1] ), Instance( [1,1], [0] ) ]
training_two =  [ Instance( [0,0], [0,0] ), Instance( [0,1], [1,1] ), Instance( [1,0], [1,1] ), Instance( [1,1], [0,0] ) ]



n_inputs = 2
n_outputs = 1
n_hiddens = 6
n_hidden_layers = 1

# specify activation functions per layer
activation_functions = [ tanh_function ]*n_hidden_layers + [ linear_function ]

# initialize your neural network
network = NeuralNet(n_inputs, n_outputs, n_hiddens, n_hidden_layers, activation_functions)

# start training
network.backpropagation(training_one, ERROR_LIMIT=1e-3)


for instance in training_one:
    print instance.features, network.update( instance.features ), "\ttarget:", instance.target