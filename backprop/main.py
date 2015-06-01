from activation_functions import sigmoid_function, tanh_function, linear_function,\
                                 ReLU_function, symmetric_elliot_function, elliot_function
from neuralnet import NeuralNet
import numpy as np


class Instance:
    def __init__(self, features, target):
        self.features = np.array(features)
        self.targets = np.array(target)
#end Instance


# training set
training_one =  [ Instance( [0,0], [0] ), Instance( [0,1], [1] ), Instance( [1,0], [1] ), Instance( [1,1], [0] ) ]
training_two =  [ Instance( [0,0], [0,0] ), Instance( [0,1], [1,1] ), Instance( [1,0], [1,1] ), Instance( [1,1], [0,0] ) ]

n_inputs = 2
n_outputs = 1
n_hiddens = 2
n_hidden_layers = 1

# specify activation functions per layer
activation_functions = [ tanh_function ]*n_hidden_layers + [ sigmoid_function ]

# initialize your neural network
network = NeuralNet(n_inputs, n_outputs, n_hiddens, n_hidden_layers, activation_functions)

# start training
network.backpropagation(training_one, ERROR_LIMIT=1e-4)


for instance in training_one:
    print instance.features, network.update( np.array([instance.features]) ), "\ttarget:", instance.targets