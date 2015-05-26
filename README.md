# Neural network written in Python (NumPy)
This is an implementation of a fully connected neural network in NumPy. The network may be batch trained by backpropagation. By implementing a batch approach, the NumPy implementation is able to harvest the power of the BLAS library to efficiently perform the required calculations. 

*The code has been tested.*

## Requirements
 * Python
 * NumPy

This script has been written with PYPY in mind. Use their [jit-compiler](http://pypy.org/download.html) to run this code blazingly fast.

## How-to
To run the code, navigate into the project folder and execute the following command in the terminal:

`$ python main.py`

To train the network on a custom dataset, you will have to alter the dataset specified in the `main.py` file. It is quite self-explanatory.

```Python
# training set  Instance( [inputs], [targets] )
trainingset          = [ Instance( [0,0], [0] ), Instance( [0,1], [1] ), Instance( [1,0], [1] ), Instance( [1,1], [0] ) ]

n_inputs             = 2 # Number of inputs to the network
n_outputs            = 1 # Number of outputs in the output layer
n_hiddens            = 2 # Number of nodes in the hidden layers
n_hidden_layers      = 1 # Number of hidden layers

# specify activation functions per layer
activation_functions = [ tanh_function ]*n_hidden_layers + [ sigmoid_function ]

# initialize your neural network
network              = NeuralNet(n_inputs, n_outputs, n_hiddens, n_hidden_layers, activation_functions)


# start training
network.backpropagation(trainingset, ERROR_LIMIT = 1e-4, learning_rate=0.3, momentum_factor=0.9 )
```

## Features:
 * Implemented with matrix operation to improve performance.
 * PYPY friendly (requires pypy-numpy).

## Activation functions:
 * tanh
 * Symmetric Elliot function (fast tanh approximation) 
 * Sigmoid
 * Rectified Linear Unit
 * Linear activation
