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

# initialize your neural network
network              = NeuralNet( settings )

# save the trained network
network.save_to_file( "trained_configuration.pkl" )

# load a stored network configuration
# network = NeuralNet.load_from_file( "trained_configuration.pkl" )

# start training
network.backpropagation( 
                trainingset,           # specify the training set
                ERROR_LIMIT     = 1e-6, # define an acceptable error limit 
                learning_rate   = 0.03, # learning rate
                momentum_factor = 0.95  # momentum
            )
```

## Features:
 * Implemented with matrix operation to improve performance.
 * Batch learning
 * Dropout to reduce overfitting ([as desribed here](http://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf))
 * PYPY friendly (requires pypy-numpy).

## Activation functions:
 * tanh
 * Sigmoid
 * Elliot function (fast sigmoid approximation)
 * Symmetric Elliot function (fast tanh approximation) 
 * Rectified Linear Unit
 * Linear activation

## Save and load learned weights
```Python
network.save_to_file() # store a trained network
network = NeuralNet.load_from_file() # load a trained network
```
