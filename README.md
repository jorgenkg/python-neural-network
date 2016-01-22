# Neural network written in Python (NumPy)
This is an implementation of a fully connected neural network in NumPy. The network can be trained by a variety of learning algorithms: backpropagation, resilient backpropagation and scaled conjugate gradient learning. By implementing a matrix approach, the NumPy implementation is able to harvest the power of the BLAS library and efficiently perform the required calculations. 

*The code has been tested.*

### Requirements
 * Python
 * NumPy

This script has been written with PYPY in mind. Use their [jit-compiler](http://pypy.org/download.html) to run this code blazingly fast.

### Learning algorithms
* Backpropagation
* Resilient backpropagation
* Scaled Conjugate Gradient

The latter two algorithms are hard to come by as Python implementations. Feel free to take a look at them if you intend to implement them yourself.

## How-to
To run the code, navigate into the project folder and execute the following command in the terminal:

`$ python main.py`

To train the network on a custom dataset, you will have to alter the dataset specified in the `main.py` file. It is quite self-explanatory.

```Python
# training set  Instance( [inputs], [targets] )
trainingset = [ Instance( [0,0], [0] ), Instance( [0,1], [1] ), Instance( [1,0], [1] ), Instance( [1,1], [0] ) ]

settings    = {
    # Required settings
    "n_inputs"              : 13,        # Number of network input signals
    "layers"                : [ (3, tanh_function), (3, sigmoid_function) ],
                                        # [ (number_of_neurons, activation_function) ]
                                        # The last pair in your list describes the number of output signals
    
    # Optional settings
    "weights_low"           : -0.1,     # Lower bound on initial weight range
    "weights_high"          : 0.1,      # Upper bound on initial weight range
    "save_trained_network"  : False,    # Whether to write the trained weights to disk
    
    "input_layer_dropout"   : 0.2,      # dropout fraction of the input layer
    "hidden_layer_dropout"  : 0.5,      # dropout fraction in all hidden layers
}

# initialize your neural network
network = NeuralNet( settings )

# save the trained network
network.save_to_file( "trained_configuration.pkl" )

# load a stored network configuration
# network = NeuralNet.load_from_file( "trained_configuration.pkl" )

# start training on test set one with scaled conjugate gradient
network.scg(
                training_one, 
                ERROR_LIMIT = 1e-4
            )

# start training on test set one with backpropagation
network.backpropagation( 
                training_wine,           # specify the training set
                ERROR_LIMIT     = 1e-3,  # define an acceptable error limit 
                learning_rate   = 0.03,  # learning rate
                momentum_factor = 0.45,   # momentum
                #max_iterations  = 100,  # continues until the error limit is reach if this argument is skipped
            )

# start training on test set one with backpropagation
network.resilient_backpropagation( 
                training_one,          # specify the training set
                ERROR_LIMIT     = 1e-3, # define an acceptable error limit
                #max_iterations = (),   # continues until the error limit is reach if this argument is skipped
                
                # optional parameters
                weight_step_max = 50., 
                weight_step_min = 0., 
                start_step = 0.5, 
                learn_max = 1.2, 
                learn_min = 0.5
            )
```

## Features:
 * Implemented with matrix operation to improve performance.
 * Dropout to reduce overfitting ([as desribed here](http://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf)). Note that dropout should only be used when using backpropagation.
 * PYPY friendly (requires pypy-numpy).

## Activation functions:
 * tanh
 * Sigmoid
 * Elliot function (fast sigmoid approximation)
 * Symmetric Elliot function (fast tanh approximation) 
 * Rectified Linear Unit
 * Linear activation

## Save and load learned weights
If the setting `save` is initialized `True`, the network will prompt you whether to store the weights after the training has succeeded.
