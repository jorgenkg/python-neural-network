# Neural network written in Python (NumPy)

This is an implementation of a fully connected neural network in NumPy. The network can be trained by a variety of learning algorithms: backpropagation, resilient backpropagation and scaled conjugate gradient learning. By implementing a matrix approach, the NumPy implementation is able to harvest the power of the BLAS library and efficiently perform the required calculations. 

*The code has been tested.*

#### [Visit the project page here.](http://jorgenkg.github.io/python-neural-network/)

<br>

### Requirements

-  Python
-  NumPy

This script has been written with PYPY in mind. Use their [jit-compiler](http://pypy.org/download.html) to run this code blazingly fast.

### Learning algorithms

-  SciPy's `minimize()`
-  Backpropagation
-  Resilient backpropagation
-  Scaled Conjugate Gradient

The latter two algorithms are hard to come by as Python implementations. Feel free to take a look at them if you intend to implement them yourself.

## Usage

To run the code, navigate into the project folder and execute the following command in the terminal:

`$ python main.py`

To train the network on a custom dataset, you will have to alter the dataset specified in the `main.py` file. It is quite self-explanatory.

``` Python
# training set  Instance( [inputs], [targets] )
dataset             = [ Instance( [0,0], [0] ), Instance( [1,0], [1] ), Instance( [0,1], [1] ), Instance( [1,1], [0] ) ]
preprocessor        = construct_preprocessor( dataset, [replace_nan, standarize] )
training_data       = preprocessor( dataset ) # using the same data for the training and 
test_data           = preprocessor( dataset ) # test set


settings    = {
    # Required settings
    "n_inputs"              : 2,        # Number of network input signals
    "layers"                : [ (2, tanh_function), (1, sigmoid_function) ],
                                        # [ (number_of_neurons, activation_function) ]
                                        # The last pair in the list dictate the number of output signals

    # Optional settings
    "weights_low"           : -0.1,     # Lower bound on initial weight range
    "weights_high"          : 0.1,      # Upper bound on initial weight range
}

# initialize your neural network
network = NeuralNet( settings )

# load a stored network configuration
# network = NeuralNet.load_network_from_file( "trained_configuration.pkl" )

# Choose a cost function
cost_function = cross_entropy_cost

# Perform a numerical gradient check
network.check_gradient( training_data, cost_function )

# start training on test set one with scaled conjugate gradient
scaled_conjugate_gradient(
        network,
        training_data,                  # specify the training set
        test_data,                      # specify the test set
        cost_function,                  # specify the cost function to calculate error
        ERROR_LIMIT          = 1e-4,    # define an acceptable error limit 
        save_trained_network = False    # Whether to write the trained weights to disk
    )

# start training on test set one with backpropagation
backpropagation(
        network,                        # the network to train
        training_data,                  # specify the training set
        test_data,                      # specify the test set
        cost_function,                  # specify the cost function to calculate error
        ERROR_LIMIT          = 1e-3,    # define an acceptable error limit 
        #max_iterations      = 100,     # continues until the error limit is reach if this argument is skipped
                    
        # optional parameters
        learning_rate        = 0.3,     # learning rate
        momentum_factor      = 0.9,     # momentum
        input_layer_dropout  = 0.0,     # dropout fraction of the input layer
        hidden_layer_dropout = 0.0,     # dropout fraction in all hidden layers
        save_trained_network = False    # Whether to write the trained weights to disk
    )

# start training on test set one with backpropagation
resilient_backpropagation(
        network,
        training_data,                  # specify the training set
        test_data,                      # specify the test set
        cost_function,                  # specify the cost function to calculate error
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

# start training on test set one with SciPy
scipyoptimize(
        network,
        training_data,                      # specify the training set
        test_data,                          # specify the test set
        cost_function,                      # specify the cost function to calculate error
        method               = "L-BFGS-B",
        save_trained_network = False        # Whether to write the trained weights to disk
    )
```

## Features:

-  Implemented with matrix operation ensure high performance.
-  Dropout to reduce overfitting ([as desribed here](http://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf)). Note that dropout should only be used when using backpropagation.
-  PYPY friendly (requires pypy-numpy).
-  Features a selection of cost functions (error functions) and activation functions

## Activation functions:

-  tanh
-  Sigmoid
-  Softmax
-  Elliot function (fast Sigmoid approximation)
-  Symmetric Elliot function (fast tanh approximation) 
-  Rectified Linear Unit (and Leaky Rectified Linear Unit)
-  Linear activation

## Cost functions

-  Sum squared error (the quadratic cost function)
-  Cross entropy cost function
-  Hellinger distance
-  Softmax log loss (required for Softmax output layers)

