# Neural network written in Python (NumPy)

This is an implementation of a fully connected neural network in NumPy. The network can be trained by a variety of learning algorithms: backpropagation, resilient backpropagation and scaled conjugate gradient learning. By implementing a matrix approach, the NumPy implementation is able to harvest the power of the BLAS library and efficiently perform the required calculations. 

*The code has been tested.*

#### [Visit the project page here.](http://jorgenkg.github.io/python-neural-network/)

<br>

### Installation

- Install with pip: `pip install nimblenet`
- Clone from Github

### Requirements

-  Python
-  NumPy
-  Optionally: SciPy

This script has been written with PYPY in mind. Use their [jit-compiler](http://pypy.org/download.html) to run this code blazingly fast.

### Learning algorithms

-  SciPy's `minimize()`
-  Backpropagation
-  Resilient backpropagation
-  Scaled Conjugate Gradient

The latter two algorithms are hard to come by as Python implementations. Feel free to take a look at them if you intend to implement them yourself.

## Usage
A walkthrough on how to use the library is provided on [the project page](http://jorgenkg.github.io/python-neural-network/). 
Please refer to the example below to get the gist of how to use the code:

```python
from nimblenet.activation_functions import tanh_function, softmax_function
from nimblenet.learning_algorithms  import resilient_backpropagation
from nimblenet.cost_functions  import softmax_neg_loss
from nimblenet.data_structures import Instance
from nimblenet.neuralnet import NeuralNet
from nimblenet.tools import print_test


dataset             = [ Instance( [0,0], [0,1] ), Instance( [1,0], [1,0] ), Instance( [0,1], [1,0] ), Instance( [1,1], [0,1] ) ]
cost_function       = softmax_neg_loss
settings            = {
    # Required settings
    "n_inputs"              : 2,       # Number of network input signals
    "layers"                : [  (5, tanh_function), (1, softmax_function) ],
                                        # [ (number_of_neurons, activation_function) ]
                                        # The last pair in the list dictate the number of output signals
    
    # Optional settings
    "weights_low"           : -0.1,     # Lower bound on the initial weight value
    "weights_high"          : 0.1,      # Upper bound on the initial weight value
}

# Initialize the neural network
network             = NeuralNet( settings )

# Perform a numerical gradient check
network.check_gradient( training_data, cost_function )

# Train the network using resilient backpropagation
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

print_test( network, training_data, cost_function )
```

## Features:

-  Implemented with matrix operation ensure high performance.
-  Dropout to reduce overfitting ([as desribed here](http://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf)). Note that dropout can only be applied to backpropagation.
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
-  Softplus function
-  Softsign function

## Cost functions

-  Sum squared error
-  Hellinger distance
-  Binary cross entropy cost function
-  Softmax categorical cross entropy cost function (required for Softmax output layers)

## Evaluation functions

-  Binary accuracy
-  Categorical accuracy

