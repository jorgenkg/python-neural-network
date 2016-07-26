# Neural network written in Python (NumPy)

This is an implementation of a fully connected neural network in NumPy. The network can be trained by a variety of learning algorithms: backpropagation, resilient backpropagation and scaled conjugate gradient learning. By implementing a matrix approach, the NumPy implementation is able to harvest the power of the BLAS library and efficiently perform the required calculations. 

*The code has been tested.*

#### [Visit the project page here.](http://jorgenkg.github.io/python-neural-network/)
#### [Read the documentation here.](https://nimblenet.readthedocs.io/en/latest/index.html)

<br>

## Installation

`pip install nimblenet`

## Requirements

-  Python
-  NumPy
-  Optionally: SciPy

This script has been written with PYPY in mind. Use their [jit-compiler](http://pypy.org/download.html) to run this code blazingly fast.


## Features:

-  Implemented with matrix operation ensure high performance.
-  Dropout to reduce overfitting ([as desribed here](http://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf)). Note that dropout can only be applied to backpropagation.
-  PYPY friendly (requires pypy-numpy).
-  Features a selection of cost functions (error functions) and activation functions



## Example Usage


    from nimblenet.activation_functions import sigmoid_function
    from nimblenet.cost_functions import cross_entropy_cost
    from nimblenet.learning_algorithms import RMSprop
    from nimblenet.data_structures import Instance
    from nimblenet.neuralnet import NeuralNet


    dataset        = [
        Instance( [0,0], [0] ), Instance( [1,0], [1] ), Instance( [0,1], [1] ), Instance( [1,1], [0] )
    ]

    settings       = {
        "n_inputs" : 2,
        "layers"   : [  (5, sigmoid_function), (1, sigmoid_function) ]
    }

    network        = NeuralNet( settings )
    training_set   = dataset
    test_set       = dataset
    cost_function  = cross_entropy_cost


    RMSprop(
            network,                            # the network to train
            training_set,                      # specify the training set
            test_set,                          # specify the test set
            cost_function,                      # specify the cost function to calculate error

            ERROR_LIMIT             = 1e-2,     # define an acceptable error limit
            #max_iterations         = 100,      # continues until the error limit is reach if this argument is skipped
        )


