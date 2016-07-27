# Neural network written in Python (NumPy)

This is an implementation of a fully connected neural network in NumPy. By using the matrix approach to neural networks, this NumPy implementation is able to harvest the power of the BLAS library and efficiently perform the required calculations. The network can be trained by a wide range of learning algorithms.

[Visit the project page](http://jorgenkg.github.io/python-neural-network/) or [Read the documentation](https://nimblenet.readthedocs.io/en/latest/index.html).

*The code has been tested.*

## Implemented learning algorithms:

* Vanilla Backpropagation
* Backpropagation with classical momentum
* Backpropagation with Nesterov momentum
* RMSprop
* Adagrad
* Adam
* Resilient Backpropagation
* Scaled Conjugate Gradient
* SciPy’s Optimize

## Installation

```bash

pip install nimblenet

```

## Requirements

-  Python
-  NumPy
-  Optionally: SciPy

This script has been written with PYPY in mind. Use their [jit-compiler](http://pypy.org/download.html) to run this code blazingly fast.


## Features:

-  Implemented with matrix operations to ensure high performance.
-  Dropout regularization is available to reduce overfitting. [Implemented as desribed here](http://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf).
-  Martin Møller's *Scaled Conjugate Gradient for Fast Supervised Learning* as published [here](http://www.sciencedirect.com/science/article/pii/S0893608005800565).
-  PYPY friendly (requires pypy-numpy).
-  Features a selection of cost functions (error functions) and activation functions



## Example Usage

```python

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
        "layers"   : [  (2, sigmoid_function), (1, sigmoid_function) ]
    }

    network        = NeuralNet( settings )
    training_set   = dataset
    test_set       = dataset
    cost_function  = cross_entropy_cost


    RMSprop(
            network,           # the network to train
            training_set,      # specify the training set
            test_set,          # specify the test set
            cost_function,     # specify the cost function to calculate error
        )

```
