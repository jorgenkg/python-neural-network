Activation Functions
=======================

The some of the most popular activation functions has already been implemented in nimblenet. However, it is very easy to specify your own activation function as described in :ref:`arbirary-activation-functions`.

.. contents::
   :local:
   :depth: 2
   :backlinks: none

Usage
*****

Using the various activation functions is as easy as importing the desired activation function and using it when declaring the network topology. Below is an example of how to use the Sigmoid activation function in a simple neural network.

.. code:: python

    from nimblenet.activation_functions import sigmoid_function
    from nimblenet.neuralnet import NeuralNet

    settings       = {
        "n_inputs" : 2,                         # Two input signals
        
        # Using the sigmoid activation function in a layer
        "layers"   : [  (1, sigmoid_function) ] # A single layer neural network with one output signal
    }

    network        = NeuralNet( settings )

A network may of course use different activation functions at each layer:


.. code:: python

    from nimblenet.activation_functions import sigmoid_function, tanh_function
    from nimblenet.neuralnet import NeuralNet

    settings       = {
        "n_inputs" : 2,                         # Two input signals
        
        # Using both tanh and sigmoid activation functions
        "layers"   : [  (2, tanh_function), (1, sigmoid_function) ] # A two layered neural network with one output signal
    }

    network        = NeuralNet( settings )


List of cost functions
**********************

Sigmoid function
----------------------------

.. code:: python

    from nimblenet.activation_functions import sigmoid_function


Tanh function
----------------------------

.. code:: python

    from nimblenet.activation_functions import tanh_function


Softmax function
----------------------------

.. code:: python

    from nimblenet.activation_functions import softmax_function


Elliot function
----------------------------

The Elliot function is a fast approximation to the Sigmoid activation function.

.. code:: python

    from nimblenet.activation_functions import elliot_function


Symmetric Elliot function
----------------------------

The Symmetric Elliot function is a fast approximation to the tanh activation function.

.. code:: python

    from nimblenet.activation_functions import symmetric_elliot_function


ReLU function
----------------------------

.. code:: python

    from nimblenet.activation_functions import ReLU_function


LReLU function
----------------------------

This is the leaky rectified linear activation function.

.. code:: python

    from nimblenet.activation_functions import LReLU_function


Linear function
----------------------------

.. code:: python

    from nimblenet.activation_functions import linear_function


Softplus function
----------------------------

.. code:: python

    from nimblenet.activation_functions import softplus_function


Softsign function
----------------------------

.. code:: python

    from nimblenet.activation_functions import softsign_function



.. _arbirary-activation-functions:

Arbitrary Activation Functions
******************************

It is easy to write your own, custom activation functions. A activation function takes the required form:

.. code:: python

    def activation_function( signal, derivative = False ):
        ...

The ``signal`` parameter is a NumPy matrix with shape ``[n_samples, n_outputs]``. When the ``derivative`` flag is true, the activation function is expected to return the partial derivation of the function.

As an example, we can look at how the tanh activation function is implemented:

.. code:: python

    def tanh_function( signal, derivative=False ):
        squashed_signal = np.tanh( signal )
    
        if derivative:
            return 1 - np.power( squashed_signal, 2 )
        else:
            return squashed_signal

How to
------


Lets define a custom cost function and use it when training the network:

.. code:: python
    
    from nimblenet.learning_algorithms import backpropagation
    from nimblenet.cost_functions import sum_squared_error
    from nimblenet.data_structures import Instance
    from nimblenet.neuralnet import NeuralNet
    import numpy as np
    
    def custom_activation_function( signal, derivative = False ):
        # This activation function amounts to a ReLU layer
        if derivative:
            return (signal > 0).astype(float)
        else:
            return np.maximum( 0, signal )
    #end
    
    dataset        = [ Instance( [0,0], [0] ), Instance( [1,0], [1] ), Instance( [0,1], [1] ), Instance( [1,1], [1] )]
    settings       = {
        "n_inputs" : 2,
        
        # This is where we apply our custom activation function:
        "layers"   : [  (2, custom_activation_function) ]
    }

    network        = NeuralNet( settings )
    training_set   = dataset
    test_set       = dataset
    cost_function  = sum_squared_error
    
    backpropagation(
            network,              # the network to train
            training_set,         # specify the training set
            test_set,             # specify the test set
            cost_function         # specify the cost function to optimize
        )