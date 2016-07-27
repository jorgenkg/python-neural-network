Cost Functions
=======================

A the most popular and applicable cost functions has already been implemented in this library, and are listed below. However, it is very easy to specify your own cost functions as described in :ref:`arbirary-cost-functions`.

.. contents::
   :local:
   :backlinks: none

.. warning::

    The Softmax Categorical Cross Entropy cost function is required when using a softmax layer in the network topology.


Usage
*****

Using the various cost functions is as easy as only importing the desired cost function and passing it to the decided learning function. Below is an example of how to use the Cross Entropy cost function when training using the vanilla backpropagation algorithm.

.. code:: python

    from nimblenet.cost_functions import binary_cross_entropy_cost
    from nimblenet.activation_functions import sigmoid_function
    from nimblenet.learning_algorithms import backpropagation
    from nimblenet.data_structures import Instance
    from nimblenet.neuralnet import NeuralNet

    dataset        = [ Instance( [0,0], [0] ), Instance( [1,0], [1] ), Instance( [0,1], [1] ), Instance( [1,1], [1] )]
    settings       = {
        "n_inputs" : 2,
        "layers"   : [  (2, sigmoid_function) ]
    }

    network        = NeuralNet( settings )
    training_set   = dataset
    test_set       = dataset
    cost_function  = binary_cross_entropy_cost
    
    backpropagation(
            network,              # the network to train
            training_set,         # specify the training set
            test_set,             # specify the test set
            
            # This is where we specify the cost function to optimize:
            cost_function         # specify the cost function to calculate error
        )

List of cost functions
**********************

.. contents::
   :local:
   :depth: 2
   :backlinks: none

   
Sum Squared Error
----------------------------------

.. code:: python

    from nimblenet.cost_functions import sum_squared_error


Binary Cross Entropy
----------------------------------

.. code:: python

    from nimblenet.cost_functions import binary_cross_entropy_cost


Softmax Categorical Cross Entropy
----------------------------------

This cost function is **required** when including a softmax layer in your network topology.

.. code:: python

    from nimblenet.cost_functions import softmax_categorical_cross_entropy_cost



Hellinger Distance
----------------------------------

.. code:: python

    from nimblenet.cost_functions import hellinger_distance



.. _arbirary-cost-functions:

Arbitrary Cost Functions
*****************************

It is easy to optimize your own, custom cost functions. A cost function has the required form:

.. code:: python

    def custom_cost_function( 
                outputs,            # the signal emitted from the network
                targets,            # the target values we would like the network to output
                derivative = False  # whether the cost function should return its derivative
            ):
        ...

The ``outputs`` and ``targets`` parameters are NumPy matrices with shape ``[n_samples, n_outputs]``.

As an example, we can look at how the Sum Squared Error function is implemented:

.. code:: python

    def sum_squared_error( outputs, targets, derivative = False ):
        if derivative:
            return outputs - targets
        else:
            return 0.5 * np.mean(np.sum( np.power(outputs - targets,2), axis = 1 ))

.. important::

    Observe that we calculate the mean of the error, per singal, across the input instances fed into the network. This detail is important to remember in order to get the derivatives correct.

How to
------


Lets define a custom cost function and use it when training the network:

.. code:: python

    from nimblenet.activation_functions import sigmoid_function
    from nimblenet.learning_algorithms import backpropagation
    from nimblenet.data_structures import Instance
    from nimblenet.neuralnet import NeuralNet
    import numpy as np
    
    def custom_cost_function( outputs, targets, derivative = False ):
        if derivative:
            return outputs - targets
        else:
            return 0.5 * np.mean(np.sum( np.power(outputs - targets,2), axis = 1 ))
    #end 
    
    dataset        = [ Instance( [0,0], [0] ), Instance( [1,0], [1] ), Instance( [0,1], [1] ), Instance( [1,1], [1] )]
    settings       = {
        "n_inputs" : 2,
        "layers"   : [  (2, sigmoid_function) ]
    }

    network        = NeuralNet( settings )
    training_set   = dataset
    test_set       = dataset
    cost_function  = custom_cost_function
    
    backpropagation(
            network,              # the network to train
            training_set,         # specify the training set
            test_set,             # specify the test set
            
            # This is where we specify the cost function to optimize:
            cost_function         # specify the cost function to calculate error
        )