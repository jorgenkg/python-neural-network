.. _getting-started:

Getting Started
=====================================

This guide will walk you through how to install nimblenet and configure a network using the library. 

.. contents::
   :local:
   :backlinks: none


Installing
--------------------

.. code::

    $ pip install nimblenet

Required dependencies
^^^^^^^^^^^^^^^^^^^^^

* Python 2.7
* NumPy

Optional dependencies
^^^^^^^^^^^^^^^^^^^^^

* SciPy

In order to speed up the code when using the Sigmoid activation functions, the SciPy package should also be installed. This is an optional dependency, but it is of course required if you intend to train the network using SciPy's ``optimize`` function.


Creating a Network
---------------------

Once nimblenet has been installed, initializing a network is simple. The following example creates a two layered network that require two input signals. 

.. code:: python
    
    from nimblenet.activation_functions import sigmoid_function
    from nimblenet.neuralnet import NeuralNet

    settings            = {
        "n_inputs"      : 2,
        "layers"        : [  (3, sigmoid_function), (1, sigmoid_function) ]
    }
    
    network             = NeuralNet( settings )


The ``layers`` parameter describe the topology of the network. The first tuple state that the hidden layer should have three neurons and apply the sigmoid activation function. The final tuple in the ``layers`` list *always* describe the number of output signals. A list of built-in activations functions are listed in :doc:`activation_functions`.

.. important::
    
    The final tuple in the layers list always describe the number of output signals.

The properties specified in the settings parameter are *required*. The initialization of a network is further customizable, please refer to the page :doc:`initializing_network`.



Training the Network
---------------------

The network can be trained by a wide range of learning functions. In this quick intro, we will see how use RMSprop to fit the network to some training data.

First off, we need some dataset to fit the network to. In this guide, we will teach the network XOR. In nimblenet, a dataset is a list of `Instances`.

.. code:: python

    from nimblenet.data_structures import Instance
    dataset = [ 
        # Instance( [inputs], [outputs] )
        Instance( [0,0], [0] ), Instance( [1,0], [1] ), Instance( [0,1], [1] ), Instance( [1,1], [0] ) 
    ]
    
The dataset above consist of four training instances with two input signals and one output signal. In general we would split the dataset into a training set and a test set, but for the XOR problem we simply specify the training and test set to be identical:

.. code:: python

    training_set = dataset
    test_set = dataset
    

The nimblenet library also offers a selection of preprocessors to manipluate the data and make training more efficient. The preprocessors are not used in this guide, please refer to :doc:`preprocessing` instead.

Before fitting the network to some training data, we need to decide which cost function we would like to optimize. There are a few cost functions already implemented in this library, and this guide will use the *Cross Entropy* cost function. However, it is easy to implement your own custom cost functions. Please refer to :doc:`cost_functions`.

.. code:: python

    from nimblenet.cost_functions import cross_entropy_cost
    cost_function = cross_entropy_cost

Now that we've specified a cost function, we can use RSMprop to train our network:

.. code:: python

    from nimblenet.learning_algorithms import *
    RMSprop(
            network,                            # the network to train
            training_set,                      # specify the training set
            test_set,                          # specify the test set
            cost_function,                      # specify the cost function to calculate error
            
            ERROR_LIMIT             = 1e-2,     # define an acceptable error limit 
            #max_iterations         = 100,      # continues until the error limit is reach if this argument is skipped
        )

If the training shows poor progression, you may try to gradient check the network to verify that the numerical and the analytical gradient are similar. If the gradient check fails, the math might be wrong. Refer to gradient checking here: :doc:`gradient_checking`.

Using the Network
---------------------

After the training has completed, we can verify the training by forward propagating some input data in the network. Since the network is written using matrices, we can forward propagate multiple input instances at once. In contrast to the instances generated when training the network, these instance will only be created with a single parameter (the input signal). The following code tests the output of two instances:

.. code:: python

    prediction_set = [ Instance([0,1]), Instance([1,0]) ]
    print network.predict( prediction_set )
    >> [[ 0.99735413]
        [ 0.99735378]]

The prediction method returns a two dimensional NumPy list (shape = [n_samples, n_outputs]). The first dimension of the list contain the outputs from the corresponing Instance.


Putting it all together
------------------------

.. code:: python

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
