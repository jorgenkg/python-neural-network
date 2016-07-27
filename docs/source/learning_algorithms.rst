
Learning Algorithms
=====================================

This library offers a wide range of analytical learning algorithms. These algorithms have been implemented in Python using NumPy and its matrices for efficiency:

.. contents::
   :local:
   :depth: 2
   :backlinks: none
 
To shorten the code examples given below, the following code snippet is implicitly called before executing the examples:

.. code:: python

    from nimblenet.activation_functions import sigmoid_function
    from nimblenet.cost_functions import cross_entropy_cost
    from nimblenet.data_structures import Instance
    from nimblenet.neuralnet import NeuralNet

    dataset        = [ 
        Instance( [0,0], [0] ), Instance( [1,0], [1] ), Instance( [0,1], [1] ), Instance( [1,1], [1] ) 
    ]

    settings       = {
        "n_inputs" : 2,
        "layers"   : [ (1, sigmoid_function) ]
    }

    network        = NeuralNet( settings )
    training_set   = dataset
    test_set       = dataset
    cost_function  = cross_entropy_cost


.. important::

    The *dropout* regularization strategy is applicable to all backpropagation variations and adaptive learning rate methods.
    The scaled conjugate gradient, SciPy's ``optimize``, and resilient backpropagation does not support this regularization.


Backpropagation Variations
**************************

These are the common parameters accepted by the following learning algorithms along with their default value:

.. _backprop-common-parameters:

.. code:: python

    learning_algorithm(
        # Required parameters
        network,                     # the neural network instance to train
        training_set,                # the training dataset
        test_set,                    # the test dataset
        cost_function,               # the cost function to optimize
                            
        # Optional parameters
        ERROR_LIMIT          = 1e-3, # Error tolerance when terminating the learning
        max_iterations       = (),   # Regardless of the achieved error, terminate after max_iterations epochs. Default: infinite
        batch_size           = 0,    # Set the batch size. 0 implies using the entire training_set as a batch, 1 equals no batch learning, and any other number dictate the batch size
        input_layer_dropout  = 0.0,  # Dropout fraction of the input layer
        hidden_layer_dropout = 0.0,  # Dropout fraction of in the hidden layer(s)
        print_rate           = 1000, # The epoch interval to print progression statistics
        save_trained_network = False # Whether to ask the user if they would like to save the network after training
    )


Vanilla Backpropagation
--------------------------------

This example show how to train your network using the vanilla backpropagation algorithm. The optional common parameters has been skipped for brevity, but the algorithm conforms to :ref:`common backpropagation variables <backprop-common-parameters>`.

.. code:: python
    
    from nimblenet.learning_algorithms import backpropagation
    backpropagation(
            # Required parameters
            network,                     # the neural network instance to train
            training_set,                # the training dataset
            test_set,                    # the test dataset
            cost_function,               # the cost function to optimize
        )


Classical Momentum
--------------------------------

This example show how to train your network using backpropagation with classical momentum. The optional common parameters has been skipped for brevity, but the algorithm conforms to :ref:`common backpropagation variables <backprop-common-parameters>`.

Named variables are shown together with their default value.

.. code:: python
    
    from nimblenet.learning_algorithms import backpropagation_classical_momentum
    backpropagation_classical_momentum(
            # Required parameters
            network,                     # the neural network instance to train
            training_set,                # the training dataset
            test_set,                    # the test dataset
            cost_function,               # the cost function to optimize
            
            # Classical momentum backpropagation specific, optional parameters
            momentum_factor = 0.9        
        )


Nesterov Momentum
--------------------------------

This example show how to train your network using backpropagation with Nesterov momentum. The optional common parameters has been skipped for brevity, but the algorithm conforms to :ref:`common backpropagation variables <backprop-common-parameters>`.

Named variables are shown together with their default value.

.. code:: python
    
    from nimblenet.learning_algorithms import backpropagation_nesterov_momentum
    backpropagation_nesterov_momentum(
            # Required parameters
            network,                     # the neural network instance to train
            training_set,                # the training dataset
            test_set,                    # the test dataset
            cost_function,               # the cost function to optimize
            
            # Nesterov momentum backpropagation specific, optional parameters
            momentum_factor = 0.9        
        )

RMSprop
--------------------------------

This example show how to train your network using RMSprop. The optional common parameters has been skipped for brevity, but the algorithm conforms to :ref:`common backpropagation variables <backprop-common-parameters>`.

Named variables are shown together with their default value.

.. code:: python
    
    from nimblenet.learning_algorithms import RMSprop
    RMSprop(
            # Required parameters
            network,                     # the neural network instance to train
            training_set,                # the training dataset
            test_set,                    # the test dataset
            cost_function,               # the cost function to optimize
            
            # RMSprop specific, optional parameters
            decay_rate  = 0.99, 
            epsilon     = 1e-8
        )


Adagrad
--------------------------------

This example show how to train your network using Adagrad. The optional common parameters has been skipped for brevity, but the algorithm conforms to :ref:`common backpropagation variables <backprop-common-parameters>`.

Named variables are shown together with their default value.

.. code:: python
    
    from nimblenet.learning_algorithms import adagrad
    adagrad(
            # Required parameters
            network,                     # the neural network instance to train
            training_set,                # the training dataset
            test_set,                    # the test dataset
            cost_function,               # the cost function to optimize
            
            # Adagrad specific, optional parameters
            epsilon     = 1e-8
        )


Adam
--------------------------------

This example show how to train your network using Adam. The optional common parameters has been skipped for brevity, but the algorithm conforms to :ref:`common backpropagation variables <backprop-common-parameters>`.

Named variables are shown together with their default value.

.. code:: python
    
    from nimblenet.learning_algorithms import Adam
    Adam(
            # Required parameters
            network,                     # the neural network instance to train
            training_set,                # the training dataset
            test_set,                    # the test dataset
            cost_function,               # the cost function to optimize
            
            # Adam specific, optional parameters
            beta1       = 0.9, 
            beta2       = 0.999, 
            epsilon     = 1e-8
        )



Additional Learning Algorithms
******************************


Resilient Backpropagation
--------------------------------

This example show how to train your network using resilient backpropagation. This is the iRprop+ variation of resilient backpropagation.

Named variables are shown together with their default value.

.. code:: python
    
    
    from nimblenet.learning_algorithms import resilient_backpropagation
    resilient_backpropagation(
        # Required parameters
        network,                     # the neural network instance to train
        training_set,                # the training dataset
        test_set,                    # the test dataset
        cost_function,               # the cost function to optimize
        
        # Resilient backpropagation specific, optional parameters
        weight_step_max      = 50.,
        weight_step_min      = 0., 
        start_step           = 0.5, 
        learn_max            = 1.2, 
        learn_min            = 0.5,
        
        # Optional parameters
        ERROR_LIMIT          = 1e-3, # Error tolerance when terminating the learning
        max_iterations       = (),   # Regardless of the achieved error, terminate after max_iterations epochs. Default: infinite
        print_rate           = 1000, # The epoch interval to print progression statistics
        save_trained_network = False # Whether to ask the user if they would like to save the network after training
    )


Scaled Conjugate Gradient
--------------------------------

This example show how to train your network using scaled conjugate gradient. This algorithm has been implemented according to `Scaled Conjugate Gradient for Fast Supervised Learning <http://www.sciencedirect.com/science/article/pii/S0893608005800565>`_ authored by Martin Møller.

Named variables are shown together with their default value.

.. code:: python
    
    from nimblenet.learning_algorithms import scaled_conjugate_gradient
    scaled_conjugate_gradient(
        # Required parameters
        network,                     # the neural network instance to train
        training_set,                # the training dataset
        test_set,                    # the test dataset
        cost_function,               # the cost function to optimize
        
        # Optional parameters
        ERROR_LIMIT          = 1e-3, # Error tolerance when terminating the learning
        max_iterations       = (),   # Regardless of the achieved error, terminate after max_iterations epochs. Default: infinite
        print_rate           = 1000, # The epoch interval to print progression statistics
        save_trained_network = False # Whether to ask the user if they would like to save the network after training
    )


SciPy's Optimize
--------------------------------

This example show how to train your network using SciPy's ``optimize`` function. This learning algorithm requires SciPy to be installed.

Named variables are shown together with their default value.

.. code:: python
    
    from nimblenet.learning_algorithms import scipyoptimize
    scipyoptimize(
        # Required parameters
        network,                     # the neural network instance to train
        training_set,                # the training dataset
        test_set,                    # the test dataset
        cost_function,               # the cost function to optimize
        
        # SciPy Optimize specific, optional parameters
        method               = "Newton-CG", # The method name correspond to the method names accepted by the SciPy optimize function. Please refer to the SciPy documentation.
        
        # Optional parameters
        save_trained_network = False # Whether to ask the user if they would like to save the network after training
    )