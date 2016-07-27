.. _gradient-checking:

Gradient Checking
=======================

Gradient checking great method for debugging neural networks. The main challenge with implementing these networks, is to get the gradient calculations correct. To verify the analytically computed gradients that are used during gradient descent, we can compare these gradients to numerically calculated gradients. This is called gradient checking.

.. warning:: 

    Gradient checking against a large dataset is *very* slow.

.. important::

    If the gradient check fails, it will query the user whether to abort or continue executing the script.


Usage
*****

Checking the gradient of a network requires both a *dataset* and a specific *cost function*.

.. code:: python

        network = NeuralNet( ... ) # parameters omitted for readability
        network.check_gradient( dataset, cost_function )

The following code snippet is a complete example on how to perform gradient checking:

.. code:: python

    from nimblenet.activation_functions import binary_cross_entropy_cost
    from nimblenet.cost_functions import cross_entropy_cost
    from nimblenet.data_structures import Instance
    from nimblenet.neuralnet import NeuralNet

    cost_function  = binary_cross_entropy_cost
    dataset        = [ Instance( [0,0], [0] ), Instance( [1,0], [1] ), Instance( [0,1], [1] ), Instance( [1,1], [0] )]
    settings       = {
        "n_inputs" : 2,
        "layers"   : [  (2, sigmoid_function), (1, sigmoid_function) ]
    }

    network        = NeuralNet( settings )
    network.check_gradient( dataset, cost_function )
