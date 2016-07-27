Initializing a Network
=======================

In nimblenet, a neural network is configured according to a dict of parameters specified upon initialization.

.. code:: python

    from nimblenet.neuralnet import NeuralNet
    network = NeuralNet({
        "n_inputs" : 2,
        "layers"   : [ (1, sigmoid_function) ],
    })

.. important::
    
    The final tuple in the layers list always describe the number of output signals.

Parameters
------------

The two dict keys ``n_inputs`` and ``layers`` are required. However, the network is further customizable through specifying any of the following dict parameters:

* ``n_inputs`` the number of input signals
* ``layers`` the topology of the network
* ``initial_bias_value`` the input signal from the bias node will be initialized to this value
* ``weights_low`` the lower bound on weight value during the random initialization
* ``weights_high`` the upper bound on weight value during the random initialization

Example
---------

.. code:: python
    
    from nimblenet.neuralnet import NeuralNet
    settings            = {
        # Required settings
        "n_inputs"              : 2,       # Number of network input signals
        "layers"                : [  (3, sigmoid_function), (1, sigmoid_function) ],
                                            # [ (number_of_neurons, activation_function) ]
                                            # The last pair in the list dictate the number of output signals
    
        # Optional settings
        "initial_bias_value"    : 0.0,
        "weights_low"           : -0.1,     # Lower bound on the initial weight value
        "weights_high"          : 0.1,      # Upper bound on the initial weight value
    }
    network = NeuralNet( settings )