Using the Network
=======================


Nimblenet is implemented using matrices rather than for-loops. This allow more efficient computation, and also enables the network to forward propagate multiple input instances at once. 

In contrast to the instances generated when training the network:

.. code:: python

    from nimblenet.data_structures import Instance
    dataset = [ 
        # Instance( [inputs], [outputs] )
        Instance( [0,0], [0] ), ...
    ]

the instances used during prediction need only to be instantiated with a single parameter (the input signal):

.. code:: python

    from nimblenet.data_structures import Instance
    dataset = [ 
        # Instance( [inputs] )
        Instance( [0,0] ), ...
    ]

The following code calculates the output from two instances:

.. code:: python

    prediction_set = [ Instance([0,1]), Instance([1,0]) ]
    print network.predict( prediction_set )
    >> [[ 0.99735413]
        [ 0.99735378]]

The prediction method returns a 2D NumPy array with shape :code:`[n_samples, n_outputs]`. That means each row in the output matrix correspond to an input Instance. The first row of the output matrix, is the output generated from the first instance. Refer to the the expected output below:

.. code:: python

    prediction_set = [ Instance([0,1]) ]
    print network.predict( prediction_set )
    >> [[ 0.99735413]]