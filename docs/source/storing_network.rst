.. _storing-network:

Saving and loading a trained network
=====================================


Save
------

A network can be easily saved to a file:

.. code:: python

    from nimblenet.neuralnet import NeuralNet
    # Create a network
    network = NeuralNet({
        "n_inputs" : 2,
        "layers"   : [ (1, sigmoid_function) ],
    })
    # Save the network to disk
    network.save_network_to_file( "%s.pkl" % "filename" )

In addition to doing this explicitly, all of the learning algorithms also offer the possibility to save the network after the training has completed. This is done by passing the named parameter ``save_trained_network = True`` when calling the learning function:

.. code:: python

    RMSprop( ..., save_trained_network = False ) # omitted parameters for readability

This will promt the user whether to save the network or not, upon completion of the training.

Load
------

If you have saved a network to a file, you can easily load the network back up by calling:

.. code:: python

    from nimblenet.neuralnet import NeuralNet
    network = NeuralNet.load_network_from_file( "%s.pkl" % "filename" )