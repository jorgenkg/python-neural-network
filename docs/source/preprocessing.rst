Preprocessing
=======================

This is a work in progress. However, the library has already implementations a few of the most used preprocessing techniques.

.. contents::
   :local:
   :backlinks: none

A preprocessor can be constructed by combining any number of these techniques, and is intended allow maximum configurability.


Usage
**************

First, we need to import ``construct_preprocessor``. This will take care of combining our preprocessors:

.. code:: python

    from nimblenet.preprocessing import construct_preprocessor

Next, we import the preprocessors we'd like to apply:

.. code:: python

    from nimblenet.preprocessing import replace_nan, standarize

Then, we combine the preprocessors. This is done by sending a list of preprocessors in addition to the dataset which we would like to fit the preprocessors againts. Note: this dataset should be the combined set of training, test and validation data.

.. code:: python

    preprocess = construct_preprocessor( dataset, [
                    ( replace_nan, {"replace_with": 0 }), 
                    standardize
                ])

This constructed preprocessor can now be applied to your datasets. Let's take a look at how we can apply this to the XOR dataset:

.. code:: python

    from nimblenet.data_structures import Instance
    dataset              = [ Instance( [0,0], [0] ), Instance( [1,0], [1] ), Instance( [0,1], [1] ), Instance( [1,1], [0] ) ]
    preprocess           = construct_preprocessor( dataset, [
                                ( replace_nan, {"replace_with": 0 }), 
                                standardize
                            ])
    preprocessed_dataset = preprocess( dataset )

Remember that if using a preprocessor before training the network, you will have to use the same preprocessor before using the network to predict based on new input signals.



    
.. important::

    The dataset given to ``construct_preprocessor`` should be the combined set of training, test and validation data.
    
    
Available preprocessors
************************

Standardize
^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    from nimblenet.preprocessing import standarize

Has no parameters.

Replace *NaN*
^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    from nimblenet.preprocessing import replace_nan

Takes an optional parameter ``replace_with``. By default, it replaces *NaN* with the mean of the given input signal.

In order to replace *NaN* with zero:

.. code:: python

    from nimblenet.preprocessing import construct_preprocessor, replace_nan
    from nimblenet.data_structures import Instance
    
    dataset    = [ Instance( [0,0], [0] ), Instance( [1,0], [1] ), Instance( [0,1], [1] ), Instance( [1,1], [0] ) ]
    preprocess = construct_preprocessor( dataset, [
                    ( replace_nan, {"replace_with": 0 }), 
                ])


Subtract Mean
^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    from nimblenet.preprocessing import subtract_mean

Has no parameters.


Normalize
^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    from nimblenet.preprocessing import normalize

Has no parameters.


Whiten
^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    from nimblenet.preprocessing import whiten

Takes an optional parameter ``epsilon``. By default, epsilon equals ``1e-5``.

In order to redefine epsilon to e.g 0.5:

.. code:: python

    from nimblenet.preprocessing import construct_preprocessor, whiten
    from nimblenet.data_structures import Instance
    
    dataset    = [ Instance( [0,0], [0] ), Instance( [1,0], [1] ), Instance( [0,1], [1] ), Instance( [1,1], [0] ) ]
    preprocess = construct_preprocessor( dataset, [
                    ( whiten, {"epsilon": 0.5 }), 
                ])