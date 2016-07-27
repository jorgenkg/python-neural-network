
Welcome to nimblenet's documentation!
=====================================

nimblenet is a lightweight and efficient Numpy library for creating feed forward neural networks. The library was developed with PYPY in mind and should play nicely with their super-fast JIT compiler. The networks can be trained by a variety of learning algorithms: backpropagation, resilient backpropagation, adaptive learning rate backpropagation, scaled conjugate gradient and SciPy's optimize function. 

This is a list of handy links to get up and running.

* :doc:`getting_started`
* :doc:`cost_functions`
* :doc:`activation_functions`
* :doc:`storing_network`


Installing
=====================

.. code::

    $ pip install nimblenet

Dependencies
--------------------

* Python 2.7
* NumPy
* SciPy (optional). This is of course a required depedency if you intend to train the network using SciPy's ``optimize`` function.
    

Content
==================

.. toctree::
   :maxdepth: 2
   
   getting_started
   initializing_network
   storing_network
   preprocessing
   gradient_checking
   activation_functions
   learning_algorithms
   cost_functions
