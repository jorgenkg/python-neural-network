# Neural network written in Python (NumPy)
This is an implementation of a fully connected neural network in NumPy. The network may be batch trained by backpropagation. By implementing a batch approach, the NumPy implementation is able to harvest the power of the BLAS library to efficiently perform the required calculations. 

*The code has been tested.*

## Requirements
 * Python
 * NumPy

This script has been written with PYPY in mind. Use their [jit-compiler](http://pypy.org/download.html) to run this code blazingly fast.

## How-to
To run the code, navigate into the project folder and execute the following command in the terminal:

`$ python main.py`

To train the network on a custom dataset, you will have to alter the dataset specified in the `main.py` file. It is quite self-explanatory.

## Features:
 * Implemented with matrix operation to improve performance.
 * PYPY friendly (requires pypy-numpy).

## Activation functions:
 * tanh
 * Symmetric Elliot function (fast tanh approximation) 
 * Sigmoid
 * Rectified Linear Unit
 * Linear activation
