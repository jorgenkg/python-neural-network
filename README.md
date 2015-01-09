# Neural network written in Python (NumPy)
This is an implementation of a fully connected neural network in NumPy. The network may be trained by the backpropagation algorithm in an on-line fashion. 

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
 * Three activation functions: tanh, sigmoid and the linear activation function.
