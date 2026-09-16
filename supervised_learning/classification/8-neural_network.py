#!/usr/bin/env python3
""" Defines a NN with one hidden layer for binary classification. """
import numpy as np


class NeuralNetwork:
    """ Neural network with one hidden layer performing binary classification.

    Public attributes:
        W1: weights for hidden layer (nodes, nx)
        b1: bias for hidden layer (nodes, 1)
        A1: activated output of hidden layer
        W2: weights for output neuron (1, nodes)
        b2: bias for output neuron (scalar)
        A2: activated output of output neuron (prediction)
    """

    def __init__(self, nx, nodes):
        """ Initializes the neural network.

        Args:
            nx : number of input features.
            nodes : number of nodes in the hidden layer.

        Raises:
            TypeError: if nx or nodes is not an integer.
            ValueError: if nx or nodes is less than 1.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        # Hidden layer weights and bias
        self.W1 = np.random.randn(nodes, nx)
        self.b1 = np.zeros((nodes, 1))
        self.A1 = 0

        # Output layer weights and bias
        self.W2 = np.random.randn(1, nodes)
        self.b2 = 0
        self.A2 = 0
