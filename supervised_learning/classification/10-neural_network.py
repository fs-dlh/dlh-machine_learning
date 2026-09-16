#!/usr/bin/env python3
""" Defines a NN with one hidden layer for binary classification. """
import numpy as np


class NeuralNetwork:
    """ Neural network with one hidden layer performing binary classification.

    Private attributes:
        __W1: weights for hidden layer (nodes, nx)
        __b1: bias for hidden layer (nodes, 1)
        __A1: activated output of hidden layer
        __W2: weights for output neuron (1, nodes)
        __b2: bias for output neuron (scalar)
        __ A2: activated output of output neuron (prediction)
    Getters:
        W1, b1, A1, W2, b2, A2
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

        # Hidden layer weights and bias (Initialized as private variables)
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0

        # Output layer weights and bias (Initialized as private variables)
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        return self.__W1

    @property
    def b1(self):
        return self.__b1

    @property
    def A1(self):
        return self.__A1

    @property
    def W2(self):
        return self.__W2

    @property
    def b2(self):
        return self.__b2

    @property
    def A2(self):
        return self.__A2

    def forward_prop(self, X):
        """ Calculates the forward propagation of the neural network.

        Args:
            X : input data of shape (nx, m).

        Returns:
            tuple: (A1, A2)
                A1: activated output of hidden layer (nodes, m)
                A2: activated output of output layer (1, m)
        """
        # Hidden layer forward propagation
        Z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-Z1))

        # Output layer forward propagation
        Z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-Z2))

        return self.__A1, self.__A2
