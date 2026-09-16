#!/usr/bin/env python3
""" Defines a Neuron class with FP using sigmoid activation. """
import numpy as np


class Neuron:
    """ Defines a single neuron performing binary classification.

    Private instance attributes:
        __W: weights vector (random normal)
        __b: bias (0)
        __A: activated output (0)
    """

    def __init__(self, nx):
        """ Constructor for the Neuron.

        Args:
            nx : Number of input features to the neuron.

        Raises:
            TypeError: If nx is not an integer.
            ValueError: If nx is less than 1.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """ Getter for the weights vector. """
        return self.__W

    @property
    def b(self):
        """ Getter for the bias. """
        return self.__b

    @property
    def A(self):
        """ Getter for the activated output. """
        return self.__A

    def forward_prop(self, X):
        """ Calculates the forward propagation of the neuron.

        Args:
            X : input data of shape (nx, m).

        Returns:
            numpy.ndarray: the activated output (sigmoid) of shape (1, m).
        """
        # Compute linear combination: z = W.X + b
        z = np.matmul(self.__W, X) + self.__b
        # Sigmoid activation
        self.__A = 1 / (1 + np.exp(-z))
        return self.__A
