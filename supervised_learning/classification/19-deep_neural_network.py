#!/usr/bin/env python3
""" Defines a deep neural network for binary classification. """
import numpy as np


class DeepNeuralNetwork:
    """ Deep neural network performing binary classification.

    Public attributes:
        L: number of layers
        cache: dictionary to hold intermediary values
        weights: dictionary to hold weights and biases (W1, b1, W2, b2, ...)
    """

    def __init__(self, nx, layers):
        """ Initializes the deep neural network.

        Args:
            nx : number of input features.
            layers : number of nodes in each layer.

        Raises:
            TypeError: if nx is not an integer,
                        or layers is not a list,
                        or layers is empty,
                        or any layer node count is not a positive integer.
            ValueError: if nx is less than 1.
        """

        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        prev_size = nx
        for i, nodes in enumerate(layers, 1):
            if not isinstance(nodes, int) or nodes <= 0:
                raise TypeError("layers must be a list of positive integers")

            scale = np.sqrt(2 / prev_size)
            W = np.random.randn(nodes, prev_size)
            self.weights[f"W{i}"] = W * scale

            self.weights[f"b{i}"] = np.zeros((nodes, 1))

            prev_size = nodes

    @property
    def L(self):
        """Getter for number of layers."""
        return self.__L

    @property
    def cache(self):
        """Getter for cache dictionary."""
        return self.__cache

    @property
    def weights(self):
        """Getter for weights dictionary."""
        return self.__weights

    def forward_prop(self, X):
        """ Calculates forward propagation of the neural network.

        Args:
            X : input data of shape (nx, m).

        Returns:
            tuple: (output, cache)
                output: activated output of the last layer (1, m)
                cache: dictionary containing all intermediary values
        """

        self.__cache["A0"] = X
        for layer in range(1, self.__L + 1):
            A_prev = self.__cache[f"A{layer - 1}"]
            W = self.__weights[f"W{layer}"]
            b = self.__weights[f"b{layer}"]

            Z = np.matmul(W, A_prev) + b
            A = 1 / (1 + np.exp(-Z))
            self.__cache[f"A{layer}"] = A

        return self.__cache[f"A{self.__L}"], self.__cache

    def cost(self, Y, A):
        """ Calculates the logistic regression cost.

        Args:
            Y : correct labels of shape (1, m).
            A : activated outputs of shape (1, m).

        Returns:
            float: the cost.
        """
        m = Y.shape[1]
        cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return cost
