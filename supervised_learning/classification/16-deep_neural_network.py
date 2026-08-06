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
        for node in layers:
            if not isinstance(node, int) or node <= 0:
                raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        for i in range(1, self.L + 1):
            if i == 1:
                input_dim = nx
            else:
                input_dim = layers[i - 2]

            output_dim = layers[i - 1]

            scale = np.sqrt(2 / input_dim)
            W = np.random.randn(output_dim, input_dim)
            self.weights[f"W{i}"] = W * scale
            self.weights[f"b{i}"] = np.zeros((output_dim, 1))
