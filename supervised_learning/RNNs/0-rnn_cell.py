#!/usr/bin/env python3
""" This module defines a simple RNN cell. """
import numpy as np


class RNNCell:
    """ Represents a cell of a simple RNN. """

    def __init__(self, i, h, o):
        """Initialize the RNN cell.

        Args:
            i : Dimensionality of the data.
            h : Dimensionality of the hidden state.
            o : Dimensionality of the outputs.
        """
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """Perform forward propagation for one time step.

        Args:
            h_prev : Previous hidden state of shape (m, h).
            x_t : Input data of shape (m, i).

        Returns:
            tuple: (h_next, y)
                h_next is the next hidden state.
                y is the output of the cell.
        """
        concat = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.matmul(concat, self.Wh) + self.bh)

        y = np.matmul(h_next, self.Wy) + self.by
        y = np.exp(y - np.max(y, axis=1, keepdims=True))
        y = y / np.sum(y, axis=1, keepdims=True)

        return h_next, y
