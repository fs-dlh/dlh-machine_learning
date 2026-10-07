#!/usr/bin/env python3
""" This module defines a GRU cell. """
import numpy as np


class GRUCell:
    """ Represents a gated recurrent unit (GRU). """

    def __init__(self, i, h, o):
        """Initialize the GRU cell.

        Args:
            i : Dimensionality of the data.
            h : Dimensionality of the hidden state.
            o : Dimensionality of the outputs.
        """
        self.Wz = np.random.randn(i + h, h)
        self.Wr = np.random.randn(i + h, h)
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)

        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """ Perform forward propagation for one time step.

        Args:
            h_prev : Previous hidden state of shape (m, h).
            x_t : Input data of shape (m, i).

        Returns:
            tuple: (h_next, y)
                h_next is the next hidden state.
                y is the output of the cell.
        """
        concat = np.concatenate((h_prev, x_t), axis=1)

        z = np.matmul(concat, self.Wz) + self.bz
        z = 1 / (1 + np.exp(-z))

        r = np.matmul(concat, self.Wr) + self.br
        r = 1 / (1 + np.exp(-r))

        concat_r = np.concatenate((r * h_prev, x_t), axis=1)
        h_tilde = np.tanh(np.matmul(concat_r, self.Wh) + self.bh)

        h_next = (1 - z) * h_prev + z * h_tilde

        y = np.matmul(h_next, self.Wy) + self.by
        y = np.exp(y - np.max(y, axis=1, keepdims=True))
        y = y / np.sum(y, axis=1, keepdims=True)

        return h_next, y
