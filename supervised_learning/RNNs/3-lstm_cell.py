#!/usr/bin/env python3
""" This module defines an LSTM cell. """
import numpy as np


class LSTMCell:
    """ Represents an LSTM unit. """

    def __init__(self, i, h, o):
        """Initialize the LSTM cell.

        Args:
            i : Dimensionality of the data.
            h : Dimensionality of the hidden state.
            o : Dimensionality of the outputs.
        """
        self.Wf = np.random.randn(i + h, h)
        self.Wu = np.random.randn(i + h, h)
        self.Wc = np.random.randn(i + h, h)
        self.Wo = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)

        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """Perform forward propagation for one time step.

        Args:
            h_prev : Previous hidden state of shape (m, h).
            c_prev : Previous cell state of shape (m, h).
            x_t : Input data of shape (m, i).

        Returns:
            tuple: (h_next, c_next, y)
                h_next is the next hidden state.
                c_next is the next cell state.
                y is the output of the cell.
        """
        concat = np.concatenate((h_prev, x_t), axis=1)

        f = np.matmul(concat, self.Wf) + self.bf
        f = 1 / (1 + np.exp(-f))

        u = np.matmul(concat, self.Wu) + self.bu
        u = 1 / (1 + np.exp(-u))

        c_tilde = np.tanh(np.matmul(concat, self.Wc) + self.bc)

        c_next = f * c_prev + u * c_tilde

        o = np.matmul(concat, self.Wo) + self.bo
        o = 1 / (1 + np.exp(-o))

        h_next = o * np.tanh(c_next)

        y = np.matmul(h_next, self.Wy) + self.by
        y = np.exp(y - np.max(y, axis=1, keepdims=True))
        y = y / np.sum(y, axis=1, keepdims=True)

        return h_next, c_next, y
