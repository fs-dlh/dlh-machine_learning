#!/usr/bin/env python3
""" This module defines the forward propagation for a simple RNN. """
import numpy as np


def rnn(rnn_cell, X, h_0):
    """Perform forward propagation for a simple RNN.

    Args:
        rnn_cell: An instance of RNNCell used for forward propagation.
        X : Data of shape (t, m, i).
        h_0 : Initial hidden state of shape (m, h).

    Returns:
        tuple: (H, Y)
            H is a numpy.ndarray of shape (t + 1, m, h) containing
              all hidden states.
            Y is a numpy.ndarray of shape (t, m, o) containing
              all outputs.
    """
    t, m, i = X.shape
    h = h_0.shape[1]
    o = rnn_cell.Wy.shape[1]

    H = np.zeros((t + 1, m, h))
    Y = np.zeros((t, m, o))

    H[0] = h_0

    for step in range(t):
        h_prev = H[step]
        x_t = X[step]
        h_next, y = rnn_cell.forward(h_prev, x_t)
        H[step + 1] = h_next
        Y[step] = y

    return H, Y
