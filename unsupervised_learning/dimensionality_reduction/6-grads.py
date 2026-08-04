#!/usr/bin/env python3
""" that calculates that calculates the gradients of Y """
import numpy as np
Q_affinities = __import__('5-Q_affinities').Q_affinities


def grads(Y, P):
    """ Compute that calculates the gradients of Y for t‑SNE,

    Parameters:
    Y : numpy.ndarray of shape (n, ndim)
        Low‑dimensional representation.
    P : numpy.ndarray of shape (n, n)
        Joint probabilities (affinities) of the high‑dimensional data.

    Returns:
    dY : numpy.ndarray of shape (n, ndim)
        Gradients of Y (cost w.r.t. Y) – not multiplied by 4.
    Q : numpy.ndarray of shape (n, n)
        The Q affinities (joint probabilities) of Y.
    """

    Q, num = Q_affinities(Y)

    A = (P - Q) * num

    row_sum_A = np.sum(A, axis=1, keepdims=True)
    dY = row_sum_A * Y - A @ Y

    return dY, Q
