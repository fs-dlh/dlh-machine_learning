#!/usr/bin/env python3
""" Module that calculates the Q affinities for t-SNE.  """
import numpy as np


def Q_affinities(Y):
    """ Calculate the Q affinities for t-SNE.

    Parameters:
    Y : Low-dimensional representation of the data.

    Returns:
    Q : The Q affinities (probability distribution over pairs).
    num : Numerator of the Q affinities,
        i.e. (1 + ||y_i - y_j||^2)^(-1) for i ≠ j, and 0 on the diagonal.
    """
    n, ndim = Y.shape

    sum_sq = np.sum(Y ** 2, axis=1)
    D = sum_sq[:, np.newaxis] + sum_sq[np.newaxis, :] - 2 * np.dot(Y, Y.T)

    np.clip(D, 0, None, out=D)

    num = 1.0 / (1.0 + D)
    np.fill_diagonal(num, 0)

    Q = num / np.sum(num)

    return Q, num
