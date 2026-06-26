#!/usr/bin/env python3
"""Module for defining the MultiNormal class for multivariate  distribution."""

import numpy as np


class MultiNormal:
    """ Class for multivariate normal distribution. """

    def __init__(self, data):
        self.data = data
        self.mean = np.mean(data, axis=1)
        self.cov = np.cov(data)

    def pdf(self, x):
        """ Calculate the probability density function. """
        diff = x - self.mean
        inv_cov = np.linalg.inv(self.cov)
        det_cov = np.linalg.det(self.cov)
        exponent = -0.5 * np.dot(np.dot(diff.T, inv_cov), diff)
        coefficient = 1 / np.sqrt((2 * np.pi) ** len(x) * det_cov)
        return coefficient * np.exp(exponent)


def correlation(C):
    """      Compute the correlation matrix from a covariance matrix.

    Args:    C (numpy.ndarray): Shape (d, d) covariance matrix.

    Returns: numpy.ndarray: Shape (d, d) correlation matrix.

    Raises: TypeError: If C is not a numpy.ndarray.
            ValueError: If C is not a square 2D matrix.
    """

    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")

    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    std_dev = np.sqrt(np.diag(C))

    std_outer = np.outer(std_dev, std_dev)

    corr = C / std_outer

    return corr
