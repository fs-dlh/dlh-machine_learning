#!/usr/bin/env python3
"""Module for calculating correlation matrix from covariance matrix."""

import numpy as np


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
