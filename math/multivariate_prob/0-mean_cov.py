#!/usr/bin/env python3
"""Module for calculating mean and covariance of a dataset."""

import numpy as np


def mean_cov(X):
    """ Calculate the mean and covariance matrix of a data set.

    Args:
        X (numpy.ndarray): n data points of d dimensions.

    Returns:
        mean (numpy.ndarray): Shape (1, d) the mean of the data set.
        cov (numpy.ndarray): Shape (d, d) the covariance matrix.

    Raises:
        TypeError: If X is not a 2D numpy.ndarray.
        ValueError: If X contains fewer than 2 data points.
    """
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a 2D numpy.ndarray")
    if X.ndim != 2:
        raise TypeError("X must be a 2D numpy.ndarray")

    n, d = X.shape
    if n < 2:
        raise ValueError("X must contain multiple data points")

    mean = np.mean(X, axis=0, keepdims=True)

    X_centered = X - mean

    cov = (X_centered.T @ X_centered) / (n - 1)

    return mean, cov
