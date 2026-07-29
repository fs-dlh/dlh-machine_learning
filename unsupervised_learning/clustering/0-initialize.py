#!/usr/bin/env python3
""" Module for initializing K-means centroids. """
import numpy as np


def initialize(X, k):
    """ Initialize cluster centroids for K-means

    Args:
        X : Dataset of shape (n, d).
        k : Number of clusters (positive integer).

    Returns:
        numpy.ndarray: Initialized centroids of shape (k, d).
    """

    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    if not isinstance(k, int) or k <= 0:
        return None

    n, d = X.shape
    if n == 0 or d == 0:
        return None

    mins = np.min(X, axis=0)
    maxs = np.max(X, axis=0)

    c = np.random.uniform(low=mins, high=maxs, size=(k, d))

    return c
