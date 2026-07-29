#!/usr/bin/env python3
""" Module for total intra-cluster variance. """
import numpy as np


def variance(X, C):
    """ Calculates the total intra-cluster variance for a data set.

    Parameters:
    X : numpy.ndarray of shape (n, d)
    C : numpy.ndarray of shape (k, d)

    Returns:
    var : Total variance, or None on failure.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(C, np.ndarray) or len(C.shape) != 2:
        return None
    if X.shape[1] != C.shape[1]:
        return None

    dist = np.sum((X[:, np.newaxis, :] - C[np.newaxis, :, :]) ** 2, axis=2)
    nearest_idx = np.argmin(dist, axis=1)
    min_distances = dist[np.arange(X.shape[0]), nearest_idx]
    var = np.sum(min_distances)
    return var
