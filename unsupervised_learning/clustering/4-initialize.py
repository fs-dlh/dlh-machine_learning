#!/usr/bin/env python3
""" Module initializes variables for a Gaussian Mixture Model. """
import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """ Initializes variables for a Gaussian Mixture Model.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        k: positive integer, number of clusters

    Returns:
        pi: numpy.ndarray of shape (k,) containing priors
        m:  numpy.ndarray of shape (k, d) containing centroid means
        S:  numpy.ndarray of shape (k, d, d) containing covariance matrices
        or (None, None, None) on failure
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None
    n, d = X.shape
    if not isinstance(k, int) or k <= 0 or k > n or d <= 0:
        return None, None, None

    pi = np.full(k, 1.0 / k)
    centroids, _ = kmeans(X, k)
    if centroids is None:
        return None, None, None
    m = centroids
    S = np.tile(np.identity(d), (k, 1, 1))

    return pi, m, S
