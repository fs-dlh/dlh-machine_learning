#!/usr/bin/env python3
""" Module for optimum number of clusters by variance. """
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """ Tests for the optimum number of clusters by variance.

    Parameters:
    X : numpy.ndarray of shape (n, d)
    kmin : Minimum number of clusters to check for (inclusive).
    kmax : Maximum number of clusters to check for (inclusive).
    iterations : Maximum number of iterations for K-means.

    Returns:
    results : Each element is a tuple (centroids, cluster_assignments).
    d_vars : Difference in variance from the smallest cluster size.
    None, None : on failure.
    """

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None

    if not isinstance(kmin, int) or kmin < 1:
        return None, None
    n, d = X.shape
    if kmax is None:
        kmax = n
    if not isinstance(kmax, int) or kmax < kmin:
        return None, None
    if kmax - kmin < 1:
        return None, None
    if not isinstance(iterations, int) or iterations < 1:
        return None, None

    results = []
    d_vars = []
    vm = kmin

    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        if C is None or clss is None:
            return None, None
        var = variance(X, C)
        if var is None:
            return None, None
        results.append((C, clss))
        d_vars.append(vm - var)

    return results, d_vars
