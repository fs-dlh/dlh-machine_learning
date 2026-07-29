#!/usr/bin/env python3
""" Module for performing K-means clustering. """
import numpy as np


def kmeans(X, k, iterations=1000):
    """ Perform K-means clustering on a dataset.

    Args:
        X : Dataset of shape (n, d).
        k : Number of clusters (positive integer).
        iterations : Maximum number of iterations to perform.

    Returns:
        tuple: (C, clss) where
            C is the centroid means of shape (k, d)
            and clss is the cluster index for each point of shape (n,),
            or (None, None) on failure.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None
    if not isinstance(k, int) or k <= 0:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    n, d = X.shape

    mins = X.min(axis=0)
    maxs = X.max(axis=0)

    C = np.random.uniform(low=mins, high=maxs, size=(k, d))

    for _ in range(iterations):
        distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
        clss = np.argmin(distances, axis=1)
        new_C = np.copy(C)

        for j in range(k):
            mask = (clss == j)
            if not np.any(mask):
                new_C[j] = np.random.uniform(mins, maxs)
            else:
                new_C[j] = X[mask].mean(axis=0)

        if np.array_equal(new_C, C):
            return new_C, clss

        C = new_C
    distances = np.linalg.norm(X[:, None] - C, axis=2)
    clss = np.argmin(distances, axis=1)

    return C, clss
