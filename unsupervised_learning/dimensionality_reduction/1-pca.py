#!/usr/bin/env python3
""" 1-pca module """
import numpy as np


def pca(X, ndim):
    """ Performs PCA on a dataset and returns the transformed data.

    Args:
        X : shape (n, d), the dataset.
        ndim : new dimensionality of the transformed data.

    Returns:
        numpy.ndarray: T of shape (n, ndim), the projected data.
    """

    X_centered = X - np.mean(X, axis=0)

    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    W = Vt.T[:, :ndim]

    T = X_centered @ W

    return T
