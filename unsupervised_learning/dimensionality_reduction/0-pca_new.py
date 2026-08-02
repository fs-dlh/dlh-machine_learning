#!/usr/bin/env python3
""" Module that performs PCA on a centered dataset. """
import numpy as np


def pca(X, var=0.95):
    """ Performs PCA on a centered dataset.

    Args:
        X : shape (n, d), centered.
        var : fraction of variance to retain.

    Returns:
        numpy.ndarray: weight matrix W of shape (d, nd), where nd is the new
                       dimensionality. Each column is a principal component.
    """
    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    explained = np.cumsum(S ** 2) / np.sum(S ** 2)

    nd = np.searchsorted(explained, var) + 1

    return Vh[:nd].T
