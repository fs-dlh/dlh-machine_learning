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
    U, S, Vh = np.linalg.svd(X)

    total_var = np.sum(S)

    cumsum = np.cumsum(S)

    k = np.argmax(cumsum >= var * total_var) + 1

    W = Vh.T[:, :k]

    return W
