#!/usr/bin/env python3
""" 2-shuffle_data module """
import numpy as np


def shuffle_data(X, Y):
    """ Shuffles the data points in two matrices the same way.

    Args:
        X : shape (m, nx) containing the first dataset.
        Y : shape (m, ny) containing the second dataset.

    Returns:
        X_shuffled : shuffled version of X
        Y_shuffled : shuffled version of Y
    """
    m = X.shape[0]
    permutation = np.random.permutation(m)
    return X[permutation], Y[permutation]
