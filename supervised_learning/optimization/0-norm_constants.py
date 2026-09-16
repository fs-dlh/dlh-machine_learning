#!/usr/bin/env python3
""" 0-norm_constants module """
import numpy as np


def normalization_constants(X):
    """ Calculates the normalization (standardization) constants of a matrix.

    Args:
        X : shape (m, nx) containing the dataset.
                           m: number of data points, nx: number of features.

    Returns:
        mean : mean of each feature, shape (nx,)
        std : standard deviation of each feature, shape (nx,)
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return mean, std
