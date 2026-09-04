#!/usr/bin/env python3
""" 1-normalize module """
import numpy as np


def normalize(X, m, s):
    """ Normalizes (standardizes) a matrix X using provided mean and sd.

    Args:
        X : shape (d, nx) containing the dataset to normalize.
                           d: number of data points, nx: number of features.
        m : shape (nx,) containing the mean of each feature.
        s : shape (nx,) containing the standard deviation of each feature.

    Returns:
        numpy.ndarray: The normalized X matrix (X - m) / s, shape (d, nx).
    """
    return (X - m) / s
