#!/usr/bin/env python3
""" 13-batch_norm module """
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """ Normalizes an unactivated output of a nn using batch normalization.

    Args:
        Z : shape (m, n) to be normalized.
        gamma : shape (1, n) containing scales.
        beta : shape (1, n) containing offsets.
        epsilon : small number to avoid division by zero.

    Returns:
        numpy.ndarray: normalized Z matrix, same shape as Z.
    """

    mean = np.mean(Z, axis=0, keepdims=True)  # shape (1, n)
    variance = np.var(Z, axis=0, keepdims=True)  # shape (1, n)

    Z_norm = (Z - mean) / np.sqrt(variance + epsilon)

    return gamma * Z_norm + beta
