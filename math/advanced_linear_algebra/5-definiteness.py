#!/usr/bin/env python3
"""Module to calculate the definiteness of a matrix."""


import numpy as np


def definiteness(matrix):
    """     Calculate the definiteness of a matrix.

    Args:
        matrix: numpy.ndarray of shape (n, n) to evaluate

    Returns:
        String indicating definiteness: 'Positive definite',
        'Positive semi-definite', 'Negative semi-definite',
        'Negative definite', or 'Indefinite'
        Returns None if matrix doesn't fit any category or is invalid

    Raises:
        TypeError: If matrix is not a numpy.ndarray
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    if matrix.size == 0:
        return None

    if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
        return None

    if not np.allclose(matrix, matrix.T):
        return None

    eigenvalues = np.linalg.eigvals(matrix)

    if np.all(eigenvalues > 0):
        return "Positive definite"

    if np.all(eigenvalues >= 0):
        return "Positive semi-definite"

    if np.all(eigenvalues < 0):
        return "Negative definite"

    if np.all(eigenvalues <= 0):
        return "Negative semi-definite"

    if np.any(eigenvalues > 0) and np.any(eigenvalues < 0):
        return "Indefinite"

    return None
