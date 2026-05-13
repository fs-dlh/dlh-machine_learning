#!/usr/bin/env python3
"""Module for matrix multiplication using numpy."""


import numpy as np


def np_matmul(mat1, mat2):
    """
    Multiply two numpy ndarrays using matrix multiplication.

    Args:
        mat1 (numpy.ndarray): First matrix.
        mat2 (numpy.ndarray): Second matrix.
    Returns:
        numpy.ndarray: Result of matrix multiplication.
    """
    return np.matmul(mat1, mat2)
