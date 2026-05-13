#!/usr/bin/env python3
"""Module for concatenating numpy ndarrays along a specified axis."""


import numpy as np


def np_cat(mat1, mat2, axis=0):
    """
    Concatenate two numpy ndarrays along a given axis.

    Args:
        mat1 (numpy.ndarray): First array.
        mat2 (numpy.ndarray): Second array.
        axis (int): Axis along which to concatenate (default is 0).
    Returns:
        numpy.ndarray: Concatenated array.
    """
    return np.concatenate((mat1, mat2), axis=axis)
