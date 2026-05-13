#!/usr/bin/env python3
"""Module for slicing a numpy ndarray along multiple axes."""


import numpy as np


def np_slice(matrix, axes={}):
    """
    Slice a numpy ndarray along specified axes.

    Args:
        matrix (numpy.ndarray): Input array.
        axes (dict): Dictionary where key is axis and value is a tuple
                     representing the slice. Tuple can have 1, 2, or 3 elements
                     - (start,)        -> slice from start to end
                     - (start, stop)   -> slice from start (inclusive) to stop 
                     - (start, stop, step) -> slice with step

    Returns:
        numpy.ndarray: Sliced array.

    Example:
        >>> mat = np.arange(12).reshape(3,4)
        >>> np_slice(mat, axes={0: (1,), 1: (1,3)})
        array([[5, 6]])
    """
    # Build list of slice objects for each dimension
    slices = [slice(None)] * matrix.ndim
    for axis, tup in axes.items():
        if len(tup) == 1:
            s = slice(tup[0], None)
        elif len(tup) == 2:
            s = slice(tup[0], tup[1])
        else:  # len(tup) == 3
            s = slice(tup[0], tup[1], tup[2])
        slices[axis] = s
    return matrix[tuple(slices)]