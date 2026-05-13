#!/usr/bin/env python3
"""Module for adding two 2D matrices element-wise."""


def add_matrices2D(mat1, mat2):
    """
    Add two 2D matrices element-wise.

    Args:
        mat1 (list of list of int/float): First 2D matrix.
        mat2 (list of list of int/float): Second 2D matrix.

    Returns:
        list of list: New matrix with element-wise sums,
        or None if shapes differ.    """

    if len(mat1) != len(mat2):
        return None
    for i in range(len(mat1)):
        if len(mat1[i]) != len(mat2[i]):
            return None
    return [
        [mat1[i][j] + mat2[i][j] for j in range(len(mat1[i]))]
        for i in range(len(mat1))
    ]
