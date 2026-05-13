#!/usr/bin/env python3
"""Module for element-wise operations on numpy ndarrays."""


def np_elementwise(mat1, mat2):
    """
    Perform element-wise addition, subtraction, multiplication, and division.

    Args:
        mat1 (numpy.ndarray): First operand.
        mat2 (numpy.ndarray or scalar): Second operand.
    Returns:
        tuple: (sum, difference, product, quotient) as numpy.ndarrays.
    """
    return mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2
