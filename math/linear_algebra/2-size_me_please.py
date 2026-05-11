#!/usr/bin/env python3
"""Module for calculating the shape of a matrix."""


def matrix_shape(matrix):
    """
    Calculate the shape of a matrix.

    Args:
        matrix (list): A nested list representing a matrix.

    Returns:
        list: A list of integers representing the dimensions.
    """

    shape = [len(matrix)]
    if matrix and isinstance(matrix[0], list):
        shape.extend(matrix_shape(matrix[0]))
    return shape
