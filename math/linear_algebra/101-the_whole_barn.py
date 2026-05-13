#!/usr/bin/env python3
"""Module for adding two matrices of arbitrary dimensions."""


def add_matrices(mat1, mat2):
    """
    Add two matrices element-wise.

    Args:
        mat1 (list of int/float or nested list): First matrix.
        mat2 (list of int/float or nested list): Second matrix.

    Returns:
        list or None: New matrix with element-wise sums, or None.
    """

    if isinstance(mat1, (int, float)) and isinstance(mat2, (int, float)):
        return mat1 + mat2

    if isinstance(mat1, list) and isinstance(mat2, list):
        if len(mat1) != len(mat2):
            return None
        result = []
        for i in range(len(mat1)):
            sub_result = add_matrices(mat1[i], mat2[i])
            if sub_result is None:
                return None
            result.append(sub_result)
        return result

    return None
