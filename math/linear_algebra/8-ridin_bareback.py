#!/usr/bin/env python3
"""Module for matrix multiplication."""


def mat_mul(mat1, mat2):
    """
    Multiply two 2D matrices.

    Args:
        mat1 (list of list of int/float): First matrix (m x n).
        mat2 (list of list of int/float): Second matrix (n x p).

    Returns:
        list of list: Resulting matrix (m x p), or None if multiplication is
        impossible (columns of mat1 != rows of mat2).
    """

    if not mat1 or not mat2:
        return None
    rows1, cols1 = len(mat1), len(mat1[0])
    rows2, cols2 = len(mat2), len(mat2[0])

    if cols1 != rows2:
        return None

    result = [[0 for _ in range(cols2)] for _ in range(rows1)]
    for i in range(rows1):
        for j in range(cols2):
            total = 0
            for k in range(cols1):
                total += mat1[i][k] * mat2[k][j]
            result[i][j] = total
    return result
