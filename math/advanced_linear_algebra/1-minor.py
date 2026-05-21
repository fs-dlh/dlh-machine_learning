#!/usr/bin/env python3
""" Module for matrix determinant calculation. """


def determinant(matrix):
    """ Calculates the determinant of a matrix.
    Args:
        matrix: A list of lists representing the matrix

    Returns:
        The determinant of the matrix

    Raises:
        TypeError: If matrix is not a list of lists
        ValueError: If matrix is not square
    """

    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")

    if len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")

    for row in matrix:
        if not isinstance(row, list):
            raise TypeError("matrix must be a list of lists")

    if len(matrix) == 1 and len(matrix[0]) == 0:
        return 1

    n = len(matrix)
    for row in matrix:
        if len(row) != n:
            raise ValueError("matrix must be a square matrix")

    if n == 1:
        return matrix[0][0]

    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    det = 0

    for j in range(n):
        submatrix = []
        for i in range(1, n):
            row = []
            for k in range(n):
                if k != j:
                    row.append(matrix[i][k])
            submatrix.append(row)
        cofactor = matrix[0][j] * determinant(submatrix)
        if j % 2 == 1:  # Alternate signs
            cofactor = -cofactor
        det += cofactor

    return det


def minor(matrix):
    """  Calculates the minor matrix of a matrix.

    Args:
        matrix: A list of lists representing the matrix

    Returns:
        A list of lists representing the minor matrix

    Raises:
        TypeError: If matrix is not a list of lists
        ValueError: If matrix is not a non-empty square matrix
    """

    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")

    if len(matrix) == 0:
        raise ValueError("matrix must be a non-empty square matrix")

    for row in matrix:
        if not isinstance(row, list):
            raise TypeError("matrix must be a list of lists")

    n = len(matrix)
    for row in matrix:
        if len(row) != n:
            raise ValueError("matrix must be a non-empty square matrix")

    if n == 1:
        return [[1]]

    minor_matrix = []

    for i in range(n):
        minor_row = []
        for j in range(n):
            submatrix = []
            for r in range(n):
                if r == i:
                    continue
                sub_row = []
                for c in range(n):
                    if c == j:
                        continue
                    sub_row.append(matrix[r][c])
                submatrix.append(sub_row)
            minor_value = determinant(submatrix)
            minor_row.append(minor_value)

        minor_matrix.append(minor_row)

    return minor_matrix
