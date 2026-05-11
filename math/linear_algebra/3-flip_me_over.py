#!/usr/bin/env python3
"""Module for transposing a 2D matrix."""

def matrix_transpose(matrix):
    """
    Return the transpose of a 2D matrix.

    Args:
        matrix (list of list): The input matrix (non-empty, rectangular).

    Returns:
        list of list: A new matrix that is the transpose of the input.
    """
    # Number of rows and columns in the original matrix
    rows = len(matrix)
    cols = len(matrix[0]) if rows > 0 else 0

    # Build the transposed matrix: cols rows, each with 'rows' elements
    transpose = [[matrix[r][c] for r in range(rows)] for c in range(cols)]
    return transpose
