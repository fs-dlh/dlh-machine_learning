#!/usr/bin/env python3
"""Module for concatenating two 2D matrices along a specified axis."""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    Concatenate two 2D matrices along a specified axis.

    Args:
        mat1 (list of list of int/float): First 2D matrix.
        mat2 (list of list of int/float): Second 2D matrix.
        axis (int): Axis along which to concatenate (0 for vertical, 1 for
                    horizontal). Defaults to 0.

    Returns:
        list of list: New matrix resulting from concatenation, or None if the
        matrices cannot be concatenated due to shape mismatch.
    """
    # Validate axis
    if axis not in (0, 1):
        return None

    # Concatenate vertically (axis=0)
    if axis == 0:
        # Check column compatibility
        if len(mat1) == 0 or len(mat2) == 0:
            # Handle empty matrices: if one is empty, return a copy of the other
            if len(mat1) == 0 and len(mat2) == 0:
                return []
            if len(mat1) == 0:
                return [row[:] for row in mat2]
            if len(mat2) == 0:
                return [row[:] for row in mat1]
        # Ensure same number of columns
        if len(mat1[0]) != len(mat2[0]):
            return None
        # Create deep copy of mat1 rows and mat2 rows
        return [row[:] for row in mat1] + [row[:] for row in mat2]

    # Concatenate horizontally (axis=1)
    if axis == 1:
        # Check row compatibility
        if len(mat1) != len(mat2):
            return None
        # Create new matrix with each row concatenated
        return [mat1[i][:] + mat2[i][:] for i in range(len(mat1))]
    