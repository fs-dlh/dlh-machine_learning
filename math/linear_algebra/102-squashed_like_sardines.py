#!/usr/bin/env python3
"""Module for concatenating two matrices along a given axis."""


def cat_matrices(mat1, mat2, axis=0):
    """ Concatenate two matrices along a specified axis.

    Args:
        mat1 (list of int/float or nested list): First matrix.
        mat2 (list of int/float or nested list): Second matrix.
        axis (int): Axis along which to concatenate.

    Returns:
        list or None: New matrix resulting from concatenation, or None if
        the matrices cannot be concatenated (shape mismatch or invalid axis).
    """

    def get_shape(m):
        """Recursively determine the shape of a nested list (matrix)."""

        shape = []
        while isinstance(m, list):
            shape.append(len(m))
            if not m:
                break
            m = m[0]
        return shape

    shape1 = get_shape(mat1)
    shape2 = get_shape(mat2)

    if len(shape1) != len(shape2):
        return None
    if axis < 0 or axis >= len(shape1):
        return None

    for d in range(len(shape1)):
        if d != axis and shape1[d] != shape2[d]:
            return None

    def deep_copy(m):
        """Recursively create a deep copy of a nested list (matrix)."""

        if isinstance(m, list):
            return [deep_copy(x) for x in m]
        return m

    def _concat(m1, m2, cur_axis):
        """Recursively concatenate two matrices along the specified axis."""

        if cur_axis == 0:
            return deep_copy(m1) + deep_copy(m2)
        return [_concat(m1[i], m2[i], cur_axis - 1) for i in range(len(m1))]

    return _concat(mat1, mat2, axis)
