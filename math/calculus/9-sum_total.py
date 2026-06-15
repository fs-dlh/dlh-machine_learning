#!/usr/bin/env python3
""" the sum of all squared natural numbers up to n."""


def summation_i_squared(n):
    """
    Calculates the sum of squares from 1 to n.

    Args:
        n: The stopping condition (positive integer).

    Returns:
        Integer sum of squares if n is a valid positive integer, or None.
    """

    if not isinstance(n, int) or n < 1:
        return None

    return n * (n + 1) * (2 * n + 1) // 6
