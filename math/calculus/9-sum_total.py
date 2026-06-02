#!/usr/bin/env python3
""" the sum of all squared natural numbers up to n."""


def summation_i_squared(n):
    """
    Calculates the sum of squares from 1 to n using the closed-form formula.

    Args:
        n: The stopping condition (positive integer).

    Returns:
        Integer sum of squares if n is a valid positive integer, or None.
    """

    if isinstance(n, int) and n >= 1:
        return n * n + summation_i_squared(n-1)
    if n == 1:
        return 1
    return None
