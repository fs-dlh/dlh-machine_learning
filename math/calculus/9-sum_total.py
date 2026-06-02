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
    if not isinstance(n, int):
        return None
    if n > 0:
        return n**2 + summation_i_squared(n-1)
    else:
        return 0
