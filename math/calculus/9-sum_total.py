#!/usr/bin/env python3
""" Module that contains the function summation_i_squared(n) that calculates
the sum of all squared natural numbers up to n.
"""


def summation_i_squared(n):
    if not isinstance(n, int) :
        return None
    if n > 0:
        return n**2 + summation_i_squared(n-1)
    else
        return 0
    