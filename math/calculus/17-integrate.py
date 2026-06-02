#!/usr/bin/env python3
""" Calculates the integral of a polynomial."""


def poly_integral(poly, C=0):
    """    Calculates the integral of a polynomial.

    Args:
        poly: A list of coefficients representing the polynomial.

    Returns:
        A list of coefficients as the integral of the polynomial."""

    if not isinstance(poly, list) or not poly:
        return None
    if not isinstance(C, int):
        return None
    for coeff in poly:
        if not isinstance(coeff, (int, float)):
            return None

    integral = [C]
    for i, coeff in enumerate(poly):
        if coeff % (i+1) == 0:
            integral.append(coeff // (i + 1))
        else:
            integral.append(coeff / (i + 1))

    while len(integral) > 0 and integral[-1] == 0:
        integral.pop()

    return integral
