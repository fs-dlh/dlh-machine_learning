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
    for coeff in poly:
        if not isinstance(coeff, (int, float)):
            return None

    integral = [0]
    for i, coeff in enumerate(poly):
        integral.append(coeff / (i + 1))
    return integral
