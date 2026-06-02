#!/usr/bin/env python3
""" Calculates the derivative of a polynomial."""


def poly_derivative(poly):
    """    Calculates the derivative of a polynomial.

    Args:
        poly: A list of coefficients representing the polynomial.

    Returns:
        A list of coefficients as the derivative of the polynomial."""

    if not isinstance(poly, list) or not poly:
        return None
    for coeff in poly:
        if not isinstance(coeff, (int, float)):
            return None

    if len(poly) == 1:
        return [0]

    derivative = []
    while len(derivative) > 1 and derivative[-1] == 0:
        derivative.pop()

    return derivative
