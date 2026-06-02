#!/usr/bin/env python3
""" Calculates the derivative of a polynomial."""


def poly_derivative(poly):
    """    Calculates the derivative of a polynomial.

    Args:
        poly: A list of coefficients representing the polynomial.

    Returns: A list of coefficients as the derivative of the polynomial."""


    if not isinstance(poly, list):
        return None

    if len(poly) == 1:
        return [0]

    derivative = []
    for i in range(1, len(poly)):
        derivative.append(i * poly[i])

    return derivative
