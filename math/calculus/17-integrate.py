#!/usr/bin/env python3
""" Calculates the integral of a polynomial."""


def poly_integral(poly):
    """    Calculates the integral of a polynomial.

    Args:
        poly: A list of coefficients representing the polynomial.

    Returns:
        A list of coefficients as the integral of the polynomial."""

    if isinstance(poly, list):
        if len(poly) == 1:
            return [0]

        integral = [0]
        for i in range(len(poly)):
            integral.append(poly[i] / (i + 1))

        return integral

    return None
