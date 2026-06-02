#!/usr/bin/env python3
""" Calculates the integral of a polynomial."""


def poly_integral(poly, C=0):
    """    Calculates the integral of a polynomial.

    Args:
        poly: A list of coefficients representing the polynomial.

    Returns:
        A list of coefficients as the integral of the polynomial."""

    if not isinstance(poly, list) or not isinstance(C, int) or not poly:
        return None
    for coeff in poly:
        if not isinstance(coeff, (int, float)):
            return None

    integral = [C]

    for i, coeff in enumerate(poly):
        denom = i + 1
        if isinstance(coeff, float) and coeff.is_integer():
            coeff = int(coeff)

        if isinstance(coeff, int) and coeff % denom == 0:
            new_coeff = coeff // denom
        else:
            new_coeff = coeff / denom
            if isinstance(new_coeff, float) and new_coeff.is_integer():
                new_coeff = int(new_coeff)

        integral.append(new_coeff)

    while integral and integral[-1] == 0:
        integral.pop()
    return integral
