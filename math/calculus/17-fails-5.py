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

    if poly == []:
        return [C]

    result = [C]

    for i, coeff in enumerate(poly):

        if coeff % (i + 1) == 0:
            result.append(coeff // (i + 1))
        else:
            result.append(coeff / (i + 1))

    while len(result) > 1 and result[-1] == 0:
        result.pop()

    return result

