#!/usr/bin/env python3
"""Module for polynomial derivative"""


def poly_derivative(poly):
    """
    Calculates the derivative of a polynomial.

    Args:
        poly: list of coefficients representing a polynomial
              index represents the power of x

    Returns:
        New list of coefficients representing the derivative,
        or None if input is invalid
    """
    # Validate input
    if not isinstance(poly, list) or len(poly) == 0:
        return None

    # Check all elements in poly are numbers
    for coef in poly:
        if not isinstance(coef, (int, float)):
            return None

    # Special case: constant polynomial (only one coefficient)
    if len(poly) == 1:
        return [0]

    # Calculate derivative
    # For each coefficient at index i (representing x^i),
    # the derivative is i * coef * x^(i-1)
    derivative = []

    for i in range(1, len(poly)):
        new_coef = poly[i] * i

        # Convert to integer if it's a whole number
        if new_coef == int(new_coef):
            new_coef = int(new_coef)

        derivative.append(new_coef)

    # Remove trailing zeros to make list as small as possible
    while len(derivative) > 1 and derivative[-1] == 0:
        derivative.pop()

    # If all coefficients are 0, return [0]
    if len(derivative) == 0 or all(c == 0 for c in derivative):
        return [0]

    return derivative
