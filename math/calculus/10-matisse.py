#!/usr/bin/env python3

def poly_derivative(poly):
    """
    This function calculates the derivative of a polynomial represented
    by a list of coefficients.

    Arguments:
    poly -- A list of coefficients representing a polynomial

    Returns:
    A list of coefficients representing the derivative of the polynomial,
    or None if the input is invalid.
    """
    if not isinstance(poly, list) or len(poly) == 0 or not all(
        isinstance(coef, int) for coef in poly
    ):
        return None
    
    # Calculate the derivative using the power rule
    derivative = []
    for power, coef in enumerate(poly):
        if power == 0:
            continue  # The derivative of the constant term (x^0) is 0
        derivative.append(coef * power)
    
    # If the derivative is an empty list, return [0] (this happens when it's a constant polynomial)
    if not derivative:
        return [0]
    
    return derivative
