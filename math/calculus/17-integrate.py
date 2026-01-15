#!/usr/bin/env python3

def poly_integral(poly, C=0):
    """
    This function calculates the integral of a polynomial represented
    by a list of coefficients.

    Arguments:
    poly -- A list of coefficients representing a polynomial
    C -- The constant of integration (default is 0)

    Returns:
    A list of coefficients representing the integral of the polynomial,
    or None if the input is invalid.
    """
    # Input validation
    if not isinstance(poly, list) or not all(isinstance(coef, (int, float)) for coef in poly) or not isinstance(C, (int, float)):
        return None

    # If the polynomial is a constant zero, return [0]
    if poly == [0]:
        return [0]
    
    # Initialize result with the constant of integration
    result = [C]
    
    # Calculate the integral of each term in the polynomial
    for power, coef in enumerate(poly):
        if coef != 0:  # Skip zero coefficients as they don't affect the integral
            integral_coef = coef / (power + 1)
            # Convert whole number floats to integers
            if integral_coef.is_integer():
                result.append(int(integral_coef))
            else:
                result.append(integral_coef)
        else:
            # Ensure missing powers (zeros) are included explicitly in the list
            result.append(0)
    
    return result
