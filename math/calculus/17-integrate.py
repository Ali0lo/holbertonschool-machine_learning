#!/usr/bin/env python3
"""Module for polynomial integration"""


def poly_integral(poly, C=0):
    """
    Calculates the integral of a polynomial.
    
    Args:
        poly: list of coefficients representing a polynomial
              index represents the power of x
        C: integration constant (default 0)
    
    Returns:
        New list of coefficients representing the integral,
        or None if inputs are invalid
    """
    # Validate inputs
    if not isinstance(poly, list) or len(poly) == 0:
        return None
    
    if not isinstance(C, (int, float)):
        return None
    
    # Check all elements in poly are numbers
    for coef in poly:
        if not isinstance(coef, (int, float)):
            return None
    
    # Special case: if polynomial is [0], integral is [C]
    if poly == [0]:
        return [C]
    
    # Calculate integral
    # Start with the constant term C
    integral = [C]
    
    # For each coefficient at index i (representing x^i),
    # the integral is coef/(i+1) * x^(i+1)
    for i, coef in enumerate(poly):
        new_coef = coef / (i + 1)
        
        # Convert to integer if it's a whole number
        if new_coef == int(new_coef):
            new_coef = int(new_coef)
        
        integral.append(new_coef)
    
    # Remove trailing zeros to make list as small as possible
    while len(integral) > 1 and integral[-1] == 0:
        integral.pop()
    
    return integral
