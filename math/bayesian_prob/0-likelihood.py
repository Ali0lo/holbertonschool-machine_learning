#!/usr/bin/env python3
""" This module will define a function namd likelihood """
import numpy as np


def likelihood(x, n, P):
    """ Function for calculating likelihood of
    obtaining data given probability of side effects
    """

    def factorial(n):
        """ Helper func for calculating factorial """
        h = 1
        for i in range(2, n + 1):
            h = h * i
        return h
    if not isinstance(n, int) or n <= 0:
        raise ValueError('n must be a positive integer')
    elif not isinstance(x, int) or x < 0:
        raise ValueError(
            'x must be an integer that is greater than or equal to 0')
    elif x > n:
        raise ValueError('x cannot be greater than n')
    elif type(P) is not np.ndarray or P.ndim != 1:
        raise TypeError('P must be a 1D numpy.ndarray')
    elif np.max(P) > 1 or np.min(P) < 0:
        raise ValueError('All values in P must be in the range [0, 1]')
    C = factorial(n) / (factorial(x) * factorial(n - x))
    return C * (P ** x) * ((1 - P) ** (n - x))
