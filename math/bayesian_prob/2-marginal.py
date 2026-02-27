#!/usr/bin/env python3
""" This module will define a function named marginal """
import numpy as np


def marginal(x, n, P, Pr):
    """ Function for calculating marginal
    probability of obtaining the data
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
    elif type(Pr) is not np.ndarray or Pr.shape != P.shape:
        raise TypeError('Pr must be a numpy.ndarray with the same shape as P')
    elif np.max(P) > 1 or np.min(P) < 0:
        raise ValueError('All values in P must be in the range [0, 1]')
    elif np.max(Pr) > 1 or np.min(Pr) < 0:
        raise ValueError('All values in Pr must be in the range [0, 1]')
    elif not np.isclose(np.sum(Pr), 1):
        raise ValueError('Pr must sum to 1')
    C = factorial(n) / (factorial(x) * factorial(n - x))
    likelihood = C * (P ** x) * ((1 - P) ** (n - x))
    return np.sum(likelihood * Pr)
