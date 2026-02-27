#!/usr/bin/env python3
""" Advanced task 1 """
from scipy import special


def posterior(x, n, p1, p2):
    """ Function for calculating posterior probability """

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
    elif not isinstance(p1, float) or not 0 <= p1 <= 1:
        raise ValueError('p1 must be a float in the range [0, 1]')
    elif not isinstance(p2, float) or not 0 <= p2 <= 1:
        raise ValueError('p2 must be a float in the range [0, 1]')
    elif p2 <= p1:
        raise ValueError('p2 must be greater than p1')
    alpha = x + 1
    beta = n - x + 1
    return special.btdtr(alpha, beta, p2) - special.btdtr(alpha, beta, p1)
