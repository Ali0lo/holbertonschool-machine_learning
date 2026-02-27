#!/usr/bin/env python3
"""Module for Exponential distribution"""


class Exponential:
    """Represents an exponential distribution"""

    def __init__(self, data=None, lambtha=1.):
        """
        Initialize Exponential distribution
        Args:
            data: list of data to estimate the distribution
            lambtha: expected number of occurrences in given time frame
        """
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            # For exponential distribution: lambtha = 1 / mean
            mean = sum(data) / len(data)
            self.lambtha = 1 / mean
