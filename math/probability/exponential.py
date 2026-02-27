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
            self.lambtha = lambtha
        elif not isinstance(data, list):
            raise TypeError("data must be a list")
        elif len(data) < 2:
            raise ValueError("data must contain multiple values")
        else:
            self.lambtha = 1 / (sum(data) / len(data))

    def pdf(self, x):
        """
        Calculates the value of the PDF for a given time period
        Args:
            x: time period
        Returns:
            PDF value for x, or 0 if x is out of range
        """
        if x < 0:
            return 0
        return self.lambtha * 2.7182818285 ** (-self.lambtha * x)

    def cdf(self, x):
        """
        Calculates the value of the CDF for a given time period
        Args:
            x: time period
        Returns:
            CDF value for x, or 0 if x is out of range
        """
        if x < 0:
            return 0
        return 1 - 2.7182818285 ** (-self.lambtha * x)
