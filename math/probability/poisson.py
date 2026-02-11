#!/usr/bin/env python3
"""Module for Poisson distribution"""


class Poisson:
    """Represents a Poisson distribution"""

    def __init__(self, data=None, lambtha=1.):
        """
        Initialize Poisson distribution

        Args:
            data: list of data to estimate distribution
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
            self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """
        Calculates the value of the PMF for a given number of successes

        Args:
            k: number of successes

        Returns:
            PMF value for k
        """
        k = int(k)
        if k < 0:
            return 0

        # Calculate factorial of k
        factorial = 1
        for i in range(1, k + 1):
            factorial *= i

        # Calculate e^(-lambtha) using Taylor series for better precision
        x = -self.lambtha
        exp_x = 1
        term = 1
        for i in range(1, 150):
            term *= x / i
            exp_x += term
            if abs(term) < 1e-15:
                break

        # PMF formula: (lambtha^k * e^(-lambtha)) / k!
        pmf_value = (self.lambtha ** k * exp_x) / factorial

        return pmf_value
