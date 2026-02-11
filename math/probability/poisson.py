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

        # Use natural logarithm approach for better precision
        # ln(PMF) = -lambtha + k*ln(lambtha) - ln(k!)
        # Then PMF = e^(ln(PMF))
        
        # Calculate e using series expansion with high precision
        e = 2.7182818284590452353602874713527

        # PMF formula: (lambtha^k * e^(-lambtha)) / k!
        pmf_value = ((self.lambtha ** k) * (e ** (-self.lambtha))) / factorial

        return pmf_value
