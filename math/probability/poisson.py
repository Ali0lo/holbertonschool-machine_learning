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
            PMF value for k, or 0 if k is out of range
        """
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0

        # Calculate e^(-lambtha)
        e_neg_lambtha = self._e_power(-self.lambtha)

        # Calculate lambtha^k
        lambtha_power_k = self.lambtha ** k

        # Calculate k!
        factorial_k = 1
        for i in range(1, k + 1):
            factorial_k *= i

        # PMF = (e^(-lambtha) * lambtha^k) / k!
        return (e_neg_lambtha * lambtha_power_k) / factorial_k

    def _e_power(self, x):
        """
        Calculate e^x using Taylor series
        e^x = 1 + x + x^2/2! + x^3/3! + ...
        """
        result = 1
        term = 1
        for i in range(1, 100):
            term *= x / i
            result += term
            if abs(term) < 1e-15:
                break
        return result
