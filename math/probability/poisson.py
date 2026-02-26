#!/usr/bin/env python3
"""Module for Poisson distribution"""


class Poisson:
    """Represents a Poisson distribution"""

    e = 2.7182818284590452353602874713527

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
        k = int(k)
        if k < 0:
            return 0

        # Calculate e^(-lambtha) using the constant e
        e_neg_lambtha = self.e ** (-self.lambtha)

        # Calculate lambtha^k
        lambtha_power_k = self.lambtha ** k

        # Calculate k!
        factorial_k = 1
        for i in range(1, k + 1):
            factorial_k *= i

        # PMF = (e^(-lambtha) * lambtha^k) / k!
        return (e_neg_lambtha * lambtha_power_k) / factorial_k
