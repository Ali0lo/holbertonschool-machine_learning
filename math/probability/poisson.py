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
        k = int(k)
        if k < 0:
            return 0

        # Use log space for better precision
        # log(PMF) = -lambtha + k*log(lambtha) - log(k!)
        log_pmf = -self.lambtha
        
        if k > 0:
            log_pmf += k * self._log(self.lambtha)
            # Subtract log(k!)
            for i in range(1, k + 1):
                log_pmf -= self._log(i)
        
        # Convert back from log space
        return self._exp(log_pmf)

    def _log(self, x):
        """Calculate natural logarithm using series expansion"""
        if x <= 0:
            return float('-inf')
        
        # For x close to 1, use ln(x) = 2 * sum((z^(2n+1))/(2n+1))
        # where z = (x-1)/(x+1)
        z = (x - 1) / (x + 1)
        z_squared = z * z
        result = z
        term = z
        n = 1
        
        while abs(term) > 1e-15 and n < 1000:
            term *= z_squared
            result += term / (2 * n + 1)
            n += 1
        
        return 2 * result

    def _exp(self, x):
        """Calculate e^x using Taylor series"""
        result = 1
        term = 1
        n = 1
        
        while abs(term) > 1e-15 and n < 1000:
            term *= x / n
            result += term
            n += 1
        
        return result
