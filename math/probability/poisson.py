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

        # PMF formula: (lambtha^k * e^(-lambtha)) / k!
        # Using log for better precision: log(PMF) = k*log(lambtha) - lambtha - log(k!)
        
        # Calculate log(k!)
        log_factorial = 0
        for i in range(1, k + 1):
            log_factorial += self._ln(i)
        
        # log(PMF) = k*log(lambtha) - lambtha - log(k!)
        log_pmf = k * self._ln(self.lambtha) - self.lambtha - log_factorial
        
        # PMF = e^(log_pmf)
        pmf_value = self._exp(log_pmf)
        
        return pmf_value
    
    def _ln(self, x):
        """Natural logarithm using series expansion"""
        if x <= 0:
            return float('-inf')
        
        # For x close to 1, use ln(x) = 2 * sum((((x-1)/(x+1))^(2n+1))/(2n+1))
        n = 1000
        y = (x - 1) / (x + 1)
        y2 = y * y
        result = 0
        power = y
        for i in range(n):
            result += power / (2 * i + 1)
            power *= y2
        return 2 * result
    
    def _exp(self, x):
        """Exponential function using series expansion"""
        result = 1
        term = 1
        for i in range(1, 150):
            term *= x / i
            result += term
            if abs(term) < 1e-16:
                break
        return result
    def pmf(self, k):
        """Calculates the value of the PMF for a given number of successes
        Args:
            k: number of successes
        Returns:
            PMF value for k, or 0 if k is out of range
        """
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0
        import math
        e = math.e
        factorial = math.factorial(k)
        numerator = (e ** (-self.lambtha)) * (self.lambtha ** k)
        return numerator / factorial
