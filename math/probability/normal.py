#!/usr/bin/env python3
"""Module for Normal distribution"""


class Normal:
    """Represents a normal distribution"""

    def __init__(self, data=None, mean=0., stddev=1.):
        """
        Initialize Normal distribution
        Args:
            data: list of data to estimate the distribution
            mean: mean of the distribution
            stddev: standard deviation of the distribution
        """
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.stddev = stddev
            self.mean = mean
        elif not isinstance(data, list):
            raise TypeError("data must be a list")
        elif len(data) < 2:
            raise ValueError("data must contain multiple values")
        else:
            self.mean = sum(data) / len(data)
            stddev = 0
            for i in data:
                stddev += (self.mean - i) ** 2
            self.stddev = (stddev / len(data)) ** (1 / 2)

    def z_score(self, x):
        """Calculate z score"""
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """Calculate x from z score"""
        return z * self.stddev + self.mean

    def pdf(self, x):
        """Calculate PDF value for given x"""
        return (1 / ((2 * 3.1415926536) ** (1 / 2) * self.stddev)
                * 2.7182818285 ** ((-(x - self.mean) ** 2) /
                (2 * self.stddev ** 2))
                )

    def cdf(self, x):
        """Calculate CDF value for given x"""
        pi = 3.1415926536
        x = (x - self.mean) / (self.stddev * 2 ** (1 / 2))
        erf = (x - x ** 3 / 3 + x ** 5 / 10 - x ** 7 / 42 + x ** 9 / 216)
        erf = 2 * erf / (pi ** (1 / 2))
        return (1 / 2) * (1 + erf)
