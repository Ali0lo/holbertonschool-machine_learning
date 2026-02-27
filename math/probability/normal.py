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
        """
        Calculates the z-score of a given x-value
        Args:
            x: the x-value
        Returns:
            z-score of x
        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """
        Calculates the x-value of a given z-score
        Args:
            z: the z-score
        Returns:
            x-value of z
        """
        return z * self.stddev + self.mean
