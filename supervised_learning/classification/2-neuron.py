#!/usr/bin/env python3
"""Module that defines a single Neuron performing binary classification."""
import numpy as np


class Neuron:
    """Class that defines a single neuron performing binary classification."""

    def __init__(self, nx):
        """Initialize Neuron."""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Getter for weights vector."""
        return self.__W

    @property
    def b(self):
        """Getter for bias."""
        return self.__b

    @property
    def A(self):
        """Getter for activated output."""
        return self.__A

    def forward_prop(self, X):
        """Calculates the forward propagation of the neuron.

        Args:
            X (numpy.ndarray): Input data with shape (nx, m).

        Returns:
            numpy.ndarray: The activated output __A.
        """
        Z = np.dot(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A
